import datetime
import time

import numpy as np
import torch

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from yiku.nets.model.Labs.labs import Labs
from utils.check import check_gpu
from yiku.nets.training_utils import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import LossHistory, EvalCallback
from yiku.data.AutoLoaderSelector import auto_ds_sel, deeplab_dataset_collate
from utils.download import download_weights
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch
from pathlib import Path

from rich import print,emoji
from rich.console import Console
# except ImportError:
#     import warnings
#
#     warnings.filterwarnings('ignore', message="Setuptools is replacing distutils.", category=UserWarning)
#     from pip._vendor.rich import print,emoji
#     from pip._vendor.rich.console import Console
'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。
   如果格式有误，参考：https://github.com/bubbliiiing/segmentation-format-fix

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''

import configparser
import argparse
import sys,os
sys.path.append(os.getcwd())
from config import LabConfig,UNetConfig,PSPNetConfig,SegFormerConfig,HarDNetConfig
import signal
import platform
try:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://d5afa0d557b2491a2aa9e34e1c803cf2@o4507524861853696.ingest.de.sentry.io/4507524873977936",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
        traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
)
except ImportError as e:
    print("远程日志已经禁用")
def signal_handler(signal, frame):
    print("操作取消 Operation Cancelled")
    if platform.system().lower()=="linux":
        print("\033[?25h")
    sys.exit(0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',default="config.ini",help="Path of config file")
    parser.add_argument('-r', '--resume', action="store_true",help="Resume training from best.pth")
    parser.add_argument('-p', '--pretrain', default="True",help="Pretrain,<True| False | path of pth>,default true")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    if os.path.exists(args.config):
        config.read(args.config,encoding='utf-8')
    else:
        config["base"]={}
        config["advance"] = {}
    CONFIG_DIR=os.path.dirname(os.path.abspath(args.config))
    ARCH = config["base"].get("arch", "lab")

    cfg_cls={"unet":UNetConfig,
             "pspnet":PSPNetConfig,
             "segformer":SegFormerConfig,
             "hardnet":HarDNetConfig,
             "lab":LabConfig}
    if ARCH.lower() in cfg_cls.keys():
        hyp_cfg = cfg_cls[ARCH.lower()]()
    else:
        hyp_cfg = LabConfig()

    freeze_batch_size = config["base"].getint("frozen_batch-size",10)
    freeze_epoch = config["base"].getint("frozen_epoch",75)
    unfreeze_batch_size = config["base"].getint("unfrozen_batch-size",6)
    unfreeze_epoch = config["base"].getint("unfrozen_epoch",100)
    IMGSZ= config["base"].getint("image_size",512)
    DATASET_PATH = config["base"].get("dataset_path", 'VOCdevkit')
    if not os.path.isabs(DATASET_PATH):
        DATASET_PATH = os.path.join(CONFIG_DIR,DATASET_PATH)
    SAVE_PATH = config["base"].get("save_path", "save")
    if not os.path.isabs(SAVE_PATH):
        SAVE_PATH=os.path.join(CONFIG_DIR,SAVE_PATH)
    backbone = config["base"].get("backbone","hgnetv2l")
    num_classes = config["base"].getint("num_classes",21)
    pp = config["base"].get("header", "transformer")
    num_workers = config["base"].getint("num_workers",4)
    fp16 = config["base"].getboolean("fp16",True)
    if "advance" not in config:
        config["advance"] = {}

    CUSTOM_DS=config["advance"].get("custom_datasets", None)

    DOWNSAMPLE_FACTOR = config["advance"].getint("downsample_factor",16)
    if num_classes<10:
        DICE_LOSS=True
    elif freeze_batch_size>10:
        DICE_LOSS=True
    else:
        DICE_LOSS=False 
    DICE_LOSS= config["advance"].getboolean("dice_loss",DICE_LOSS)
    focal_loss = config["advance"].getboolean("focal_loss", False)
    RESUME = args.resume

    #HyperParam Config
    init_lr = config["advance"].getfloat("init_lr", hyp_cfg.init_lr)
    MIN_LR_MULTIPLY = config["advance"].getfloat("min_lr_mutliply", hyp_cfg.min_lr_mutliply)
    optimizer_type = config["advance"].get("optimizer_type", hyp_cfg.optimizer_type)
    momentum = config["advance"].getfloat("momentum", hyp_cfg.momentum)
    weight_decay= config["advance"].getfloat("weight_decay", hyp_cfg.weight_decay)
    lr_decay_type = config["advance"].getfloat("lr_decay_type", hyp_cfg.lr_decay_type)

    # Aug
    jitter =  0.3
    flip =  0.5
    blur =  0.25
    if config.has_section("augmentation"):
        aug = config["augmentation"].getboolean("enable", True)
        jitter=config["augmentation"].getfloat("jitter", 0.3 if aug else 0)
        flip=config["augmentation"].getfloat("flip", 0.5 if aug else 0)
        blur=config["augmentation"].getfloat("blur", 0.25 if aug else 0)
    signal.signal(signal.SIGINT, signal_handler)
    if CUSTOM_DS:
        DS_File, DS_Class = CUSTOM_DS.split(":")
        if DS_File and Path(DS_File).is_file() and Path(DS_File).suffix==".py":

            print("[red bold]:warning: :warning: :warning: 你正在导入自定义的数据加载器，这可能包含恶意代码")
            print("[red bold]:warning: :warning: :warning: You are importing a custom DatasetsLoader, which may contain malicious code")

            print(
            "[yellow bold]:warning: :warning: :warning: 将会等待5秒，如果需要取消操作 请ctrl+c终止程序")
            print(
            "[yellow bold]:warning: :warning: :warning: Will waitting 5s，if you need cancel，press Ctrl+C")
            time.sleep(5)
            print("-------")


    # ---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    _c=check_gpu()
    cuda = _c["device"]=="cuda"
    xpu = _c["device"]=="xpu"
    if cuda:
        import torch.backends.cudnn as cudnn
    else:
        cudnn=None
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = cuda and _c["amp"] and fp16
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = args.pretrain
    if not os.path.isfile(pretrained):
        pretrained= str(pretrained).lower()=="true"
        model_path = ""
    else:
        model_path = str(pretrained)
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#


    downsample_factor = DOWNSAMPLE_FACTOR
    input_shape = [IMGSZ, IMGSZ]
    if RESUME:
        meta = torch.load(os.path.join(SAVE_PATH, "last.meta"))
        init_epoch = meta["curr_epoch"]
        model_path=os.path.join(SAVE_PATH, "last.pth")
        pretrained=False
    else:
        init_epoch = 0

    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    freeze_Train = True

    #数据增强
    aug_blur=True
    aug_hsv=False

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    min_lr = init_lr * MIN_LR_MULTIPLY
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 5
    save_dir = SAVE_PATH
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 5

    VOCdevkit_path = DATASET_PATH
    dice_loss = DICE_LOSS

    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------------------#

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = getattr(torch,_c["device"]).device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device(_c["device"], local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device(_c["device"] if getattr(torch,_c["device"]).is_available() else 'cpu')
        local_rank = 0

    # ----------------------------------------------------#
    #   下载预训练权重
    # ----------------------------------------------------#

    if pretrained and isinstance(pretrained,bool):
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)
    else:
        pretrained=False

    if ARCH.lower()=="unet":
        from nets.model.UNet import UNet
        model=UNet(num_classes=num_classes,pretrained=pretrained,backbone=backbone)
    elif ARCH.lower()=="pspnet":
        from nets.model.PSPNet import pspnet
        model=pspnet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                 pretrained=pretrained)
    elif ARCH.lower()=="segformer":
        from nets.model.SegFormer import SegFormer
        model=SegFormer(num_classes=num_classes, backbone=backbone,
                 pretrained=pretrained)
    elif ARCH.lower()=="hardnet":
        from yiku.nets.model.hardnet import hardnet
        model=hardnet(num_classes=num_classes,pretrained=pretrained)
    elif ARCH.lower()=="unetpp":
        from yiku.nets.model.UNetplusplus.UnetPP import UNetPlusPlus
        model = UNetPlusPlus(num_classes=num_classes, pretrained=pretrained)
    else:
        model = Labs(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                 pretrained=pretrained, header=pp,img_sz=(IMGSZ,IMGSZ))
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n[bold blue] head部分没有载入是正常现象，Backbone部分没有载入是错误的。")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
        if RESUME:
            loss_history.val_loss=meta["val_his_loss"]
            loss_history.losses=meta["his_loss"]
            loss_history.miou = meta["miou"]
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    if fp16:
        from torch.amp import GradScaler
        scaler = GradScaler(device=device.type)
    else:
        scaler = None

    # if ARCH.lower()=="lab" and not os.path.isfile(Path(DATASET_PATH)/"fm"):
    #     from yiku.utils.get_featuremap import get_featureMap
    #     get_featureMap(m=model.eval(),mode="val",ds_dir=DATASET_PATH,sz=IMGSZ,bb=backbone)
    #     get_featureMap(m=model.eval(), mode="train", ds_dir=DATASET_PATH, sz=IMGSZ,bb=backbone)
    #     with open(Path(DATASET_PATH)/"fm", 'w') as fp:pass


    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    if CUSTOM_DS:
        DS_File, DS_Class = CUSTOM_DS.split(":")
    else:
        DS_File, DS_Class = False, ""
    if DS_File and Path(DS_File).is_file() and Path(DS_File).suffix == ".py":
        from importlib.util import spec_from_file_location
        from importlib.util import module_from_spec
        spec = spec_from_file_location(Path(DS_File).name, DS_File)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        ds_cls = getattr(module, DS_Class)
        train_dataset = ds_cls(input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = ds_cls(input_shape, num_classes, False, VOCdevkit_path)
    else:
        train_dataset = auto_ds_sel( input_shape, num_classes, True, VOCdevkit_path,jitter_prop=jitter,flip_prop=flip,blur_prop=blur)
        val_dataset = auto_ds_sel( input_shape, num_classes, False, VOCdevkit_path)
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    if not aug_blur:
        train_dataset.blur = None
    if not aug_hsv:
        train_dataset.hsv_jitter = None

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=init_epoch, Freeze_Epoch=freeze_epoch, UnFreeze_Epoch=unfreeze_epoch,
            Freeze_batch_size=freeze_batch_size, Unfreeze_batch_size=unfreeze_batch_size, Freeze_Train=freeze_Train,
            init_lr=init_lr, min_lr=min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,arch=ARCH,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // unfreeze_batch_size * unfreeze_epoch
        if total_step <= wanted_step:
            if num_train // unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // unfreeze_batch_size) + 1
            print("\n:warning: 使用%s优化器时，建议将训练总步长设置到%d以上。:warning: " % (
                optimizer_type, wanted_step))
            print(
                ":warning:  本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。 :warning: " % (
                    num_train, unfreeze_batch_size, unfreeze_epoch, total_step))
            print(":warning:  由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。 :warning: " % (
                total_step, wanted_step, wanted_epoch))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if freeze_Train:
            if hasattr(model,"grad_backbone"):
                model.grad_backbone(False)
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = False

        # -------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = freeze_batch_size if freeze_Train else unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                             weight_decay=weight_decay),
            'adamw': optim.AdamW(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")




        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_dataset.annotation_lines, VOCdevkit_path, log_dir, cuda, \
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        import csv
        if not RESUME:
            with open(os.path.join(save_dir, "logs.csv"), 'w') as f:
                csv_write = csv.writer(f)
                csv_head = ["Epoch", "TotalLoss", "ValLoss","mIoU","mPA"]
                csv_write.writerow(csv_head)
        for epoch in range(init_epoch, unfreeze_epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= freeze_epoch and not UnFreeze_flag and freeze_Train:
                batch_size = unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                # -------------------------------------------------------------------#
                nbs = 16

                if ARCH.lower()=="unet":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                else:
                    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)

                if hasattr(model,"grad_backbone"):
                    model.grad_backbone(True)
                else:
                    for param in model.backbone.parameters():
                        param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, _c["device"], dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
            c = Console()
            c.rule()
            del c



            if distributed:
                dist.barrier()

        if local_rank == 0:
            pass
if __name__ == "__main__":
    main()