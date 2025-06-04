import argparse
import os
import configparser
import torch
class Opt():
    def __getitem__(self, item):
        return getattr(self, item)
    def __setitem__(self, key, value):
        if key=="kw":
            raise KeyError("Keyword can not be 'kw'")
        self.kw[key] = value
        setattr(self, key, value)
    def __init__(self):
        self.kw=dict()
        self.kw['latent']=100
        self.kw['dataset']="mvtec"
        self.kw["print_epoch_freq"]=1
        self.kw["save_epoch_freq"]=50
        self.kw["n_epochs"]=1000
        self.kw["epoch_count"]=0
        self.kw["n_epochs_decay"]=500
        self.kw["beta1"]=0.5
        self.kw["beta2"]=0.999
        self.kw["lr"]=0.0002
        self.kw["lr_policy"]='linear'
        self.kw["lr_decay_iters"]=50
        self.kw["batch_size"]=8
        self.kw["num_threads"]=0
        self.kw["no_dropout"]=False
        self.kw["model"]="aae"
        self.kw["gpu"]=True
        self.kw["init_type"]='normal'
        self.kw["init_gain"] = 0.02
        self.kw['no_dropout']=False
        self.kw["rotate"]=True
        self.kw["brightness"]=0
        self.kw["mode"]="train"
        for k,v in self.kw.items():
            setattr(self, k, v)

    def update(self,kw):
        self.kw.update(kw)
        for k,v in kw.items():
            setattr(self, k, v)

class TrainOptionParser:
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-c', '--config', default="config.ini", help="Path of config file")
        parser.add_argument('-r', '--resume', action="store_true", help="Resume training from best.pth")
        args = parser.parse_args()
        config = configparser.ConfigParser()
        self._cfg_path=os.path.dirname(os.path.abspath(args.config))
        self.resume = args.resume
        if os.path.exists(args.config):
            config.read(args.config, encoding='utf-8')
        else:
            config["base"] = {}
            config["advance"] = {}
        self._config = config
        self._cfg2opt()

    def _cfg2opt(self):
        self.opt=Opt()
        self.opt["model"]= self._config["base"].get("arch", "aae")
        self.opt["object"]= self._config["base"].get("object", "bottle")
        self.opt["channels"]= self._config["base"].getint("channels", 3)
        self.opt["cropsize"]=self.opt['img_size']= self._config["base"].getint("image_size", 512)
        self.opt['n_epochs'] = max(self._config["base"].getint("frozen_epoch", 750),self._config["base"].getint("unfrozen_epoch",1000))
        self.opt['batch_size']=min(self._config["base"].getint("frozen_batch-size",10),self._config["base"].getint("unfrozen_batch-size",10))
        SAVE_PATH = self._config["base"].get("save_path", "save")
        if not os.path.isabs(SAVE_PATH):
            SAVE_PATH = os.path.join(self._cfg_path, SAVE_PATH)
        self.opt['save_dir']=SAVE_PATH
        self.opt['num_workers'] = self._config["base"].getint("num_workers", 4)
        self.opt['gpu'] = torch.cuda.is_available() and self._config["base"].getboolean("cuda", True)
        dataset=self._config["base"].get("dataset_path", 'MVTec')
        if not os.path.isabs(dataset):
            dataset = os.path.join(self._cfg_path, dataset)
        self.opt['data_dir'] = dataset
        self.opt["lr"]= self._config["advance"].getfloat("init_lr", 0.0002)
        self.opt["lr_policy"]= self._config["advance"].getfloat("lr_decay_type", 'linear')



if __name__ == '__main__':
    opt = TrainOptionParser()
    opt._cfg2opt()
    print(opt.opt.save_dir)







