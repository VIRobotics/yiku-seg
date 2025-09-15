import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from yiku.nets.BackBone import mobilenetv3s, mobilenetv3l, hgnetv2l, hgnetv2x, xception, mobilenetv2
from yiku.nets.Head import aspp, transformer
from yiku.utils.utils import fuse_conv_and_bn

# try:
#     from nets.BackBone import yolov8m, yolov8s
# except ImportError:
#     print("Use YOLOv8 Backbone need run pip install ultralytics==8.0.154")


class Labs(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16, header="aspp",
                 img_sz=(512, 512), **kwargs):
        super(Labs, self).__init__()
        header = header.lower()
        mod = sys.modules[__name__]
        backbone_list = ["mobilenetv2", "mobilenetv3s", "mobilenetv3l",
                         'hgnetv2l', 'hgnetv2x', 'yolov8m', 'yolov8s', 'xception']

        headlist = ["aspp", "transformer"]
        self.save_featuremap = None
        self.featuremap_result = []
        if (backbone not in backbone_list) and hasattr(mod, backbone):
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use {";".join(backbone_list)} .')
        backbone_func = getattr(mod, backbone)
        self.backbone = backbone_func(downsample_factor=downsample_factor, pretrained=pretrained)
        in_channels = self.backbone.feature_ch
        low_level_channels = self.backbone.low_ch
        if (header not in headlist) and hasattr(mod, header):
            raise ValueError(f'Unsupported Head - `{header}`, Use {";".join(headlist)} .')
        headerfunc = getattr(mod, header)
        print(f"{header} selected")
        H, W = img_sz
        self.header = headerfunc(H, W, num_classes, low_level_channels, in_channels,
                                 use_c2f=kwargs.get("use_c2f", False))

    def forward(self, x):
        low_level_features, y = self.backbone(x)
        if self.save_featuremap:
            self.featuremap_result = [low_level_features, y]
        ret = self.header(low_level_features, y)
        return ret

    def fuse(self):

        self.header = fuse_conv_and_bn(self.header)
        self.backbone = fuse_conv_and_bn(self.backbone)
