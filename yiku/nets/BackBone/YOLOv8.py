import sys
from torch import nn
import os
try:
    from ultralytics.models import YOLO
except ImportError as e:
    pass
from yiku.PATH import WTS_STORAGE_DIR
class YOLOv8_backbone(nn.Module):   #### Not Recommand (Low mIOU)
    def __init__(self, *args, **kwargs):
        if "model" not in kwargs.keys():
            model = "yolov8s-cls.pt"
        else:
            model = kwargs["model"]
            kwargs.__delitem__("model")
        super().__init__(*args, **kwargs)
        m = YOLO(os.path.join(WTS_STORAGE_DIR,model))
        self.__model = m.model.model
        arch=model[6]
        self.__dict__.update(**{"s": {"feature_ch": 512, "low_ch": 128},
                                "m": {"feature_ch": 768, "low_ch": 192}
                                }[arch])

    def forward(self, x):
        for i, n in enumerate(self.__model):
            x = n(x)
            if i == len(self.__model) - 1 - 2:
                func = nn.Upsample(None, 2, 'nearest')
                y1 = func(x)
                break # Get the featuremap
            if i == 4:
                func = nn.Upsample(None, 2, 'nearest')
                low = func(x) # low feature
        return low, y1

def yolov8s(pretrained=True, **kwargs):
    return YOLOv8_backbone(model="yolov8s-cls.pt")

def yolov8m(pretrained=True, **kwargs):
    return YOLOv8_backbone(model="yolov8m-cls.pt")