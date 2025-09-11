import warnings

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.transforms as transforms
from yiku.PATH import ASSETS
from PIL import Image
import torch

pre = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def check_gpu():
    d={"device":"cpu","amp":False}
    if hasattr(torch,"cuda") and torch.cuda.is_available():
        d.update({"device":"cuda"})
    elif hasattr(torch,"xpu") and torch.xpu.is_available():
        d["device"] = "xpu"
    if d["device"] == "cuda":
        d["amp"] = check_amp()
    elif d["device"] == "xpu":
        d["amp"] = True
        gpu = torch.xpu.get_device_name(torch.device("xpu"))
        for t in ("A770", "A580", "A380", "A310", "A40", "A50", "A60"):
            if t in gpu.upper():
                warnings.warn(
                    f"checks failed ❌. AMP training on {gpu} GPU may cause "
                    f"NaN losses or zero-mAP results, so AMP will be disabled during training."
                )
                d["amp"] = False
        if d["amp"]:
            d["amp"] = check_amp(d["device"])
    return d



def check_amp(device="cuda"):
    DEVICE = torch.device(device)
    m = mobilenet_v3_small(weights=None)
    m.load_state_dict(torch.load(ASSETS / 'mobilenet_v3_small.pth'))
    im_path = ASSETS / 'amp-test.jpg'  # image to check
    im = pre(Image.open(im_path))
    im = torch.unsqueeze(im, 0)
    im = im.to(DEVICE)
    m.eval()
    m.to(DEVICE)
    a = m(im)
    with torch.amp.autocast(device_type=DEVICE.type):
        b = m(im)
    del m
    del im
    torch.cuda.empty_cache()
    return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)
