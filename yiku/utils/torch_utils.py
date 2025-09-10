import torch
def get_mem(device="cuda",fraction=False):
    mem=0.0
    total=0.0
    if device == "cuda":
        mem = torch.cuda.memory_reserved() / 1E9
        if fraction:
            total = torch.cuda.get_device_properties(device=device).total_memory
    elif device == "xpu":
        mem = torch.xpu.memory_allocated() / 1E9
        if fraction:
            total = torch.xpu.get_device_properties(device=device).total_memory
    return mem, total