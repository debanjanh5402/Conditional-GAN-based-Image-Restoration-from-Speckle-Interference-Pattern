import torch

def _normalize_speckle(x:torch.Tensor, gamma:float) -> torch.Tensor:
    x_norm = (x - x.min()) / (x.max() - x.min())
    x_gamma = x_norm**gamma
    x_norm = (2.0*x_gamma - 1.0)
    x_norm = torch.clamp(x_norm, min=-1.0, max=+1.0)
    return x_norm.to(torch.float32)

def _normalize_img(x:torch.Tensor) -> torch.Tensor: 
    x_norm = x / 255.0
    x_norm = 2*x_norm - 1
    x_norm = torch.clamp(x_norm, min=-1.0, max=+1.0)
    return x_norm.to(torch.float32)