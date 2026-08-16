import torch

def _fromdynamic_to01(x:torch.Tensor):
    return ((x - x.min())/(x.max() - x.min())).to(torch.float32)

def _from01_to11(x:torch.Tensor):
    x_norm = (2.0*x - 1.0)
    x_norm = torch.clamp(x_norm, max=+1.0, min=-1.0)
    return x_norm.to(torch.float32)

def _from0255_to11(x:torch.Tensor):
    x = x/255.0
    x = (2.0 * x - 1.0)
    x = torch.clamp(x, min=-1.0, max=1.0)
    return x

def _from11_to01(x:torch.Tensor):
    x = (x + 1.0)/2.0
    x = torch.clamp(x, min=0.0, max=1.0)
    return x.to(torch.float32)


def _normalize_speckle(x:torch.Tensor, gamma:float) -> torch.Tensor:
    x_norm = _fromdynamic_to01(x)
    x_gamma = x_norm**gamma
    x_norm = _from01_to11(x_gamma)
    return x_norm

def _normalize_img(x:torch.Tensor) -> torch.Tensor: 
    return _from0255_to11(x)