from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import tqdm

def from11_to01(x:torch.Tensor):
    x = (x+1.0)/2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x.to(torch.float32)

def _log_epoch_summary(epoch:int, train_history:dict, val_history:dict|None = None):
    tqdm.write(f"{'-'*35} Epoch {epoch} Summary {'-'*35}")
    tqdm.write(f"train: loss={train_history['loss']:0.4f}, ssim={train_history['ssim']:0.4f}, psnr={train_history['psnr']:0.4f}")
    if val_history is not None:
        tqdm.write(f"  val: loss={val_history['loss']:0.4f}, ssim={val_history['ssim']:0.4f}, psnr={val_history['psnr']:0.4f}")
        

def _save_checkpoint(filepath:str|Path,
                     epoch: int, 
                     history: dict,
                     model: Module|None = None,
                     optimizer: Optimizer|None = None, 
                     log_str:str|None = None):
    filepath = Path(filepath)
    checkpoint = {"epoch":epoch, "history": history}

    if model is not None:
        checkpoint['model'] = model.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    if log_str is not None:
        tqdm.write(str(log_str))


def _load_checkpoint(checkpoint_path:str|Path, 
                     device:torch.device,
                     model:Module|None, 
                     optimizer:Optimizer|None = None):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(state_dict=checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(state_dict=checkpoint['optimizer'])

    return checkpoint['epoch'], checkpoint['history']
    