from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import tqdm

def _save_checkpoint_unet(
        filepath:str|Path,
        epoch: int, 
        history: dict,
        model: Module|None = None,
        optimizer: Optimizer|None = None, 
        scheduler = None,
        log_str:str|None = None):
    
    filepath = Path(filepath)
    checkpoint = {"epoch":epoch, "history": history}

    if model is not None:
        checkpoint['model'] = model.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    if log_str is not None:
        tqdm.write(str(log_str))



def _load_checkpoint_unet(
        checkpoint_path:str|Path, 
        device:torch.device,
        model:Module|None, 
        optimizer:Optimizer|None = None,
        scheduler = None):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(state_dict=checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(state_dict=checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint['epoch'], checkpoint['history']



def _save_checkpoint_pix2pix(
          filepath: str|Path,
          step: int,
          history: dict,
          generator: Module|None = None,
          discriminator: Module|None = None,
          g_optimizer: Optimizer|None = None,
          d_optimizer: Optimizer|None = None,
          g_scheduler = None,
          d_scheduler = None,
          log_str: str|None= None
          ): 

     filepath = Path(filepath)
     filepath.parent.mkdir(parents=True, exist_ok=True)

     checkpoint = {'step': step, 'history': history}

     if generator is not None: checkpoint["generator"] = generator.state_dict()
     if discriminator is not None: checkpoint["discriminator"] = discriminator.state_dict()
     if g_optimizer is not None: checkpoint["optimizer_g"] = g_optimizer.state_dict()
     if d_optimizer is not None: checkpoint["optimizer_d"] = d_optimizer.state_dict()
     if g_scheduler is not None: checkpoint["scheduler_g"] = g_scheduler.state_dict()
     if d_scheduler is not None: checkpoint["scheduler_d"] = d_scheduler.state_dict()


     torch.save(checkpoint, filepath)

     if log_str is not None:
          tqdm.write(str(log_str))


def _load_checkpoint_pix2pix(
          filepath: str | Path, 
          device: torch.device,
          generator=None, 
          discriminator=None, 
          g_optimizer=None, 
          d_optimizer=None,
          g_scheduler = None,
          d_scheduler = None,):
    
    checkpoint = torch.load(Path(filepath), map_location=device)

    if generator is not None: generator.load_state_dict(checkpoint["generator"])
    if discriminator is not None: discriminator.load_state_dict(checkpoint["discriminator"])
    if g_optimizer is not None: g_optimizer.load_state_dict(checkpoint["optimizer_g"])
    if d_optimizer is not None: d_optimizer.load_state_dict(checkpoint["optimizer_d"])
    if g_scheduler is not None: g_scheduler.load_state_dict(checkpoint["scheduler_g"])
    if d_scheduler is not None: d_scheduler.load_state_dict(checkpoint["scheduler_d"])


    return checkpoint["step"], checkpoint["history"]