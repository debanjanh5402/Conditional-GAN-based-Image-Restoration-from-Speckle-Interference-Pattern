from pathlib import Path

import torch
from tqdm import tqdm

def _save_checkpoint(filepath:str|Path, *,
                     history: dict,
                     generator = None, discriminator = None, 
                     g_optimizer = None, d_optimizer = None,
                     running_step:int|None = None):
    
    filepath = Path(filepath)
    checkpoint = {"history": history}

    if generator is not None:
          checkpoint["generator"] = generator.state_dict()

    if discriminator is not None:
          checkpoint["discriminator"] = discriminator.state_dict()

    if g_optimizer is not None:
          checkpoint["g_optimizer"] = g_optimizer.state_dict()

    if d_optimizer is not None:
          checkpoint["d_optimizer"] = d_optimizer.state_dict()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    if running_step is not None:
         tqdm.write(f"\nCheckpoint from step {running_step} saved at {filepath}")




def _load_checkpoint(filepath: str|Path, 
                     generator = None, 
                     discriminator=None, 
                     g_optimizer = None, 
                     d_optimizer = None,
                     map_location = "cpu"):

    checkpoint = torch.load(Path(filepath), map_location=map_location)

    if generator is not None:
          generator.load_state_dict(checkpoint["generator"])

    if g_optimizer is not None:
         g_optimizer.load_state_dict(checkpoint["g_optimizer"])

    if discriminator is not None:
        discriminator.load_state_dict(checkpoint["discriminator"])

    if d_optimizer is not None:
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])

    return checkpoint["history"]