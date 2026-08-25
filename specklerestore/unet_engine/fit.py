from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from .train_step import train_step
from .val_step import val_step
from .predict import predict
from ..utils import (_load_checkpoint_unet, _save_checkpoint_unet,
                     _log_epoch_summary_unet)



def fit_unet(
        *,
        train_loader:DataLoader, 
        model: Module,
        optimizer: Optimizer,
        loss_fn: Module,
        epochs: int,
        device:torch.device|str,
        checkpoint_dir:str|Path|None=None,
        fname_identifier: str|None = None,
        val_loader:DataLoader|None = None,
        test_loader: DataLoader|None = None,
        resume:bool|None = None):

    # Checkpoint paths
    if checkpoint_dir is not None:
        do_save_latest_checkpoint = True
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
        best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pt'

        if fname_identifier is not None:
            latest_checkpoint_path = checkpoint_dir / f"latest_checkpoint_{str(fname_identifier)}.pt"
            best_checkpoint_path = checkpoint_dir / f"best_checkpoint_{str(fname_identifier)}.pt"

    if val_loader is not None:
        do_validation = True

    # Initialize training state
    history = {"train" : {'loss': [], 'ssim': [], 'psnr': []},
               "val": {'loss': [], 'ssim': [], 'psnr': []},
               "best": None}
    
    best_val_ssim = -1.0
    start_epoch = 1

    ssim_monitor = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_monitor = PeakSignalNoiseRatio(data_range=1.0).to(device)


    # Resume from last checkpoint
    if resume:
        tqdm.write(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        last_epoch, history = _load_checkpoint_unet(latest_checkpoint_path, device, 
                                                    model, optimizer)
        start_epoch = last_epoch + 1
        if history['best'] is not None:
            best_val_ssim = history['best']['val_ssim']

        tqdm.write(f"Resuming from epoch {start_epoch}. Previous best SSIM: {best_val_ssim:.6f}\n")

    # Training setup
    model.to(device)
    model.train()

    # Main training loop
    for epoch in tqdm(range(start_epoch, epochs+1), desc="    Epochs", unit="epoch"):  
        is_best = False

        # Training
        train_history = train_step(train_loader, 
                                   model, optimizer, 
                                   loss_fn, device, 
                                   ssim_monitor, psnr_monitor)
        history['train']['loss'].append(train_history['loss'])
        history['train']['ssim'].append(train_history['ssim'])
        history['train']['psnr'].append(train_history['psnr'])

        # Validation
        if do_validation:
            val_history = val_step(val_loader, 
                                   model, loss_fn, device, 
                                   ssim_monitor, psnr_monitor)
            
            history['val']['loss'].append(val_history['loss'])
            history['val']['ssim'].append(val_history['ssim'])
            history['val']['psnr'].append(val_history['psnr'])

            _log_epoch_summary_unet(epoch, train_history, val_history)

            # Best checkpoint
            is_best = True if val_history['ssim'] > best_val_ssim else False
            if is_best:
                best_val_ssim = val_history['ssim']
                history["best"] = {'epoch': epoch, 
                                   'train_loss': train_history['loss'], 
                                   'train_ssim': train_history['ssim'], 
                                   'train_psnr': train_history['psnr'],
                                   'val_loss': val_history['loss'], 
                                   'val_ssim': val_history['ssim'], 
                                   'val_psnr': val_history['psnr']}
                _save_checkpoint_unet(best_checkpoint_path, epoch, history, model, optimizer,
                                 log_str=f"Best checkpoint from epoch {epoch} saved at {best_checkpoint_path}")

        else:
            _log_epoch_summary_unet(epoch, train_history)

        # Latest checkpoint
        if do_save_latest_checkpoint:
            _save_checkpoint_unet(latest_checkpoint_path, epoch, history, model, optimizer)

        # Test Prediction
        if test_loader is not None and is_best:
            predict(test_loader, model, device, ssim_monitor, psnr_monitor, fname_identifier)

        tqdm.write("\n")

    return history