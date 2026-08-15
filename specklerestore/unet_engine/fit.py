from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from .train_step import train_step
from .val_step import val_step
from .utils import _log_epoch_summary, _save_checkpoint, _load_checkpoint
from .predict import predict




def fit_unet(
        train_loader:DataLoader, 
        model: Module, *,
        optimizer: Optimizer,
        loss_fn: Module,
        epochs: int,
        device:torch.device|str,
        checkpoint_dir:str|Path,
        val_loader:DataLoader|None = None,
        fname_identifier: str|None = None, 
        test_loader: DataLoader|None = None,
        resume:bool|None = None):

    # Checkpoint Paths
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
    best_checkpoint_path = checkpoint_dir / 'best_checkpoint.pt'
    if fname_identifier is not None:
        latest_checkpoint_path = checkpoint_dir / f"latest_checkpoint_{fname_identifier}.pt"
        best_checkpoint_path = checkpoint_dir / f"best_checkpoint_{fname_identifier}.pt"

    # Initialize training state
    history = {"train" : {'loss': [], 'ssim': [], 'psnr': []},
               "val": {'loss': [], 'ssim': [], 'psnr': []},
               "best": None}
    
    is_best = False
    best_ssim = -1.0
    start_epoch = 1

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)


    # Resume from last checkpoint
    if resume:
        tqdm.write(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        last_epoch, history = _load_checkpoint(latest_checkpoint_path, device, model, optimizer)
        start_epoch = last_epoch + 1
        if history['best'] is not None:
            best_ssim = history['best']['val_ssim']
        tqdm.write(f"Resuming from epoch {start_epoch}. Previous best SSIM: {best_ssim:.6f}\n")

    # Training setup
    model.to(device)
    model.train()
    pbar = tqdm(total=epochs, initial=start_epoch-1, desc="Epoch", unit='epoch')

    # Main training loop
    for epoch in range(start_epoch, epochs+1):  

        is_best = False

        # Training
        train_history = train_step(train_loader, model, optimizer, loss_fn, device, ssim_metric, psnr_metric)
        history['train']['loss'].append(train_history['loss'])
        history['train']['ssim'].append(train_history['ssim'])
        history['train']['psnr'].append(train_history['psnr'])

        # Validation
        if val_loader is not None:
            val_history = val_step(val_loader, model, loss_fn, device, ssim_metric, psnr_metric)
            history['val']['loss'].append(val_history['loss'])
            history['val']['ssim'].append(val_history['ssim'])
            history['val']['psnr'].append(val_history['psnr'])

            pbar.set_postfix(train_loss=f"{train_history['loss']:.4f}",
                             val_loss=f"{val_history['loss']:.4f}")

            _log_epoch_summary(epoch, train_history, val_history)

            # Best checkpoint
            if val_history['ssim'] > best_ssim:
                is_best = True
                best_ssim = val_history['ssim']
                history["best"] = {'epoch': epoch, 
                                   'train_loss': train_history['loss'], 'train_ssim': train_history['ssim'], 'train_psnr': train_history['psnr'],
                                   'val_loss': val_history['loss'], 'val_ssim': val_history['ssim'], 'val_psnr': val_history['psnr']}
                _save_checkpoint(best_checkpoint_path, epoch, history, model, optimizer, 
                                 log_str=f"Best checkpoint from epoch {epoch} saved at {best_checkpoint_path}")

        else:
            pbar.set_postfix(train_loss=f"{train_history['loss']:.4f}")
            _log_epoch_summary(epoch, train_history)

        # Latest checkpoint
        _save_checkpoint(latest_checkpoint_path, epoch, history, model, optimizer)

        # Test Prediction
        if test_loader is not None:
            predict(test_loader, model, device, is_best, ssim_metric, psnr_metric, fname_identifier=fname_identifier)

        tqdm.write("\n")
        pbar.update(1)
    pbar.close()

    return history