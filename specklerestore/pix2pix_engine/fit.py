from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)

from tqdm import tqdm


from .train_step import train_step
from .val_step import val_step
from .predict import predict
from ..losses import Pix2PixLoss
from ..utils import (_log_epoch_summary_pix2pix, _save_checkpoint_pix2pix, _load_checkpoint_pix2pix)


def fit_pix2pix(
        *,
        train_loader: DataLoader,
        generator: Module,
        discriminator: Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        loss_fn: Pix2PixLoss,
        epochs: int,
        device: torch.device|str,
        checkpoint_dir: str|Path|None = None,
        fname_identifier: str|None = None,
        val_loader: DataLoader|None = None,
        test_loader: DataLoader|None = None,
        resume:bool|None = None
        ):

    do_validation = False
    do_save_latest_checkpoint = False

    # Checkpoint paths
    if checkpoint_dir is not None:
        do_save_latest_checkpoint = True
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pt"
        best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

        if fname_identifier is not None:
            latest_checkpoint_path = checkpoint_dir / f"latest_checkpoint_{str(fname_identifier)}.pt"
            best_checkpoint_path = checkpoint_dir / f"best_checkpoint_{str(fname_identifier)}.pt"

    if val_loader is not None:
        do_validation = True


    # Initialize training state
    history = {"train": {"g_loss": [], "g_adv_loss": [], "g_recon_loss": [],
                         "d_loss": [], "d_real_loss": [], "d_fake_loss": [],
                         "ssim": [], "psnr": []},
               "val": {"g_loss": [], "g_adv_loss": [], "g_recon_loss": [],
                       "d_loss": [], "d_real_loss": [], "d_fake_loss": [],
                       "ssim": [], "psnr": []},
               "best": None}

    best_val_ssim = -1.0
    start_epoch = 1

    ssim_monitor = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_monitor = PeakSignalNoiseRatio(data_range=1.0).to(device)


    # Resume from last checkpoint
    if resume:
        tqdm.write(f"Resuming training from checkpoint: {latest_checkpoint_path}")
        last_epoch, history = _load_checkpoint_pix2pix(latest_checkpoint_path, device, 
                                                       generator, discriminator, 
                                                       optimizer_g, optimizer_d)
        start_epoch = last_epoch + 1
        if history['best'] is not None:
            best_val_ssim = history['best']['val_ssim']

        tqdm.write(f"Resuming from epoch {start_epoch}. Previous best SSIM: {best_val_ssim:0.6f}")

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    for epoch in tqdm(range(start_epoch, epochs+1), desc="Epochs", unit="epoch"):
        is_best = False

        # Training
        train_history = train_step(train_loader, 
                                   generator, discriminator,
                                   optimizer_g, optimizer_d,
                                   loss_fn, device, 
                                   ssim_monitor, psnr_monitor)
        history['train']['g_loss'].append(train_history['g_loss'])
        history['train']['g_adv_loss'].append(train_history['g_adv_loss'])
        history['train']['g_recon_loss'].append(train_history['g_recon_loss'])
        history['train']['d_loss'].append(train_history['d_loss'])
        history['train']['d_real_loss'].append(train_history['d_real_loss'])
        history['train']['d_fake_loss'].append(train_history['d_fake_loss'])
        history['train']['ssim'].append(train_history['ssim'])
        history['train']['psnr'].append(train_history['psnr'])


        if do_validation:
            val_history = val_step(val_loader, 
                                   generator, discriminator,
                                   loss_fn, device,
                                   ssim_monitor, psnr_monitor)

            history['val']['g_loss'].append(val_history['g_loss'])
            history['val']['g_adv_loss'].append(val_history['g_adv_loss'])
            history['val']['g_recon_loss'].append(val_history['g_recon_loss'])
            history['val']['d_loss'].append(val_history['d_loss'])
            history['val']['d_real_loss'].append(val_history['d_real_loss'])
            history['val']['d_fake_loss'].append(val_history['d_fake_loss'])
            history['val']['ssim'].append(val_history['ssim'])
            history['val']['psnr'].append(val_history['psnr'])

            _log_epoch_summary_pix2pix(epoch, train_history, val_history)

            is_best = True if val_history['ssim'] > best_val_ssim else False
            if is_best:
                best_val_ssim = val_history['ssim']
                history['best'] = {"epoch": epoch,
                                   "train_g_loss": train_history['g_loss'], 
                                   "train_g_adv_loss": train_history['g_adv_loss'], 
                                   "train_g_recon_loss": train_history['g_recon_loss'],
                                   "train_d_loss": train_history['d_loss'],
                                   "train_d_real_loss": train_history['d_real_loss'],
                                   "train_d_fake_loss": train_history['d_fake_loss'],
                                   "train_ssim": train_history['ssim'],
                                   "train_psnr": train_history['psnr'],
                                   "val_g_loss": val_history['g_loss'], 
                                   "val_g_adv_loss": val_history['g_adv_loss'], 
                                   "val_g_recon_loss": val_history['g_recon_loss'],
                                   "val_d_loss": val_history['d_loss'],
                                   "val_d_real_loss": val_history['d_real_loss'],
                                   "val_d_fake_loss": val_history['d_fake_loss'],
                                   "val_ssim": val_history['ssim'],
                                   "val_psnr": val_history['psnr']}
                _save_checkpoint_pix2pix(best_checkpoint_path, epoch, history, 
                                 generator, discriminator,
                                 optimizer_g, optimizer_d, 
                                 log_str=f"Best checkpoint from epoch {epoch} saved at {best_checkpoint_path}")

        else:
            _log_epoch_summary_pix2pix(epoch, train_history)
            is_best = True

        # Latest checkpoint
        if do_save_latest_checkpoint:
            _save_checkpoint_pix2pix(latest_checkpoint_path, epoch, history,
                                     generator, discriminator, optimizer_g, optimizer_d)

        # Prediction on Test Set
        if test_loader is not None and is_best:
            predict(test_loader, generator, device, ssim_monitor, psnr_monitor, fname_identifier)

        tqdm.write("\n")

    return history