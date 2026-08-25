from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from torchmetrics.image import (StructuralSimilarityIndexMeasure,
                                PeakSignalNoiseRatio)

from tqdm import tqdm

from .train_step import train_step
from .val_step import val_step
from .predict import predict
from ..losses import Pix2PixLoss
from ..utils import (_save_checkpoint_pix2pix, _load_checkpoint_pix2pix,
                     _log_step_summary_pix2pix)
from ..config import TEST_DIR



def fit_pix2pix(
        train_loader: DataLoader,
        generator: Module,
        discriminator: Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        loss_fn: Pix2PixLoss,
        total_steps: int,
        val_interval: int,
        device: torch.device|str,
        g_updates_per_step: int = 1,
        d_updates_per_step: int = 1,
        checkpoint_dir: str|Path|None = None,
        fname_identifier: str|None = None,
        val_loader: DataLoader|None = None,
        test_loader: DataLoader|None = None,
        resume:bool = False,
        discriminator_reset_interval: int | None = None,
        ) -> dict:

    do_validation = False
    do_save_ckpt = False

    if checkpoint_dir is not None:
        do_save_ckpt = True
        checkpoint_dir = Path(checkpoint_dir)
        latest_ckpt_path = checkpoint_dir / "latest_checkpoint.pt"
        best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"

        if fname_identifier is not None:
            latest_ckpt_path = checkpoint_dir / f"latest_checkpoint_{fname_identifier}.pt"
            best_ckpt_path = checkpoint_dir / f"best_checkpoint_{fname_identifier}.pt"

    if val_loader is not None: do_validation = True

    history = {
        "train": {"steps": [],
                  "g_loss": [], "g_adv_loss": [], "g_recon_loss": [],
                  "d_loss": [], "d_real_loss": [], "d_fake_loss": []}, 
        "val": {"steps": [],
                "g_loss": [], "g_adv_loss": [], "g_recon_loss": [],
                "d_loss": [], "d_real_loss": [], "d_fake_loss": [],
                "ssim": [], "psnr": []}, 
        "best": None}
    
    best_val_ssim = -1.0
    start_step = 1

    ssim_monitor = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_monitor = PeakSignalNoiseRatio(data_range=1.0).to(device)

    if resume and latest_ckpt_path.exists():
        last_step, history = _load_checkpoint_pix2pix(latest_ckpt_path, device, 
                                                            generator, discriminator, 
                                                            optimizer_g, optimizer_d)
        if history.get("best"): best_val_ssim = history['best'].get("val_ssim", -1.0)
        start_step = last_step + 1

    generator.train()
    discriminator.train()
    train_iter = iter(train_loader)

    loop = tqdm(range(start_step, total_steps+1), desc="Steps", unit="step")

    for step in loop:

        try: 
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        update_g = (step % g_updates_per_step == 0)
        update_d = (step % d_updates_per_step == 0)

        if (
            discriminator_reset_interval is not None
            and discriminator_reset_interval > 0
            and step > start_step
            and step % discriminator_reset_interval == 0
        ):
            reset_discriminator(discriminator, optimizer_d)

        train_history = train_step(batch, generator, discriminator, optimizer_g, optimizer_d, 
                                   loss_fn, device, update_d, update_g)


        history['train']['steps'].append(step)
        history['train']['g_loss'].append(train_history['g_loss'])
        history['train']['g_adv_loss'].append(train_history['g_adv_loss'])
        history['train']['g_recon_loss'].append(train_history['g_recon_loss'])

        history['train']['d_loss'].append(train_history['d_loss'])
        history['train']['d_real_loss'].append(train_history['d_real_loss'])
        history['train']['d_fake_loss'].append(train_history['d_fake_loss'])

        if step % val_interval == 0:
            is_best = False

            if do_validation:
                val_history = val_step(val_loader, generator, discriminator, loss_fn, device, ssim_monitor, psnr_monitor)

                history['val']['steps'].append(step)
                history['val']['g_loss'].append(val_history['g_loss'])
                history['val']['g_adv_loss'].append(val_history['g_adv_loss'])
                history['val']['g_recon_loss'].append(val_history['g_recon_loss'])

                history['val']['d_loss'].append(val_history['d_loss'])
                history['val']['d_real_loss'].append(val_history['d_real_loss'])
                history['val']['d_fake_loss'].append(val_history['d_fake_loss'])

                history['val']['ssim'].append(val_history['ssim'])
                history['val']['psnr'].append(val_history['psnr'])

                _log_step_summary_pix2pix(step, train_history, val_history)

                if val_history['ssim'] > best_val_ssim:
                    is_best = True
                    best_val_ssim = val_history['ssim']
                    history['best'] = {'step': step, 
                                       'train_g_loss': train_history['g_loss'], 'train_g_adv_loss': train_history['g_adv_loss'], 'train_g_recon_loss': train_history['g_recon_loss'],
                                       'train_d_loss': train_history['d_loss'], 'train_d_real_loss': train_history['d_real_loss'], 'train_d_fake_loss': train_history['d_fake_loss'],
                                       'val_g_loss': val_history['g_loss'], 'val_g_adv_loss': val_history['g_adv_loss'], 'val_g_recon_loss': val_history['g_recon_loss'],
                                       'val_d_loss': val_history['d_loss'], 'val_d_real_loss': val_history['d_real_loss'], 'val_d_fake_loss': val_history['d_fake_loss'],
                                       'val_ssim': val_history['ssim'], 'val_psnr': val_history['psnr']}

                    if do_save_ckpt: _save_checkpoint_pix2pix(best_ckpt_path, step, history, 
                                                                   generator, discriminator, 
                                                                   optimizer_g, optimizer_d,
                                                                   log_str=f"Best checkpoint at step {step} saved at {best_ckpt_path}")

            else:
                _log_step_summary_pix2pix(step, train_history)

            if do_save_ckpt: _save_checkpoint_pix2pix(latest_ckpt_path, step, history, 
                                                      generator, discriminator, 
                                                      optimizer_g, optimizer_d)

            if test_loader is not None and is_best:
                predict(test_loader, generator, device, ssim_monitor, psnr_monitor, TEST_DIR, fname_identifier)

    return history    



def reset_discriminator(discriminator: Module, optimizer_d: Optimizer):
    for module in discriminator.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    optimizer_d.state.clear()