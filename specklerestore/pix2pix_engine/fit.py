from pathlib import Path

from itertools import cycle

from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch import device

from tqdm import tqdm

from .train_step import train_discriminator, train_generator
from .val_step import validate
from .utils import _save_checkpoint
from ..losses import Pix2PixLoss



def fit_pix2pix(train_loader: DataLoader,
                *,
                generator: Module, discriminator: Module,
                optimizer_g: Optimizer, optimizer_d: Optimizer,
                loss_fn: Pix2PixLoss, 
                steps: int,
                device: str|device,
                gd_update_ratio: int = 1,
                val_loader: DataLoader|None = None,
                validate_every: int|None = None,
                latest_checkpoint_dir: str|Path|None = None,
                checkpoint_every: int|None = None,
                best_model_dir: str|Path|None = None,
                fname_identifier: str =""):

    # validate update ratio
    if gd_update_ratio < 1: 
        raise ValueError(f"generator_discriminator_update_ratio must be >= 1")

    # training history
    history = {"train": {"steps": [], 
                         "g_loss": [], "g_adv_loss": [], "g_recon_loss": [],
                         "d_loss": [], "d_real_loss": [], "d_fake_loss": []},
                "val": {"steps": [],
                        "ssim": [], "psnr": [], "mse": []}}

    # validation setup
    do_validate = False
    if val_loader is not None:
        if validate_every is None:
            validate_every = len(train_loader)
        do_validate = True


    # latest checkpoint setup
    do_save = False
    if latest_checkpoint_dir is not None:
        if checkpoint_every is None:
            checkpoint_every = len(train_loader)
        do_save = True

    
    # best checkpoint setup
    do_save_best = False
    best_ssim = -1.0
    if best_model_dir is not None:
        if val_loader is None:
            raise ValueError("best_model_dir requires val_loader to be provided.")
        do_save_best = True


    # Training setup
    running_step = 1

    generator.to(device).train()
    discriminator.to(device).train()

    pbar = tqdm(total=steps, desc="Training", unit="step")

    train_loader = cycle(train_loader)


    # Training Loop
    while running_step <= steps:
        # Get batch
        batch = next(iter(train_loader))
        x, y = batch['input'].to(device), batch['target'].to(device)

        # Discriminator update
        d_losses = train_discriminator(x, y, generator, discriminator, optimizer_d, loss_fn)

        # Number of generator updates
        for _ in range(gd_update_ratio):
            g_losses = train_generator(x, y, generator, discriminator, optimizer_g, loss_fn)

        # Record training history
        history['train']['steps'].append(running_step)
        history['train']['g_loss'].append(g_losses['g_loss'])
        history['train']['g_adv_loss'].append(g_losses['g_adv_loss'])
        history['train']['g_recon_loss'].append(g_losses['g_recon_loss'])
        history['train']['d_loss'].append(d_losses['d_loss'])
        history['train']['d_real_loss'].append(d_losses['d_real_loss'])
        history['train']['d_fake_loss'].append(d_losses['d_fake_loss'])

        # Progress bar
        pbar.set_postfix(g=f"{g_losses['g_loss']:.4f}", d=f"{d_losses['d_loss']:.4f}")

        # Validation
        if do_validate and running_step % validate_every == 0:
            val_metrics = validate(val_loader, generator, device)
            history['val']['steps'].append(running_step)
            history['val']['ssim'].append(val_metrics['ssim'])
            history['val']['psnr'].append(val_metrics['psnr'])
            history['val']['mse'].append(val_metrics['mse'])

            if do_save_best:
                current_ssim = val_metrics['ssim']
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_model_path = Path(best_model_dir) / f"best_checkpoint_{fname_identifier}.pt"
                    #best_model_path = Path(best_model_dir) / f"best_checkpoint.pt"
                    _save_checkpoint(best_model_path, history=history, 
                                    generator=generator, discriminator=discriminator, 
                                    g_optimizer=optimizer_g, d_optimizer=optimizer_d)
                    tqdm.write(f"\nNew best model at step {running_step} | SSIM: {best_ssim:.6f}")

        # Latest checkpoint
        if do_save and running_step % checkpoint_every == 0:
            latest_checkpoint_path = Path(latest_checkpoint_dir) / f"latest_checkpoint_{fname_identifier}.pt"
            #latest_checkpoint_path = Path(latest_checkpoint_dir) / f"latest_checkpoint.pt"
            _save_checkpoint(latest_checkpoint_path, history=history, 
                             generator=generator, discriminator=discriminator,
                             g_optimizer=optimizer_g, d_optimizer=optimizer_d)

            

        running_step += 1
        pbar.update(1)

    pbar.close()

    return history 

            


        






    