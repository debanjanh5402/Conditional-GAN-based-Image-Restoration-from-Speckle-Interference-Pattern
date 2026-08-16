import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses import Pix2PixLoss
from ..utils.pix2pix_train_utils import from11_to01

def train_step(
        dataloader: DataLoader,
        generator: Module,
        discriminator: Module,
        g_opt: Optimizer,
        d_opt: Optimizer,
        loss_fn: Pix2PixLoss,
        device:torch.device|str,
        ssim_metric,
        psnr_metric) -> dict[str, float]:

    num_samples = 0
    total_g_loss, total_g_adv_loss, total_g_recon_loss = 0.0, 0.0, 0.0
    total_d_loss, total_d_real_loss, total_d_fake_loss = 0.0, 0.0, 0.0
    ssim_metric.reset()
    psnr_metric.reset()

    for batch in tqdm(dataloader, desc="  Training", leave=False, unit="batches"):
        x, y = batch['input'].to(device), batch['target'].to(device)
        batch_size = x.shape[0]

        # Train Discriminator
        d_opt.zero_grad()
        with torch.no_grad():
            y_pred = generator(x)

        d_out_real = discriminator(x, y)
        d_out_fake = discriminator(x, y_pred)
        d_loss, d_real_loss, d_fake_loss = loss_fn.discriminator_loss(d_out_real, d_out_fake)
        d_loss.backward()
        d_opt.step()


        # Train Generator
        g_opt.zero_grad()
        for p in discriminator.parameters():
            p.requires_grad = False
        y_pred = generator(x)
        d_out_fake = discriminator(x, y_pred)
        g_loss, g_adv_loss, g_recon_loss = loss_fn.generator_loss(d_out_fake, y_pred, y)
        g_loss.backward()
        g_opt.step()
        for p in discriminator.parameters():
            p.requires_grad = True

        y_norm = from11_to01(y)
        y_pred_norm = from11_to01(y_pred)
        ssim_metric.update(y_pred_norm, y_norm)
        psnr_metric.update(y_pred_norm, y_norm)

        num_samples += batch_size
        total_g_loss += g_loss.item() * batch_size
        total_g_adv_loss += g_adv_loss.item() * batch_size
        total_g_recon_loss += g_recon_loss.item() * batch_size

        total_d_loss += d_loss.item() * batch_size
        total_d_real_loss += d_real_loss.item() * batch_size
        total_d_fake_loss += d_fake_loss.item() * batch_size

    return {"g_loss": total_g_loss/num_samples,
            "g_adv_loss": total_g_adv_loss/num_samples,
            "g_recon_loss": total_g_recon_loss/num_samples,
            "d_loss": total_d_loss/num_samples,
            "d_real_loss": total_d_real_loss/num_samples,
            "d_fake_loss": total_d_fake_loss/num_samples,
            "ssim": ssim_metric.compute().item(),
            "psnr": psnr_metric.compute().item()}