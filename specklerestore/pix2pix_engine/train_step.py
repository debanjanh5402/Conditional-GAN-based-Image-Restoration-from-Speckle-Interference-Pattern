import torch
from torch.nn import Module
from torch.optim import Optimizer

from ..losses import Pix2PixLoss

def train_step(
        batch: dict,
        generator: Module,
        discriminator: Module,
        g_opt: Optimizer,
        d_opt: Optimizer,
        loss_fn: Pix2PixLoss,
        device: torch.device,
        update_d: bool = True,
        update_g: bool = True,
        ) -> dict:

    x, y = batch['input'].to(device), batch['target'].to(device)

    history = {
        "g_loss": 0.0,
        "g_adv_loss": 0.0,
        "g_recon_loss": 0.0,
        "d_loss": 0.0,
        "d_real_loss": 0.0,
        "d_fake_loss": 0.0,
        }

    if update_d:
        d_opt.zero_grad()
        with torch.no_grad():
            y_pred_d = generator(x)
        d_out_real = discriminator(x, y)
        d_out_fake = discriminator(x, y_pred_d)
        d_loss, d_real_loss, d_fake_loss = loss_fn.discriminator_loss(d_out_real, d_out_fake)
        d_loss.backward()
        d_opt.step()

        history.update({
            "d_loss": d_loss.item(),
            "d_real_loss": d_real_loss.item(),
            "d_fake_loss": d_fake_loss.item()
        })

    if update_g:
        g_opt.zero_grad()

        for p in discriminator.parameters():
            p.requires_grad = False

        y_pred_g = generator(x)
        d_out_fake_g = discriminator(x, y_pred_g)
        g_loss, g_adv_loss, g_recon_loss = loss_fn.generator_loss(d_out_fake_g, y_pred_g, y)
        g_loss.backward()
        g_opt.step()

        for p in discriminator.parameters():
            p.requires_grad = True

        history.update({
            "g_loss": g_loss.item(),
            "g_adv_loss": g_adv_loss.item(),
            "g_recon_loss": g_recon_loss.item()
        })

    return history 