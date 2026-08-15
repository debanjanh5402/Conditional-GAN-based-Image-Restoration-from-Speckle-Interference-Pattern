import torch
from torch import nn, optim

from ..losses import Pix2PixLoss


def train_discriminator(
        x:torch.Tensor, y:torch.Tensor,
        generator:nn.Module, discriminator:nn.Module,
        optimizer_d: optim.Optimizer,
        loss_fn: Pix2PixLoss
        ) -> dict[str, float]:

    with torch.no_grad():
        y_pred = generator(x)


    optimizer_d.zero_grad()

    d_out_real = discriminator(x, y)
    d_out_fake = discriminator(x, y_pred)

    d_loss, d_real_loss, d_fake_loss = loss_fn.discriminator_loss(d_out_real=d_out_real, d_out_fake=d_out_fake)

    d_loss.backward()
    optimizer_d.step()

    return {"d_loss": d_loss.item(),
            "d_real_loss": d_real_loss.item(),
            "d_fake_loss": d_fake_loss.item()}


def train_generator(
        x:torch.Tensor, y:torch.Tensor,
        generator:nn.Module, discriminator:nn.Module,
        optimizer_g: optim.Optimizer,
        loss_fn: Pix2PixLoss,
        ) -> dict[str, float]:

    for parameter in discriminator.parameters():
        parameter.requires_grad = False

    optimizer_g.zero_grad()

    y_pred = generator(x)
    d_out_fake = discriminator(x, y_pred)

    g_loss, g_adv_loss, g_recon_loss = loss_fn.generator_loss(d_out_fake=d_out_fake, y_pred=y_pred, y=y)

    g_loss.backward()
    optimizer_g.step()

    for parameter in discriminator.parameters():
        parameter.requires_grad = True

    return {"g_loss": g_loss.item(),
            "g_adv_loss": g_adv_loss.item(),
            "g_recon_loss": g_recon_loss.item()}