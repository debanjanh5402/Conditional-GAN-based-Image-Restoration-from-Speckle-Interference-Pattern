import torch
import torch.nn as nn


class Pix2PixLoss:
    def __init__(self, recon_loss:nn.Module = nn.L1Loss(), lambda_recon: float = 100.0):
        self.lambda_recon = lambda_recon

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.recon_loss = recon_loss

    def generator_loss(self, d_out_fake: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        adv_loss = self.bce_loss(d_out_fake, torch.ones_like(d_out_fake))
        recon_loss = self.recon_loss(y_pred, y)
        total_loss = adv_loss + self.lambda_recon * recon_loss
        return total_loss, adv_loss, recon_loss

    def discriminator_loss(self, d_out_real: torch.Tensor, d_out_fake: torch.Tensor):
        real_loss = self.bce_loss(d_out_real, torch.ones_like(d_out_real))
        fake_loss = self.bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        total_loss = 0.5 * (real_loss + fake_loss)
        return total_loss, real_loss, fake_loss