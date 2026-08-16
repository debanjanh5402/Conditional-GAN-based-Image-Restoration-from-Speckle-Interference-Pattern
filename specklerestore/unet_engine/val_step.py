import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from tqdm import tqdm

from ..utils import _from11_to01

def val_step(
        val_loader:DataLoader,
        model: Module,
        loss_fn:Module,
        device: torch.device|str,
        ssim_metric,
        psnr_metric) -> dict[str, float]:

    was_training = model.training
    model.eval()

    running_loss = 0.0
    running_sample = 0

    ssim_metric.reset()
    psnr_metric.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", unit="batches", leave=False):
            x, y = batch['input'].to(device), batch['target'].to(device)
            running_sample += x.shape[0]

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            y_norm = _from11_to01(y)
            y_pred_norm = _from11_to01(y_pred)
            ssim_metric.update(y_pred_norm, y_norm)
            psnr_metric.update(y_pred_norm, y_norm)

            running_loss += loss.item() * x.shape[0]

    if was_training:
        model.train()

    return {'loss': running_loss/running_sample,
            'ssim': ssim_metric.compute().item(),
            'psnr': psnr_metric.compute().item()}