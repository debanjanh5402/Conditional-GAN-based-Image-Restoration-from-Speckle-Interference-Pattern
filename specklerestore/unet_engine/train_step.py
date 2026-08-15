import torch 
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from tqdm import tqdm

from .utils import from11_to01

def train_step(
        train_loader: DataLoader,
        model: Module, 
        optimizer: Optimizer, 
        loss_fn: Module,
        device: torch.device|str,
        ssim_metric,
        psnr_metric) -> dict[str, float]:

    running_loss = 0.0
    running_sample = 0

    pbar = tqdm(total=len(train_loader), desc="  Training", unit="batch", leave=False)

    ssim_metric.reset()
    psnr_metric.reset()

    for batch in train_loader:
        x, y = batch['input'].to(device), batch['target'].to(device)
        running_sample += x.shape[0]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        y_norm = from11_to01(y)
        y_pred_norm = from11_to01(y_pred)
        ssim_metric.update(y_pred_norm, y_norm)
        psnr_metric.update(y_pred_norm, y_norm)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.shape[0]

        pbar.set_postfix(loss = f"{(running_loss/running_sample):.4f}")
        pbar.update(1)

    pbar.close()

    return {'loss': running_loss/running_sample,
            'ssim': ssim_metric.compute().item(),
            'psnr': psnr_metric.compute().item()}


