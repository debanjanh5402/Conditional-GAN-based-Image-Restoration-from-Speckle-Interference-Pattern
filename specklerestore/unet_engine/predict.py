from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from torchvision import io

from tqdm import tqdm

from ..config import TEST_DIR
from .utils import from11_to01

def predict(test_loader: DataLoader,
            model: Module,
            device: torch.device,
            save_data:bool,
            ssim_metric,
            psnr_metric,
            fname_identifier: str|None = None):

    was_training = model.training
    model.eval()

    ssim_metric.reset()
    psnr_metric.reset()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Prediction:", unit="sample", leave=False):
            name, x, y = batch['name'][0], batch['input'].to(device), batch['target'].to(device)

            y_pred = model(x)

            y_norm = from11_to01(y)
            y_pred_norm = from11_to01(y_pred)

            # Metrics
            ssim = (ssim_metric(y_pred_norm, y_norm)).item()
            psnr = (psnr_metric(y_pred_norm, y_norm)).item()

            if save_data:
                data = {'name': name,
                        'speckle1': x[0, 0].unsqueeze(0), 'speckle2':x[0, 1].unsqueeze(0), 'speckle3': x[0, 2].unsqueeze(0),
                        'target': y.squeeze(0), 'predicted': y_pred.squeeze(0), 
                        'ssim': ssim, 'psnr': psnr}
                torch.save(data, Path(TEST_DIR) / name / f"{name}_{fname_identifier}_data.pt")

            ssim_metric.reset()
            psnr_metric.reset()

    if was_training:
        model.train()