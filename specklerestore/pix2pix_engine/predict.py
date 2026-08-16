from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from tqdm import tqdm

from ..utils.pix2pix_train_utils import from11_to01
from ..config import TEST_DIR


def predict(
        dataloader: DataLoader,
        generator: Module,
        device:torch.device | str,
        ssim_metric,
        psnr_metric,
        fname_identifier:str): 

    was_training = generator.training
    generator.eval()

    ssim_metric.reset()
    psnr_metric.reset()

    for sample in tqdm(dataloader, desc="Prediction", leave=False, unit="samples"):
        name, x, y = sample['name'][0], sample['input'].to(device), sample['target'].to(device)

        with torch.no_grad():
            y_pred = generator(x)

            y_norm = from11_to01(y)
            y_pred_norm = from11_to01(y_pred)
            ssim = ssim_metric(y_pred_norm, y_norm).item()
            psnr = psnr_metric(y_pred_norm, y_norm).item()

        data = {"name": name,
                "speckle1": x[0, 0].unsqueeze(dim=0).cpu(),
                "speckle2": x[0, 1].unsqueeze(dim=0).cpu(),
                "speckle3": x[0, 2].unsqueeze(dim=0).cpu(),
                "target": y.squeeze(dim=0).cpu(),
                "predicted": y_pred.squeeze(dim=0).cpu(),
                "ssim": ssim, "psnr": psnr}
        
        save_path = Path(TEST_DIR) / name / f"{name}_{fname_identifier}_data.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_path)

    if was_training:
        generator.train()