from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from tqdm import tqdm

from ..utils import _from11_to01



def predict(
        dataloader: DataLoader,
        generator: Module,
        device: torch.device,
        ssim_metric,
        psnr_metric,
        output_dir:str|Path,
        fname_identifier:str
        ):

    was_traning = generator.training
    generator.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(dataloader, desc="Prediction", leave=False, unit="samples"):
        name = sample['name'][0]
        x, y = sample['input'].to(device), sample['target'].to(device)

        with torch.no_grad():
            y_pred = generator(x)

            ssim_metric.reset()
            psnr_metric.reset()
            ssim_val = ssim_metric(_from11_to01(y_pred), _from11_to01(y)).item()
            psnr_val = psnr_metric(_from11_to01(y_pred), _from11_to01(y)).item()

        data = {
            "name": name,
            "speckle1": x[0,0].unsqueeze(0).cpu(),
            "speckle2": x[0,1].unsqueeze(0).cpu(),
            "speckle3": x[0,2].unsqueeze(0).cpu(),
            "target": y.squeeze(0).cpu(),
            "predicted": y_pred.squeeze(0).cpu(),
            "ssim": ssim_val,
            "psnr": psnr_val
        }

        sample_dir = output_dir / name
        sample_dir.mkdir(parents=True, exist_ok = True)
        torch.save(data, sample_dir / f"{name}_{fname_identifier}_data.pt")

    if was_traning: generator.train()