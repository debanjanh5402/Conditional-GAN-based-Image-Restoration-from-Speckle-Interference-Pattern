import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm


@torch.no_grad()
def validate(dataloader: DataLoader,
             generator:nn.Module,
             device:torch.device|str|None = None) -> dict[str, float]:

    was_training = generator.training
    generator.eval()

    mse_metric = MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    running_mse = 0
    for batch in tqdm(dataloader, leave=False, desc="Validation"):
        x, y = batch["input"].to(device), batch["target"].to(device)
        y_pred = generator(x)

        mse = mse_metric(y_pred, y)
        running_mse += mse.item()

        y_pred_norm = (y_pred+1.0)/2.0
        y_norm = (y+1.0)/2.0

        ssim_metric.update(y_pred_norm, y_norm)
        psnr_metric.update(y_pred_norm, y_norm)

    if was_training:
        generator.train()

    running_mse /= len(dataloader)

    return {
        "ssim": ssim_metric.compute().item(),
        "psnr": psnr_metric.compute().item(),
        "mse": running_mse
        }