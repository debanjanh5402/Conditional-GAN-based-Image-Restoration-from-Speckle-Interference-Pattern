import torch
from torch import nn
from ..utils import _from11_to01
from torchmetrics.image import StructuralSimilarityIndexMeasure

class CustomLoss(nn.Module):
    def __init__(self, lambda_ssim: float, device:torch.device):
        super().__init__()

        self.lambda_ssim = lambda_ssim
        self.l1_loss = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor):
        loss_l1 = self.l1_loss(y_pred, y)

        y_pred_01 = _from11_to01(y_pred)
        y_01 = _from11_to01(y)

        ssim_loss = 1.0 - self.ssim(y_pred_01, y_01)

        total = loss_l1 + self.lambda_ssim * ssim_loss

        return total


import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable variant of L1 Loss)"""
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate element-wise loss
        loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        
        # Apply specified reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'mean', 'sum', or 'none'.")
