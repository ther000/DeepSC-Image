"""Training objectives for DeepSC image reconstruction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .metrics import ssim_torch


class MixedMseSsimLoss(nn.Module):
    """Weighted MSE + (1 - SSIM) reconstruction objective."""

    def __init__(self, ssim_weight: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= ssim_weight <= 1.0:
            raise ValueError("ssim_weight must be between 0 and 1")
        self.ssim_weight = ssim_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(prediction, target)
        ssim_loss = 1.0 - ssim_torch(prediction.clamp(0, 1), target.clamp(0, 1))
        return (1.0 - self.ssim_weight) * mse + self.ssim_weight * ssim_loss
