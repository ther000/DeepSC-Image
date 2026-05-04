"""PSNR and SSIM metrics implemented with lightweight PyTorch code."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    one_d = torch.exp(-(coords.pow(2)) / (2 * sigma * sigma))
    one_d = one_d / one_d.sum()
    two_d = one_d[:, None] @ one_d[None, :]
    return two_d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_torch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Compute mean structural similarity for NCHW tensors in [0, 1]."""

    if x.shape != y.shape:
        raise ValueError(f"SSIM expects matching shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.dim() != 4:
        raise ValueError("SSIM expects NCHW tensors")
    channels = x.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-8)).mean()


def psnr_torch(x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(x, y).clamp_min(1e-12)
    return 20 * torch.log10(torch.tensor(max_value, device=x.device, dtype=x.dtype)) - 10 * torch.log10(mse)


def tensor_metrics(reference: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        psnr = psnr_torch(reconstruction.clamp(0, 1), reference.clamp(0, 1)).item()
        ssim = ssim_torch(reconstruction.clamp(0, 1), reference.clamp(0, 1)).item()
    if not math.isfinite(psnr):
        psnr = 99.0
    return {"psnr": float(psnr), "ssim": float(ssim)}
