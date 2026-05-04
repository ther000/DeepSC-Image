"""Traditional JPEG baseline with channel-like visual degradation."""

from __future__ import annotations

from io import BytesIO

import torch
from PIL import Image, ImageFilter

from .channels import ChannelConfig
from .utils import pil_to_tensor, tensor_to_pil


def jpeg_roundtrip(image: Image.Image, quality: int = 35) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade_tensor_like_channel(tensor: torch.Tensor, config: ChannelConfig) -> torch.Tensor:
    """Approximate modulation/channel artifacts for visual baseline comparison."""

    channel = config.normalized_type()
    if channel == "none":
        return tensor.clamp(0, 1)
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    noise_std = (1.0 / max(snr_linear, 1e-6)) ** 0.5 * 0.18
    noisy = tensor + torch.randn_like(tensor) * noise_std
    if channel == "rayleigh":
        fade_shape = (tensor.shape[0], 1, 1, 1) if tensor.dim() == 4 else (1, 1, 1)
        real = torch.randn(fade_shape, device=tensor.device, dtype=tensor.dtype)
        imag = torch.randn(fade_shape, device=tensor.device, dtype=tensor.dtype)
        fading = torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(0.2) / (2.0 ** 0.5)
        noisy = noisy * fading
    return noisy.clamp(0, 1)


def jpeg_baseline_tensor(tensor: torch.Tensor, config: ChannelConfig, quality: int = 35) -> torch.Tensor:
    """Run an in-memory JPEG encode/decode then add channel-like degradation."""

    if tensor.dim() != 4:
        raise ValueError("JPEG baseline expects NCHW tensor")
    outputs = []
    for sample in tensor.detach().cpu():
        pil = tensor_to_pil(sample.unsqueeze(0))
        jpeg = jpeg_roundtrip(pil, quality=quality)
        outputs.append(pil_to_tensor(jpeg))
    stacked = torch.cat(outputs, dim=0).to(device=tensor.device, dtype=tensor.dtype)
    degraded = degrade_tensor_like_channel(stacked, config)
    if config.normalized_type() == "rayleigh":
        pil_outputs = []
        for sample in degraded.detach().cpu():
            pil_outputs.append(pil_to_tensor(tensor_to_pil(sample.unsqueeze(0)).filter(ImageFilter.GaussianBlur(radius=0.35))))
        degraded = torch.cat(pil_outputs, dim=0).to(device=tensor.device, dtype=tensor.dtype)
    return degraded.clamp(0, 1)
