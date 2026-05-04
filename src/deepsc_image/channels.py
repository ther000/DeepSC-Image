"""Differentiable wireless channel layers for semantic image transmission."""

from __future__ import annotations

from dataclasses import dataclass

import torch


VALID_CHANNELS = {"none", "awgn", "rayleigh"}


@dataclass(frozen=True)
class ChannelConfig:
    """Runtime channel settings.

    Attributes:
        channel_type: One of ``none``, ``awgn`` or ``rayleigh``.
        snr_db: Signal-to-noise ratio in dB. The GUI and configs expose -5..20 dB.
    """

    channel_type: str = "awgn"
    snr_db: float = 10.0

    def normalized_type(self) -> str:
        name = self.channel_type.lower()
        if name not in VALID_CHANNELS:
            raise ValueError(f"Unsupported channel '{self.channel_type}'. Expected one of {sorted(VALID_CHANNELS)}")
        return name


def normalize_average_power(symbols: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each sample to unit average power while preserving gradients."""

    if symbols.dim() < 2:
        raise ValueError("Channel symbols must include batch and feature dimensions")
    reduce_dims = tuple(range(1, symbols.dim()))
    power = symbols.pow(2).mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return symbols / torch.sqrt(power)


def _noise_for_snr(symbols: torch.Tensor, snr_db: float, eps: float = 1e-8) -> torch.Tensor:
    reduce_dims = tuple(range(1, symbols.dim()))
    signal_power = symbols.pow(2).mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return torch.randn_like(symbols) * torch.sqrt(noise_power)


def awgn_channel(symbols: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Apply additive white Gaussian noise at the requested SNR."""

    return symbols + _noise_for_snr(symbols, snr_db)


def rayleigh_channel(symbols: torch.Tensor, snr_db: float, eps: float = 1e-8) -> torch.Tensor:
    """Apply flat Rayleigh fading with simple perfect channel equalization."""

    batch_shape = (symbols.shape[0],) + (1,) * (symbols.dim() - 1)
    real = torch.randn(batch_shape, device=symbols.device, dtype=symbols.dtype)
    imag = torch.randn(batch_shape, device=symbols.device, dtype=symbols.dtype)
    fading = torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(eps) / (2.0 ** 0.5)
    faded = symbols * fading
    noisy = faded + _noise_for_snr(faded, snr_db)
    return noisy / fading


def apply_channel(symbols: torch.Tensor, config: ChannelConfig | None) -> torch.Tensor:
    """Dispatch symbols through the selected differentiable channel."""

    if config is None or config.normalized_type() == "none":
        return symbols
    channel_type = config.normalized_type()
    if channel_type == "awgn":
        return awgn_channel(symbols, config.snr_db)
    if channel_type == "rayleigh":
        return rayleigh_channel(symbols, config.snr_db)
    raise ValueError(f"Unsupported channel type: {channel_type}")
