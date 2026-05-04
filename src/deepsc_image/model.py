"""Compact convolution-attention DeepSC image model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .channels import ChannelConfig, apply_channel, normalize_average_power


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    """Lightweight channel attention for semantic feature fusion."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool2d(x, output_size=1)
        weights = F.silu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights


class SpatialAttention(nn.Module):
    """Small spatial attention gate to emphasize visually important regions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        max_values = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * gate


class SemanticEncoder(nn.Module):
    def __init__(self, semantic_channels: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(3, base_channels)
        self.down1 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.att1 = SEBlock(base_channels * 2)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.att2 = nn.Sequential(SEBlock(base_channels * 4), SpatialAttention())
        self.project = nn.Conv2d(base_channels * 4, semantic_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.att1(self.down1(x))
        x = self.att2(self.down2(x))
        return normalize_average_power(self.project(x))


class SemanticDecoder(nn.Module):
    def __init__(self, semantic_channels: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.expand = ConvBlock(semantic_channels, base_channels * 4)
        self.att = SEBlock(base_channels * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.SiLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = None) -> torch.Tensor:
        x = self.att(self.expand(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.out(x)
        if output_size is not None and x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class DeepSCImageModel(nn.Module):
    """End-to-end semantic image transmitter with pluggable wireless channel."""

    def __init__(self, semantic_channels: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = SemanticEncoder(semantic_channels=semantic_channels, base_channels=base_channels)
        self.decoder = SemanticDecoder(semantic_channels=semantic_channels, base_channels=base_channels)

    def forward(self, image: torch.Tensor, channel: ChannelConfig | None = None) -> torch.Tensor:
        output_size = image.shape[-2:]
        symbols = self.encoder(image)
        received = apply_channel(symbols, channel)
        return self.decoder(received, output_size=output_size)
