"""Inference helpers shared by CLI and Streamlit GUI."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .baseline import baseline_tensor
from .channels import ChannelConfig
from .metrics import tensor_metrics
from .model import DeepSCImageModel


@dataclass(frozen=True)
class InferenceResult:
    reconstruction: torch.Tensor
    baseline: torch.Tensor
    deepsc_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    latency_ms: float
    baseline_latency_ms: float
    baseline_codec: str


def run_inference(
    model: DeepSCImageModel,
    image: torch.Tensor,
    channel: ChannelConfig,
    jpeg_quality: int = 95,
    baseline_codec: str = "jpeg",
    bpg_qp: int = 0,
    device: torch.device | None = None,
) -> InferenceResult:
    target_device = device or next(model.parameters()).device
    model.eval()
    image = image.to(target_device)
    start = time.perf_counter()
    with torch.no_grad():
        reconstruction = model(image, channel=channel).clamp(0, 1)
    latency_ms = (time.perf_counter() - start) * 1000.0

    start_baseline = time.perf_counter()
    baseline = baseline_tensor(
        image,
        channel,
        codec=baseline_codec,
        jpeg_quality=jpeg_quality,
        bpg_qp=bpg_qp,
    ).clamp(0, 1)
    baseline_latency_ms = (time.perf_counter() - start_baseline) * 1000.0

    return InferenceResult(
        reconstruction=reconstruction.detach().cpu(),
        baseline=baseline.detach().cpu(),
        deepsc_metrics=tensor_metrics(image.detach().cpu(), reconstruction.detach().cpu()),
        baseline_metrics=tensor_metrics(image.detach().cpu(), baseline.detach().cpu()),
        latency_ms=latency_ms,
        baseline_latency_ms=baseline_latency_ms,
        baseline_codec=baseline_codec,
    )
