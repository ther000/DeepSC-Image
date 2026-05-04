from __future__ import annotations

import torch

from deepsc_image.channels import ChannelConfig
from deepsc_image.inference import run_inference
from deepsc_image.model import DeepSCImageModel


def test_run_inference_returns_latency() -> None:
    model = DeepSCImageModel(semantic_channels=4, base_channels=8)
    image = torch.rand(1, 3, 32, 32)
    channel = ChannelConfig("none", 10.0)

    result = run_inference(model, image, channel)

    assert hasattr(result, "latency_ms")
    assert hasattr(result, "baseline_latency_ms")
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.baseline_latency_ms, float)
    assert result.latency_ms >= 0.0
    assert result.baseline_latency_ms >= 0.0
