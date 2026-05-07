from __future__ import annotations

import pytest
import torch
from PIL import Image

from deepsc_image.baseline import bpg_available, bpg_baseline_tensor, bpg_roundtrip
from deepsc_image.channels import ChannelConfig


@pytest.mark.skipif(not bpg_available(), reason="BPG command line tools are not available")
def test_bpg_roundtrip_returns_rgb_image() -> None:
    image = Image.new("RGB", (32, 32), color=(64, 128, 192))

    decoded = bpg_roundtrip(image, qp=35)

    assert decoded.mode == "RGB"
    assert decoded.size == image.size


@pytest.mark.skipif(not bpg_available(), reason="BPG command line tools are not available")
def test_bpg_baseline_tensor_preserves_shape() -> None:
    image = torch.rand(1, 3, 32, 32)

    baseline = bpg_baseline_tensor(image, ChannelConfig("none", 10.0), qp=35)

    assert baseline.shape == image.shape
    assert torch.isfinite(baseline).all()
