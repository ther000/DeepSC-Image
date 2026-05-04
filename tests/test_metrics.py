from __future__ import annotations

import math

import torch

from deepsc_image.metrics import tensor_metrics


def test_tensor_metrics_accepts_mixed_float_dtypes() -> None:
    reference = torch.rand(2, 3, 16, 16, dtype=torch.float32)
    reconstruction = reference.to(dtype=torch.float16)

    metrics = tensor_metrics(reference, reconstruction)

    assert math.isfinite(metrics["psnr"])
    assert math.isfinite(metrics["ssim"])
