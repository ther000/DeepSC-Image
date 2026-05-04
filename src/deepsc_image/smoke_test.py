"""CPU smoke test that does not download datasets."""

from __future__ import annotations

import torch

from .baseline import jpeg_baseline_tensor
from .channels import ChannelConfig, apply_channel
from .losses import MixedMseSsimLoss
from .metrics import tensor_metrics
from .model import DeepSCImageModel
from .utils import set_seed


def main() -> None:
    set_seed(7)
    device = torch.device("cpu")
    images = torch.rand(2, 3, 32, 32, device=device)
    model = DeepSCImageModel(semantic_channels=8, base_channels=8).to(device)
    loss_fn = MixedMseSsimLoss(ssim_weight=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for channel_type in ("none", "awgn", "rayleigh"):
        channel = ChannelConfig(channel_type=channel_type, snr_db=5.0)
        symbols = model.encoder(images)
        received = apply_channel(symbols, channel)
        assert received.shape == symbols.shape
        reconstruction = model(images, channel=channel)
        assert reconstruction.shape == images.shape
        metrics = tensor_metrics(images, reconstruction)
        assert "psnr" in metrics and "ssim" in metrics
    baseline = jpeg_baseline_tensor(images, ChannelConfig("awgn", 5.0), quality=40)
    assert baseline.shape == images.shape
    optimizer.zero_grad(set_to_none=True)
    prediction = model(images, channel=ChannelConfig("awgn", 10.0))
    loss = loss_fn(prediction, images)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item()
    print(f"SMOKE_TEST_OK loss={loss.item():.6f} baseline_shape={tuple(baseline.shape)}")


if __name__ == "__main__":
    main()
