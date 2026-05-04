"""Evaluation CLI over image folders or CIFAR-10 without implicit downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .baseline import jpeg_baseline_tensor
from .channels import ChannelConfig
from .data import build_dataset, unwrap_batch
from .metrics import tensor_metrics
from .model import DeepSCImageModel
from .utils import estimate_semantic_bandwidth, load_checkpoint, load_yaml, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepSC image model")
    parser.add_argument("--config", default="configs/eval_kodak.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def _image_shape(dataset_cfg: dict[str, Any]) -> tuple[int, int, int]:
    image_size = int(dataset_cfg.get("image_size", 256))
    return (3, image_size, image_size)


def run_evaluation(cfg: dict[str, Any], checkpoint: str | None = None) -> list[dict[str, Any]]:
    device = resolve_device(cfg.get("device", "auto"))
    dataset = build_dataset(cfg.get("dataset", {}), train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model_cfg = cfg.get("model", {})
    semantic_channels = int(model_cfg.get("semantic_channels", 32))
    model = DeepSCImageModel(
        semantic_channels=semantic_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)
    checkpoint_path = checkpoint or cfg.get("checkpoint")
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, device)
    else:
        print("No checkpoint supplied; evaluating random initialized model for pipeline validation only.")
    model.eval()
    snr_values = [float(v) for v in cfg.get("channel", {}).get("snr_db", [10])]
    channel_type = str(cfg.get("channel", {}).get("type", "awgn"))
    eval_cfg = cfg.get("evaluation", {})
    monte_carlo_samples = int(eval_cfg.get("monte_carlo_samples", 1))
    if monte_carlo_samples <= 0:
        raise ValueError("evaluation.monte_carlo_samples must be a positive integer")
    jpeg_quality = int(cfg.get("baseline", {}).get("jpeg_quality", 35))
    bandwidth = estimate_semantic_bandwidth(semantic_channels, input_shape=_image_shape(cfg.get("dataset", {})))
    results = []
    with torch.no_grad():
        for snr_db in snr_values:
            deep_psnr = deep_ssim = base_psnr = base_ssim = 0.0
            count = 0
            repetitions = 0
            channel = ChannelConfig(channel_type=channel_type, snr_db=snr_db)
            for step, batch in enumerate(loader, start=1):
                images = unwrap_batch(batch).to(device)
                for _ in range(monte_carlo_samples):
                    reconstruction = model(images, channel=channel).clamp(0, 1)
                    baseline = jpeg_baseline_tensor(images, channel, quality=jpeg_quality)
                    deep_metrics = tensor_metrics(images.cpu(), reconstruction.cpu())
                    base_metrics = tensor_metrics(images.cpu(), baseline.cpu())
                    deep_psnr += deep_metrics["psnr"]
                    deep_ssim += deep_metrics["ssim"]
                    base_psnr += base_metrics["psnr"]
                    base_ssim += base_metrics["ssim"]
                    repetitions += 1
                count += 1
            if count == 0 or repetitions == 0:
                raise RuntimeError(
                    "Evaluation processed zero samples; check dataset path and image files."
                )
            row = {
                "snr_db": snr_db,
                "channel": channel_type,
                "deepsc_psnr": deep_psnr / repetitions,
                "deepsc_ssim": deep_ssim / repetitions,
                "jpeg_psnr": base_psnr / repetitions,
                "jpeg_ssim": base_ssim / repetitions,
                "samples": count,
                "monte_carlo_samples": monte_carlo_samples,
                "repetitions": repetitions,
                "bandwidth_estimate": bandwidth,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))
    output_dir = Path(cfg.get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    run_evaluation(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
