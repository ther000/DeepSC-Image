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
from .interactive_cli import apply_eval_interactive_config, set_nested
from .utils import estimate_semantic_bandwidth, load_checkpoint, load_yaml, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepSC image model")
    parser.add_argument("--config", default="configs/eval_kodak.yaml")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for evaluation parameters")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", choices=("cifar10", "image_folder"), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--semantic-channels", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--channel", choices=("none", "awgn", "rayleigh"), default=None)
    parser.add_argument("--snr-db", type=float, nargs="+", default=None, help="One or more evaluation SNR values")
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--monte-carlo-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    if args.checkpoint is not None:
        set_nested(cfg, ["checkpoint"], args.checkpoint)
    if args.device is not None:
        set_nested(cfg, ["device"], args.device)
    if args.dataset is not None:
        set_nested(cfg, ["dataset", "name"], args.dataset)
    if args.data_root is not None:
        set_nested(cfg, ["dataset", "root"], args.data_root)
    if args.image_size is not None:
        set_nested(cfg, ["dataset", "image_size"], args.image_size)
    if args.semantic_channels is not None:
        set_nested(cfg, ["model", "semantic_channels"], args.semantic_channels)
    if args.base_channels is not None:
        set_nested(cfg, ["model", "base_channels"], args.base_channels)
    if args.channel is not None:
        set_nested(cfg, ["channel", "type"], args.channel)
    if args.snr_db is not None:
        set_nested(cfg, ["channel", "snr_db"], list(args.snr_db))
    if args.jpeg_quality is not None:
        set_nested(cfg, ["baseline", "jpeg_quality"], args.jpeg_quality)
    if args.monte_carlo_samples is not None:
        set_nested(cfg, ["evaluation", "monte_carlo_samples"], args.monte_carlo_samples)
    if args.output_dir is not None:
        set_nested(cfg, ["output_dir"], args.output_dir)
    if args.interactive:
        apply_eval_interactive_config(cfg)
    return cfg


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
    cfg = build_config_from_args(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
