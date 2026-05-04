"""Evaluation CLI over image folders or CIFAR-10 without implicit downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

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


def _plot_curves_enabled(eval_cfg: dict[str, Any]) -> bool:
    artifacts_cfg = dict(eval_cfg.get("artifacts", {}) or {})
    if "plot_curves" not in artifacts_cfg:
        return True
    return bool(artifacts_cfg.get("plot_curves"))


def _draw_comparison_curve(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    y_label: str,
    deepsc_key: str,
    jpeg_key: str,
) -> None:
    width, height = 800, 480
    margin_left, margin_top, margin_right, margin_bottom = 80, 40, 30, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black")
    draw.text((plot_left, 12), title, fill="black")
    draw.text((width // 2 - 20, height - 35), "snr_db", fill="black")
    draw.text((12, plot_top), y_label, fill="black")

    if not rows:
        image.save(path)
        return

    sorted_rows = sorted(rows, key=lambda row: float(row["snr_db"]))
    snr_values = [float(row["snr_db"]) for row in sorted_rows]
    deepsc_values = [float(row[deepsc_key]) for row in sorted_rows]
    jpeg_values = [float(row[jpeg_key]) for row in sorted_rows]

    x_min = min(snr_values)
    x_max = max(snr_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    x_span = x_max - x_min

    y_min = min(min(deepsc_values), min(jpeg_values))
    y_max = max(max(deepsc_values), max(jpeg_values))
    if y_min == y_max:
        pad = 0.1 if y_min == 0 else abs(y_min) * 0.1
        y_min -= pad
        y_max += pad
    y_span = y_max - y_min

    def _project_point(snr_db: float, value: float) -> tuple[float, float]:
        x = plot_left + (plot_right - plot_left) * (snr_db - x_min) / x_span
        y = plot_bottom - (plot_bottom - plot_top) * (value - y_min) / y_span
        return x, y

    for tick in range(5):
        y = plot_bottom - (plot_bottom - plot_top) * tick / 4
        value = y_min + y_span * tick / 4
        draw.line((plot_left - 5, y, plot_left, y), fill="black")
        draw.text((8, y - 7), f"{value:.4g}", fill="black")
    for tick in range(5):
        x = plot_left + (plot_right - plot_left) * tick / 4
        value = x_min + x_span * tick / 4
        draw.line((x, plot_bottom, x, plot_bottom + 5), fill="black")
        draw.text((x - 12, plot_bottom + 10), f"{value:.4g}", fill="black")

    deep_points = [_project_point(snr_db, metric) for snr_db, metric in zip(snr_values, deepsc_values)]
    jpeg_points = [_project_point(snr_db, metric) for snr_db, metric in zip(snr_values, jpeg_values)]

    deep_color = "blue"
    jpeg_color = "red"
    if len(deep_points) == 1:
        x, y = deep_points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=deep_color)
    else:
        draw.line(deep_points, fill=deep_color, width=3)
        for x, y in deep_points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=deep_color)

    if len(jpeg_points) == 1:
        x, y = jpeg_points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=jpeg_color)
    else:
        draw.line(jpeg_points, fill=jpeg_color, width=3)
        for x, y in jpeg_points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=jpeg_color)

    legend_x = plot_right - 170
    legend_y = plot_top + 12
    draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=deep_color, width=3)
    draw.text((legend_x + 35, legend_y), "DeepSC", fill="black")
    draw.line((legend_x, legend_y + 30, legend_x + 28, legend_y + 30), fill=jpeg_color, width=3)
    draw.text((legend_x + 35, legend_y + 22), "JPEG baseline", fill="black")

    image.save(path)


def _write_curve_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    _draw_comparison_curve(
        output_dir / "psnr_vs_snr.png",
        rows,
        title="PSNR vs SNR (DeepSC vs JPEG)",
        y_label="psnr_db",
        deepsc_key="deepsc_psnr",
        jpeg_key="jpeg_psnr",
    )
    _draw_comparison_curve(
        output_dir / "ssim_vs_snr.png",
        rows,
        title="SSIM vs SNR (DeepSC vs JPEG)",
        y_label="ssim",
        deepsc_key="deepsc_ssim",
        jpeg_key="jpeg_ssim",
    )


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
    if _plot_curves_enabled(eval_cfg):
        _write_curve_artifacts(output_dir, results)
    return results


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
