"""Evaluation CLI over image folders or CIFAR-10 without implicit downloads."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baseline import VALID_BASELINE_CODECS, baseline_tensor
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
    parser.add_argument("--baseline-codec", choices=VALID_BASELINE_CODECS, nargs="+", default=None)
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--bpg-qp", type=int, default=None)
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
    if args.baseline_codec is not None:
        set_nested(cfg, ["baseline", "codecs"], list(args.baseline_codec))
        set_nested(cfg, ["baseline", "codec"], args.baseline_codec[0])
    if args.jpeg_quality is not None:
        set_nested(cfg, ["baseline", "jpeg_quality"], args.jpeg_quality)
    if args.bpg_qp is not None:
        set_nested(cfg, ["baseline", "bpg_qp"], args.bpg_qp)
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


def _safe_name_part(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    return text.strip("-_.") or "na"


def _evaluation_output_dir(base_output_dir: Path, checkpoint_path: str | Path | None) -> Path:
    if not checkpoint_path:
        return base_output_dir

    checkpoint = Path(checkpoint_path)
    model_part = _safe_name_part(checkpoint.parent.name)
    checkpoint_part = _safe_name_part(checkpoint.stem)
    return base_output_dir / f"{model_part}__ckpt_{checkpoint_part}"


def _draw_comparison_curve(
    output_base: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    y_label: str,
    deepsc_key: str,
    baseline_keys: list[tuple[str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))

    if rows:
        sorted_rows = sorted(rows, key=lambda row: float(row["snr_db"]))
        snr_values = [float(row["snr_db"]) for row in sorted_rows]
        deepsc_values = [float(row[deepsc_key]) for row in sorted_rows]

        ax.plot(snr_values, deepsc_values, color="blue", marker="o", linewidth=2, markersize=4, label="DeepSC")
        colors = ["red", "green", "purple", "orange"]
        markers = ["s", "^", "D", "x"]
        for index, (key, label) in enumerate(baseline_keys):
            values = [float(row[key]) for row in sorted_rows if key in row]
            if len(values) != len(snr_values):
                continue
            ax.plot(
                snr_values,
                values,
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                linewidth=2,
                markersize=4,
                label=label,
            )
        ax.legend(loc="best", fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _format_curve_title(metric: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{metric} vs SNR"

    first = rows[0]
    semantic_channels = first.get("semantic_channels")
    jpeg_quality = first.get("jpeg_quality")
    bpg_qp = first.get("bpg_qp")

    parts = []
    if semantic_channels is not None:
        parts.append(f"semantic_channels={semantic_channels}")
    if "jpeg" in _result_baseline_codecs(rows) and jpeg_quality is not None:
        parts.append(f"JPEG Q={jpeg_quality}")
    if "bpg" in _result_baseline_codecs(rows) and bpg_qp is not None:
        parts.append(f"BPG QP={bpg_qp}")

    if not parts:
        return f"{metric} vs SNR"
    return f"{metric} vs SNR\n" + ", ".join(parts)


def _write_curve_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    codecs = _result_baseline_codecs(rows)
    psnr_keys = [(f"{codec}_psnr", f"{codec.upper()} baseline") for codec in codecs]
    ssim_keys = [(f"{codec}_ssim", f"{codec.upper()} baseline") for codec in codecs]
    _draw_comparison_curve(
        output_dir / "psnr_vs_snr",
        rows,
        title=_format_curve_title("PSNR", rows),
        y_label="PSNR (dB)",
        deepsc_key="deepsc_psnr",
        baseline_keys=psnr_keys,
    )
    _draw_comparison_curve(
        output_dir / "ssim_vs_snr",
        rows,
        title=_format_curve_title("SSIM", rows),
        y_label="SSIM",
        deepsc_key="deepsc_ssim",
        baseline_keys=ssim_keys,
    )


def _baseline_codecs(baseline_cfg: dict[str, Any]) -> list[str]:
    raw_codecs = baseline_cfg.get("codecs", baseline_cfg.get("codec", "jpeg"))
    if isinstance(raw_codecs, str):
        codecs = [raw_codecs]
    else:
        codecs = [str(codec) for codec in raw_codecs]
    normalized = []
    for codec in codecs:
        codec_name = codec.lower().strip()
        if codec_name not in VALID_BASELINE_CODECS:
            raise ValueError(f"Unsupported baseline codec: {codec!r}. Expected one of {VALID_BASELINE_CODECS}.")
        if codec_name not in normalized:
            normalized.append(codec_name)
    return normalized or ["jpeg"]


def _result_baseline_codecs(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["jpeg"]
    first = rows[0]
    raw_codecs = first.get("baseline_codecs", first.get("baseline_codec", "jpeg"))
    if isinstance(raw_codecs, str):
        return [raw_codecs]
    return [str(codec) for codec in raw_codecs]


def _write_table_artifacts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    codecs = _result_baseline_codecs(rows)
    fieldnames = ["snr_db", "channel", "deepsc_psnr", "deepsc_ssim"]
    for codec in codecs:
        fieldnames.extend([f"{codec}_psnr", f"{codec}_ssim"])
    fieldnames.extend(["samples", "monte_carlo_samples", "repetitions", "bandwidth_estimate"])

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    headers = ["SNR(dB)", "DeepSC PSNR", "DeepSC SSIM"]
    for codec in codecs:
        headers.extend([f"{codec.upper()} PSNR", f"{codec.upper()} SSIM"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in sorted(rows, key=lambda item: float(item["snr_db"])):
        values = [
            f"{float(row['snr_db']):.0f}",
            f"{float(row['deepsc_psnr']):.2f}",
            f"{float(row['deepsc_ssim']):.4f}",
        ]
        for codec in codecs:
            values.extend([
                f"{float(row[f'{codec}_psnr']):.2f}",
                f"{float(row[f'{codec}_ssim']):.4f}",
            ])
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "thesis_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    baseline_cfg = cfg.get("baseline", {})
    baseline_codecs = _baseline_codecs(baseline_cfg)
    jpeg_quality = int(baseline_cfg.get("jpeg_quality", 95))
    bpg_qp = int(baseline_cfg.get("bpg_qp", 0))
    bandwidth = estimate_semantic_bandwidth(semantic_channels, input_shape=_image_shape(cfg.get("dataset", {})))
    results = []
    with torch.no_grad():
        for snr_db in snr_values:
            deep_psnr = deep_ssim = 0.0
            baseline_sums = {
                codec: {"psnr": 0.0, "ssim": 0.0}
                for codec in baseline_codecs
            }
            count = 0
            repetitions = 0
            channel = ChannelConfig(channel_type=channel_type, snr_db=snr_db)
            for step, batch in enumerate(loader, start=1):
                images = unwrap_batch(batch).to(device)
                for _ in range(monte_carlo_samples):
                    reconstruction = model(images, channel=channel).clamp(0, 1)
                    deep_metrics = tensor_metrics(images.cpu(), reconstruction.cpu())
                    deep_psnr += deep_metrics["psnr"]
                    deep_ssim += deep_metrics["ssim"]
                    for codec in baseline_codecs:
                        baseline = baseline_tensor(
                            images,
                            channel,
                            codec=codec,
                            jpeg_quality=jpeg_quality,
                            bpg_qp=bpg_qp,
                        )
                        base_metrics = tensor_metrics(images.cpu(), baseline.cpu())
                        baseline_sums[codec]["psnr"] += base_metrics["psnr"]
                        baseline_sums[codec]["ssim"] += base_metrics["ssim"]
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
                "baseline_codecs": baseline_codecs,
                "semantic_channels": semantic_channels,
                "image_size": int(cfg.get("dataset", {}).get("image_size", 256)),
                "jpeg_quality": jpeg_quality,
                "bpg_qp": bpg_qp,
                "samples": count,
                "monte_carlo_samples": monte_carlo_samples,
                "repetitions": repetitions,
                "bandwidth_estimate": bandwidth,
            }
            for codec in baseline_codecs:
                row[f"{codec}_psnr"] = baseline_sums[codec]["psnr"] / repetitions
                row[f"{codec}_ssim"] = baseline_sums[codec]["ssim"] / repetitions
            first_codec = baseline_codecs[0]
            row["baseline_codec"] = first_codec
            row["baseline_psnr"] = row[f"{first_codec}_psnr"]
            row["baseline_ssim"] = row[f"{first_codec}_ssim"]
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))
    output_dir = _evaluation_output_dir(Path(cfg.get("output_dir", "outputs/eval")), checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_table_artifacts(output_dir, results)
    if _plot_curves_enabled(eval_cfg):
        _write_curve_artifacts(output_dir, results)
    return results


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
