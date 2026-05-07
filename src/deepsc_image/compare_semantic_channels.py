"""Plot PSNR/SSIM curves across models with different semantic channels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DeepSC models by semantic channel count")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="One or more metrics.json files or directories containing metrics.json",
    )
    parser.add_argument("--output-dir", default="outputs/semantic_channel_compare")
    parser.add_argument("--prefix", default="semantic_channels")
    return parser.parse_args(argv)


def _metrics_paths(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            found.extend(sorted(path.rglob("metrics.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"Metrics path not found: {path}")
    unique: list[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise ValueError("No metrics.json files found")
    return unique


def _semantic_channels_from_path(path: Path) -> int | None:
    for part in [path.parent.name, *path.parts]:
        match = re.search(r"(?:^|__)sem_(\d+)(?:__|$)", part)
        if match:
            return int(match.group(1))
    return None


def _semantic_channels_from_row(row: dict[str, Any], path: Path) -> int:
    value = row.get("semantic_channels")
    if value is not None:
        return int(value)

    bandwidth = row.get("bandwidth_estimate")
    if isinstance(bandwidth, dict):
        semantic_shape = bandwidth.get("semantic_shape")
        if isinstance(semantic_shape, list) and semantic_shape:
            return int(semantic_shape[0])

    path_value = _semantic_channels_from_path(path)
    if path_value is not None:
        return path_value

    raise ValueError(f"Cannot determine semantic_channels for {path}")


def load_comparison_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in _metrics_paths(paths):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {metrics_path}")
        for raw_row in data:
            if not isinstance(raw_row, dict):
                continue
            semantic_channels = _semantic_channels_from_row(raw_row, metrics_path)
            rows.append(
                {
                    "semantic_channels": semantic_channels,
                    "snr_db": float(raw_row["snr_db"]),
                    "channel": str(raw_row.get("channel", "")),
                    "deepsc_psnr": float(raw_row["deepsc_psnr"]),
                    "deepsc_ssim": float(raw_row["deepsc_ssim"]),
                    "source": str(metrics_path),
                }
            )
    if not rows:
        raise ValueError("No comparison rows loaded")
    return sorted(rows, key=lambda row: (int(row["semantic_channels"]), float(row["snr_db"])))


def _group_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["semantic_channels"]), []).append(row)
    return {
        semantic_channels: sorted(group, key=lambda item: float(item["snr_db"]))
        for semantic_channels, group in sorted(grouped.items())
    }


def _comparison_title(metric: str, rows: list[dict[str, Any]]) -> str:
    channels = sorted({str(row.get("channel", "")).upper() for row in rows if row.get("channel")})
    if len(channels) == 1:
        return f"{metric} vs SNR\nchannel={channels[0]}"
    return f"{metric} vs SNR"


def _draw_metric_curve(
    output_base: Path,
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    metric_label: str,
    y_label: str,
) -> None:
    grouped = _group_rows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    markers = ["o", "s", "^", "D", "x", "*"]

    for index, (semantic_channels, group) in enumerate(grouped.items()):
        snr_values = [float(row["snr_db"]) for row in group]
        metric_values = [float(row[metric_key]) for row in group]
        ax.plot(
            snr_values,
            metric_values,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linewidth=2,
            markersize=4,
            label=f"semantic_channels={semantic_channels}",
        )

    ax.set_title(_comparison_title(metric_label, rows), fontsize=12)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_comparison_csv(output_dir: Path, rows: list[dict[str, Any]], prefix: str) -> None:
    path = output_dir / f"{prefix}_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["semantic_channels", "snr_db", "channel", "deepsc_psnr", "deepsc_ssim", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_artifacts(output_dir: Path, rows: list[dict[str, Any]], prefix: str = "semantic_channels") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_csv(output_dir, rows, prefix)
    _draw_metric_curve(
        output_dir / f"{prefix}_psnr_vs_snr",
        rows,
        metric_key="deepsc_psnr",
        metric_label="PSNR",
        y_label="PSNR (dB)",
    )
    _draw_metric_curve(
        output_dir / f"{prefix}_ssim_vs_snr",
        rows,
        metric_key="deepsc_ssim",
        metric_label="SSIM",
        y_label="SSIM",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_comparison_rows(args.metrics)
    write_comparison_artifacts(Path(args.output_dir), rows, prefix=str(args.prefix))


if __name__ == "__main__":
    main()
