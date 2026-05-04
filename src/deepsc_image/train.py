"""Training CLI: python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import shutil
from collections.abc import Sized
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml
from PIL import Image, ImageDraw

from .channels import ChannelConfig
from .data import build_dataset, unwrap_batch
from .losses import MixedMseSsimLoss
from .metrics import tensor_metrics
from .model import DeepSCImageModel
from .interactive_cli import apply_train_interactive_config, set_nested
from .utils import estimate_semantic_bandwidth, load_yaml, resolve_device, save_checkpoint, set_seed


DEFAULT_AMP_CONFIG: dict[str, Any] = {"enabled": False, "dtype": "float16"}
DEFAULT_DATALOADER_CONFIG: dict[str, Any] = {
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
    "prefetch_factor": None,
}
DEFAULT_ARTIFACT_CONFIG: dict[str, Any] = {
    "history_every_epochs": 1,
    "plot_every_epochs": 1,
    "checkpoint_every_epochs": 1,
}


def _checkpoint_payload(model: torch.nn.Module, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepSC image model")
    parser.add_argument("--config", default="configs/train_cifar10_awgn.yaml", help="YAML config path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for training parameters")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", choices=("cifar10", "image_folder"), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--semantic-channels", type=int, nargs="+", default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--channel", choices=("none", "awgn", "rayleigh"), default=None)
    parser.add_argument("--snr-db", type=float, nargs="+", default=None, help="One or more training SNR values")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--ssim-weight", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    if args.seed is not None:
        set_nested(cfg, ["seed"], args.seed)
    if args.device is not None:
        set_nested(cfg, ["device"], args.device)
    if args.dataset is not None:
        set_nested(cfg, ["dataset", "name"], args.dataset)
    if args.data_root is not None:
        set_nested(cfg, ["dataset", "root"], args.data_root)
    if args.image_size is not None:
        set_nested(cfg, ["dataset", "image_size"], args.image_size)
    if args.download:
        set_nested(cfg, ["dataset", "download"], True)
    if args.semantic_channels is not None:
        set_nested(cfg, ["model", "semantic_channels"], list(args.semantic_channels))
    if args.base_channels is not None:
        set_nested(cfg, ["model", "base_channels"], args.base_channels)
    if args.channel is not None:
        set_nested(cfg, ["channel", "type"], args.channel)
    if args.snr_db is not None:
        set_nested(cfg, ["channel", "train_snr_db"], list(args.snr_db))
    if args.epochs is not None:
        set_nested(cfg, ["training", "epochs"], args.epochs)
    if args.batch_size is not None:
        set_nested(cfg, ["training", "batch_size"], args.batch_size)
    if args.learning_rate is not None:
        set_nested(cfg, ["training", "learning_rate"], args.learning_rate)
    if args.weight_decay is not None:
        set_nested(cfg, ["training", "weight_decay"], args.weight_decay)
    if args.ssim_weight is not None:
        set_nested(cfg, ["training", "ssim_weight"], args.ssim_weight)
    if args.output_dir is not None:
        set_nested(cfg, ["training", "output_dir"], args.output_dir)
    if args.interactive:
        apply_train_interactive_config(cfg)
    return cfg


def _training_section(cfg: dict[str, Any]) -> dict[str, Any]:
    training = cfg.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training config must be a mapping")
    return training


def normalize_speed_config(cfg: dict[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    training = _training_section(cfg)

    amp_cfg = {**DEFAULT_AMP_CONFIG, **dict(training.get("amp", {}) or {})}
    requested_amp = bool(amp_cfg.get("enabled", False))
    device_type = device.type if device is not None else str(cfg.get("device", "auto"))
    amp_cfg["enabled"] = requested_amp
    amp_cfg["requested"] = requested_amp
    amp_cfg["effective"] = requested_amp and device_type == "cuda"
    amp_cfg["dtype"] = str(amp_cfg.get("dtype", "float16"))
    amp_cfg["device_type"] = device_type
    training["amp"] = amp_cfg

    dataloader_cfg = {**DEFAULT_DATALOADER_CONFIG, **dict(training.get("dataloader", {}) or {})}
    num_workers = int(dataloader_cfg.get("num_workers", 0) or 0)
    if num_workers < 0:
        raise ValueError("training.dataloader.num_workers must be non-negative")
    dataloader_cfg["num_workers"] = num_workers
    dataloader_cfg["pin_memory"] = bool(dataloader_cfg.get("pin_memory", False))
    if num_workers <= 0:
        dataloader_cfg["persistent_workers"] = False
        dataloader_cfg["prefetch_factor"] = None
    else:
        dataloader_cfg["persistent_workers"] = bool(dataloader_cfg.get("persistent_workers", False))
        prefetch_factor = dataloader_cfg.get("prefetch_factor")
        if prefetch_factor is None:
            dataloader_cfg["prefetch_factor"] = None
        else:
            prefetch_value = int(prefetch_factor)
            if prefetch_value <= 0:
                raise ValueError("training.dataloader.prefetch_factor must be a positive integer when set")
            dataloader_cfg["prefetch_factor"] = prefetch_value
    training["dataloader"] = dataloader_cfg

    artifact_cfg = {**DEFAULT_ARTIFACT_CONFIG, **dict(training.get("artifacts", {}) or {})}
    for key in DEFAULT_ARTIFACT_CONFIG:
        raw_value = artifact_cfg.get(key, 1)
        value = int(1 if raw_value is None else raw_value)
        if value <= 0:
            raise ValueError(f"training.artifacts.{key} must be a positive integer")
        artifact_cfg[key] = value
    training["artifacts"] = artifact_cfg
    return cfg


def _dataloader_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    dataloader_cfg = {**DEFAULT_DATALOADER_CONFIG, **dict(train_cfg.get("dataloader", {}) or {})}
    num_workers = int(dataloader_cfg.get("num_workers", 0) or 0)
    kwargs: dict[str, Any] = {
        "batch_size": int(train_cfg.get("batch_size", 64)),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": bool(dataloader_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(dataloader_cfg.get("persistent_workers", False))
        prefetch_factor = dataloader_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("training.amp.dtype must be one of: float16, bfloat16")


def _make_grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, amp_cfg: dict[str, Any]):
    enabled = bool(amp_cfg.get("effective", False))
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device.type, dtype=_amp_dtype(str(amp_cfg.get("dtype", "float16"))), enabled=True)


def _should_write_epoch(epoch: int, total_epochs: int, every_epochs: int) -> bool:
    return epoch == total_epochs or epoch % every_epochs == 0


def _semantic_channel_values(model_cfg: dict[str, Any]) -> list[int]:
    configured = model_cfg.get("semantic_channels", 32)
    if isinstance(configured, list):
        values = [int(value) for value in configured]
    else:
        values = [int(configured)]
    if not values or any(value <= 0 for value in values):
        raise ValueError("model.semantic_channels must be a positive integer or a non-empty list of positive integers")
    return values


def _image_shape(dataset_cfg: dict[str, Any]) -> tuple[int, int, int]:
    image_size = int(dataset_cfg.get("image_size", 32))
    return (3, image_size, image_size)


def _format_snr_value(value: float) -> str:
    text = f"{value:g}"
    text = text.replace("-", "m").replace("+", "")
    return text.replace(".", "p")


def _safe_name_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    return text.strip("-_.") or "na"


def _experiment_output_dir(
    base_output_dir: Path,
    *,
    channel_type: str,
    snr_values: list[float],
    semantic_channels: int,
    base_channels: int,
    image_size: int,
    seed: int,
    timestamp: str,
) -> Path:
    snr_part = "_".join(_format_snr_value(value) for value in snr_values)
    dirname = "__".join(
        [
            f"ts_{timestamp}",
            f"ch_{_safe_name_part(channel_type)}",
            f"snr_{snr_part}",
            f"sem_{semantic_channels}",
            f"base_{base_channels}",
            f"img_{image_size}",
            f"seed_{seed}",
        ]
    )
    return base_output_dir / dirname


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = ["epoch", "train_loss", "psnr", "ssim", "batches"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _draw_loss_curve(path: Path, history: list[dict[str, Any]]) -> None:
    width, height = 800, 480
    margin_left, margin_top, margin_right, margin_bottom = 80, 40, 30, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black")
    draw.text((plot_left, 12), "Training loss", fill="black")
    draw.text((width // 2 - 25, height - 35), "epoch", fill="black")
    draw.text((12, plot_top), "train_loss", fill="black")
    if not history:
        image.save(path)
        return

    losses = [float(row["train_loss"]) for row in history]
    min_loss = min(losses)
    max_loss = max(losses)
    if min_loss == max_loss:
        min_loss -= 0.5 if min_loss == 0 else abs(min_loss) * 0.1
        max_loss += 0.5 if max_loss == 0 else abs(max_loss) * 0.1
    x_span = max(1, len(history) - 1)
    y_span = max_loss - min_loss
    points: list[tuple[float, float]] = []
    for index, row in enumerate(history):
        x = plot_left + (plot_right - plot_left) * index / x_span
        y = plot_bottom - (float(row["train_loss"]) - min_loss) * (plot_bottom - plot_top) / y_span
        points.append((x, y))
    for tick in range(5):
        y = plot_bottom - (plot_bottom - plot_top) * tick / 4
        value = min_loss + y_span * tick / 4
        draw.line((plot_left - 5, y, plot_left, y), fill="black")
        draw.text((8, y - 7), f"{value:.4g}", fill="black")
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="blue")
    else:
        draw.line(points, fill="blue", width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")
    image.save(path)


def _write_history_artifacts(output_dir: Path, history: list[dict[str, Any]], *, include_plot: bool = True) -> None:
    _write_history_csv(output_dir / "history.csv", history)
    _write_json(output_dir / "history.json", history)
    if include_plot:
        _draw_loss_curve(output_dir / "loss_curve.png", history)


def _write_epoch_artifacts(
    output_dir: Path,
    history: list[dict[str, Any]],
    *,
    write_history: bool,
    write_plot: bool,
) -> None:
    if write_history:
        _write_history_csv(output_dir / "history.csv", history)
        _write_json(output_dir / "history.json", history)
    if write_plot:
        _draw_loss_curve(output_dir / "loss_curve.png", history)


def _write_training_artifacts(output_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    _write_history_artifacts(output_dir, history)
    _write_json(output_dir / "summary.json", summary)


def _run_experiment(
    cfg: dict[str, Any],
    device: torch.device,
    semantic_channels: int,
    timestamp: str,
    epochs: int | None = None,
) -> None:
    normalize_speed_config(cfg, device)
    dataset = build_dataset(cfg.get("dataset", {}), train=True)
    if not isinstance(dataset, Sized):
        raise TypeError("Training dataset must provide __len__")
    train_cfg = cfg.get("training", {})
    loader = DataLoader(dataset, **_dataloader_kwargs(train_cfg))
    model_cfg = cfg.get("model", {})
    base_channels = int(model_cfg.get("base_channels", 32))
    model = DeepSCImageModel(
        semantic_channels=semantic_channels,
        base_channels=base_channels,
    ).to(device)
    criterion = MixedMseSsimLoss(ssim_weight=float(train_cfg.get("ssim_weight", 0.2)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp_cfg = train_cfg.get("amp", DEFAULT_AMP_CONFIG)
    scaler = _make_grad_scaler(device, bool(amp_cfg.get("effective", False)))
    transfer_non_blocking = bool(train_cfg.get("dataloader", {}).get("pin_memory", False)) and device.type == "cuda"
    channel_cfg = cfg.get("channel", {})
    channel_type = str(channel_cfg.get("type", "awgn"))
    snr_values = [float(v) for v in channel_cfg.get("train_snr_db", [10])]
    epoch_count = int(epochs or train_cfg.get("epochs", 1))
    image_size = int(cfg.get("dataset", {}).get("image_size", 32))
    seed = int(cfg.get("seed", 42))
    output_dir = _experiment_output_dir(
        Path(train_cfg.get("output_dir", "outputs/train")),
        channel_type=channel_type,
        snr_values=snr_values,
        semantic_channels=semantic_channels,
        base_channels=base_channels,
        image_size=image_size,
        seed=seed,
        timestamp=timestamp,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bandwidth = estimate_semantic_bandwidth(semantic_channels, input_shape=_image_shape(cfg.get("dataset", {})))
    experiment_cfg = copy.deepcopy(cfg)
    experiment_cfg.setdefault("model", {})["semantic_channels"] = semantic_channels
    experiment_cfg.setdefault("model", {})["base_channels"] = base_channels
    experiment_cfg.setdefault("training", {})["output_dir"] = str(output_dir)
    experiment_cfg["training"]["base_output_dir"] = str(train_cfg.get("output_dir", "outputs/train"))
    experiment_cfg["training"]["timestamp"] = timestamp
    experiment_cfg["training"]["amp"] = amp_cfg
    experiment_cfg["training"]["dataloader"] = train_cfg.get("dataloader", DEFAULT_DATALOADER_CONFIG)
    experiment_cfg["training"]["artifacts"] = train_cfg.get("artifacts", DEFAULT_ARTIFACT_CONFIG)
    experiment_cfg["bandwidth_estimate"] = bandwidth
    _write_yaml(output_dir / "config.yaml", experiment_cfg)
    print(
        f"Training on {device} with {len(dataset)} samples, channel={channel_type}, snr={snr_values}, "
        f"semantic_channels={semantic_channels}, output_dir={output_dir}"
    )
    print(
        "bandwidth_estimate="
        f"input_shape={bandwidth['input_shape']} semantic_shape={bandwidth['semantic_shape']} "
        f"bandwidth_ratio={bandwidth['bandwidth_ratio']:.4f} compression_ratio={bandwidth['compression_ratio']:.4f}"
    )
    history: list[dict[str, Any]] = []
    best_train_loss: float | None = None
    best_epoch: int | None = None
    best_checkpoint_payload: dict[str, Any] | None = None
    for epoch in range(1, epoch_count + 1):
        model.train()
        total_loss = 0.0
        last_metrics = {"psnr": 0.0, "ssim": 0.0}
        batch_count = 0
        for step, batch in enumerate(loader, start=1):
            images = unwrap_batch(batch).to(device, non_blocking=transfer_non_blocking)
            snr_db = random.choice(snr_values)
            channel = ChannelConfig(channel_type=channel_type, snr_db=snr_db)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_cfg):
                reconstruction = model(images, channel=channel)
                loss = criterion(reconstruction, images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            batch_count += 1
            if step == 1 or step % 20 == 0:
                last_metrics = tensor_metrics(images.detach().cpu(), reconstruction.detach().cpu())
        if batch_count == 0:
            raise RuntimeError("Training processed zero batches; check dataset path and batch size settings.")
        avg_loss = total_loss / batch_count
        print(f"epoch={epoch:03d} loss={avg_loss:.5f} psnr={last_metrics['psnr']:.2f} ssim={last_metrics['ssim']:.4f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "psnr": float(last_metrics["psnr"]),
                "ssim": float(last_metrics["ssim"]),
                "batches": batch_count,
            }
        )
        artifact_cfg = train_cfg.get("artifacts", DEFAULT_ARTIFACT_CONFIG)
        _write_epoch_artifacts(
            output_dir,
            history,
            write_history=_should_write_epoch(epoch, epoch_count, int(artifact_cfg.get("history_every_epochs", 1))),
            write_plot=_should_write_epoch(epoch, epoch_count, int(artifact_cfg.get("plot_every_epochs", 1))),
        )
        checkpoint_extra = {"epoch": epoch, "config": experiment_cfg, "bandwidth_estimate": bandwidth}
        should_save_last = _should_write_epoch(epoch, epoch_count, int(artifact_cfg.get("checkpoint_every_epochs", 1)))
        if should_save_last:
            save_checkpoint(
                output_dir / "last_model.pth",
                model,
                checkpoint_extra,
            )
        if best_train_loss is None or avg_loss < best_train_loss:
            best_train_loss = avg_loss
            best_epoch = epoch
            best_checkpoint_payload = copy.deepcopy(_checkpoint_payload(model, checkpoint_extra))
            if should_save_last:
                shutil.copy2(output_dir / "last_model.pth", output_dir / "best_model.pth")
        elif should_save_last and best_checkpoint_payload is not None:
            torch.save(best_checkpoint_payload, output_dir / "best_model.pth")

    if best_epoch is None or best_train_loss is None or best_checkpoint_payload is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    torch.save(best_checkpoint_payload, output_dir / "best_model.pth")
    last_row = history[-1]
    summary = {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "last_epoch": int(last_row["epoch"]),
        "last_train_loss": float(last_row["train_loss"]),
        "checkpoint_filenames": {
            "best": "best_model.pth",
            "last": "last_model.pth",
        },
        "output_dir": str(output_dir),
        "bandwidth_estimate": bandwidth,
    }
    _write_training_artifacts(output_dir, history, summary)


def run_training(cfg: dict[str, Any], epochs: int | None = None) -> None:
    if epochs is not None and epochs <= 0:
        raise ValueError("epochs must be a positive integer when provided")
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    normalize_speed_config(cfg, device)
    semantic_values = _semantic_channel_values(cfg.get("model", {}))
    timestamp = datetime.now().strftime("%m%d%H%M")
    for semantic_channels in semantic_values:
        _run_experiment(cfg, device, semantic_channels, timestamp, epochs=epochs)


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    run_training(cfg)


if __name__ == "__main__":
    main()

