"""Bounded training benchmark CLI."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from .channels import ChannelConfig
from .losses import MixedMseSsimLoss
from .model import DeepSCImageModel
from .train import (
    _autocast_context,
    _dataloader_kwargs,
    _make_grad_scaler,
    _semantic_channel_values,
    normalize_speed_config,
)
from .utils import ensure_parent, load_yaml, resolve_device, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark bounded DeepSC training throughput")
    parser.add_argument("--config", default="configs/train_cifar10_awgn.yaml", help="YAML training config path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amp", action="store_true", help="Request AMP for this benchmark run")
    parser.add_argument("--semantic-channels", type=int, default=None, help="Single semantic channel count to benchmark")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    return parser.parse_args(argv)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_positive(value: float) -> bool:
    return value > 0 and bool(torch.isfinite(torch.tensor(value)).item())


def _build_benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    training = cfg.setdefault("training", {})
    if args.amp:
        training.setdefault("amp", {})["enabled"] = True
    if args.semantic_channels is not None:
        cfg.setdefault("model", {})["semantic_channels"] = int(args.semantic_channels)
    dataloader = training.setdefault("dataloader", {})
    if args.num_workers is not None:
        dataloader["num_workers"] = args.num_workers
    if args.pin_memory:
        dataloader["pin_memory"] = True
    return cfg


def _benchmark_semantic_channels(model_cfg: dict[str, Any]) -> int:
    values = _semantic_channel_values(model_cfg)
    if len(values) != 1:
        raise ValueError("benchmark requires a single model.semantic_channels value or --semantic-channels")
    return values[0]


def _synthetic_loader(cfg: dict[str, Any]) -> tuple[DataLoader[tuple[torch.Tensor, ...]], int]:
    dataset_cfg = cfg.get("dataset", {})
    image_size = int(dataset_cfg.get("image_size", 32))
    training = cfg.get("training", {})
    batch_size = int(training.get("batch_size", 64))
    max_batches = int(training.get("benchmark_max_batches", 5))
    sample_count = max(batch_size * max_batches, batch_size)
    images = torch.rand(sample_count, 3, image_size, image_size)
    dataset = TensorDataset(images)
    return DataLoader(dataset, **_dataloader_kwargs(training)), sample_count


def _run_once(cfg: dict[str, Any], device: torch.device, *, epochs: int, max_batches: int) -> dict[str, float | int]:
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = cfg.get("training", {})
    train_cfg["benchmark_max_batches"] = max_batches
    loader, _ = _synthetic_loader(cfg)
    model_cfg = cfg.get("model", {})
    model = DeepSCImageModel(
        semantic_channels=_benchmark_semantic_channels(model_cfg),
        base_channels=int(model_cfg.get("base_channels", 32)),
    ).to(device)
    criterion = MixedMseSsimLoss(ssim_weight=float(train_cfg.get("ssim_weight", 0.2)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    amp_cfg = train_cfg.get("amp", {})
    scaler = _make_grad_scaler(device, bool(amp_cfg.get("effective", False)))
    channel_cfg = cfg.get("channel", {})
    channel = ChannelConfig(
        channel_type=str(channel_cfg.get("type", "awgn")),
        snr_db=float((channel_cfg.get("train_snr_db", [10]) or [10])[0]),
    )
    non_blocking = bool(train_cfg.get("dataloader", {}).get("pin_memory", False)) and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync_if_cuda(device)
    start = time.perf_counter()
    processed = 0
    batches = 0
    last_loss = 0.0
    for _epoch in range(epochs):
        for step, batch in enumerate(loader, start=1):
            if step > max_batches:
                break
            images = batch[0].to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_cfg):
                reconstruction = model(images, channel=channel)
                loss = criterion(reconstruction, images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            processed += int(images.shape[0])
            batches += 1
            last_loss = float(loss.item())
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    if not _finite_positive(elapsed) or processed <= 0:
        raise RuntimeError("Benchmark did not produce a finite positive duration with processed samples")
    return {
        "epoch_seconds": elapsed / epochs,
        "samples_per_second": processed / elapsed,
        "processed_samples": processed,
        "batches": batches,
        "last_loss": last_loss,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.warmup < 0 or args.repeat <= 0 or args.max_batches <= 0:
        raise ValueError("epochs, repeat, and max-batches must be positive; warmup must be non-negative")
    cfg = _build_benchmark_config(args)
    device = resolve_device(cfg.get("device", "auto"))
    normalize_speed_config(cfg, device)
    for _ in range(args.warmup):
        _run_once(copy.deepcopy(cfg), device, epochs=args.epochs, max_batches=args.max_batches)
    runs = [
        _run_once(copy.deepcopy(cfg), device, epochs=args.epochs, max_batches=args.max_batches)
        for _ in range(args.repeat)
    ]
    epoch_seconds = [float(run["epoch_seconds"]) for run in runs]
    samples_per_second = [float(run["samples_per_second"]) for run in runs]
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    semantic_channels = _benchmark_semantic_channels(model_cfg)
    base_channels = int(model_cfg.get("base_channels", 32))
    result: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "config_path": str(args.config),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "amp_requested": bool(train_cfg.get("amp", {}).get("requested", False)),
        "amp_enabled": bool(train_cfg.get("amp", {}).get("effective", False)),
        "amp": train_cfg.get("amp", {}),
        "dataloader": train_cfg.get("dataloader", {}),
        "model": {"semantic_channels": semantic_channels, "base_channels": base_channels},
        "warmup": args.warmup,
        "repeat": args.repeat,
        "epochs": args.epochs,
        "max_batches": args.max_batches,
        "runs": runs,
        "timing": {
            "epoch_seconds_median": statistics.median(epoch_seconds),
            "epoch_seconds_mean": statistics.mean(epoch_seconds),
            "samples_per_second_median": statistics.median(samples_per_second),
            "samples_per_second_mean": statistics.mean(samples_per_second),
        },
        "cuda_memory": cuda_memory,
    }
    output_path = ensure_parent(args.output)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
