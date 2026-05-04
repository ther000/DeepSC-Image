from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest

import importlib

benchmark_training = importlib.import_module("deepsc_image.benchmark_training")


def _args(output: Path, **overrides: str | int | bool | None) -> argparse.Namespace:
    values: dict[str, str | int | bool | None] = {
        "config": "configs/train_cifar10_awgn.yaml",
        "epochs": 1,
        "warmup": 0,
        "repeat": 1,
        "max_batches": 1,
        "output": str(output),
            "amp": False,
        "semantic_channels": 16,
            "num_workers": None,
            "pin_memory": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_benchmark_writes_json_schema(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"

    result = benchmark_training.run_benchmark(_args(output))
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted == result
    assert persisted["torch_version"]
    assert persisted["config_path"] == "configs/train_cifar10_awgn.yaml"
    assert persisted["repeat"] == 1
    assert persisted["max_batches"] == 1
    assert persisted["amp_requested"] is False
    assert persisted["amp_enabled"] is False
    assert persisted["model"] == {"semantic_channels": 16, "base_channels": 32}
    assert persisted["dataloader"]["num_workers"] == 0
    assert len(persisted["runs"]) == 1
    assert math.isfinite(persisted["timing"]["epoch_seconds_median"])
    assert persisted["timing"]["epoch_seconds_median"] > 0
    assert math.isfinite(persisted["timing"]["samples_per_second_median"])
    assert persisted["timing"]["samples_per_second_median"] > 0


def test_benchmark_rejects_invalid_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="epochs, repeat, and max-batches"):
        benchmark_training.run_benchmark(_args(tmp_path / "benchmark.json", repeat=0))


def test_benchmark_requires_single_semantic_channel_when_not_overridden(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single model.semantic_channels"):
        benchmark_training.run_benchmark(_args(tmp_path / "benchmark.json", semantic_channels=None))


def test_benchmark_preserves_yaml_amp_when_not_overridden(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        """
seed: 1
device: cpu
dataset:
  image_size: 32
model:
  semantic_channels: 2
  base_channels: 2
channel:
  type: none
  train_snr_db: [10]
training:
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0
  ssim_weight: 0.2
  amp:
    enabled: true
    dtype: float16
""".strip(),
        encoding="utf-8",
    )

    result = benchmark_training.run_benchmark(_args(output, config=str(config)))

    assert result["amp_requested"] is True
    assert result["amp_enabled"] is False


def test_benchmark_cli_parse_amp_and_loader_flags(tmp_path: Path) -> None:
    args = benchmark_training.parse_args(
        [
            "--config",
            "configs/train_cifar10_awgn.yaml",
            "--epochs",
            "1",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--max-batches",
            "1",
            "--amp",
            "--semantic-channels",
            "16",
            "--num-workers",
            "2",
            "--pin-memory",
            "--output",
            str(tmp_path / "benchmark.json"),
        ]
    )

    assert args.amp is True
    assert args.semantic_channels == 16
    assert args.num_workers == 2
    assert args.pin_memory is True


def test_docs_placeholder_marker() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python -m deepsc_image.train" in readme
    assert "training.amp.enabled" in readme
    assert "training.dataloader.num_workers" in readme
    assert "training.artifacts.*_every_epochs" in readme
    assert "python -m deepsc_image.benchmark_training" in readme
