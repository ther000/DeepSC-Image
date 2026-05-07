from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from deepsc_image import train


def test_default_speed_config_is_backward_compatible() -> None:
    cfg = train.build_config_from_args(
        train.parse_args(["--config", "configs/train_cifar10_awgn.yaml"])
    )

    train.normalize_speed_config(cfg, torch.device("cpu"))

    assert cfg["training"]["amp"]["enabled"] is False
    assert cfg["training"]["amp"]["effective"] is False
    assert cfg["training"]["dataloader"] == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }
    assert cfg["training"]["artifacts"] == {
        "history_every_epochs": 1,
        "plot_every_epochs": 1,
        "checkpoint_every_epochs": 1,
    }
    assert train._dataloader_kwargs(cfg["training"]) == {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False,
    }


def test_amp_cpu_noop_when_requested() -> None:
    cfg: dict[str, object] = {
        "device": "cpu",
        "training": {
            "amp": {"enabled": True, "dtype": "float16"},
        },
    }

    train.normalize_speed_config(cfg, torch.device("cpu"))

    training = cfg["training"]
    assert isinstance(training, dict)
    amp = training["amp"]
    assert isinstance(amp, dict)

    assert amp["requested"] is True
    assert amp["enabled"] is True
    assert amp["effective"] is False
    assert amp["device_type"] == "cpu"


def test_dataloader_omits_worker_only_kwargs_at_zero_workers() -> None:
    cfg: dict[str, object] = {
        "training": {
            "batch_size": 8,
            "dataloader": {
                "num_workers": 0,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
            },
        },
    }

    train.normalize_speed_config(cfg, torch.device("cpu"))

    training = cfg["training"]
    assert isinstance(training, dict)
    dataloader = training["dataloader"]
    assert isinstance(dataloader, dict)

    assert dataloader["persistent_workers"] is False
    assert dataloader["prefetch_factor"] is None
    assert train._dataloader_kwargs(training) == {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": True,
    }


def test_dataloader_keeps_worker_kwargs_when_workers_enabled() -> None:
    cfg: dict[str, object] = {
        "training": {
            "batch_size": 8,
            "dataloader": {
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 3,
            },
        },
    }

    train.normalize_speed_config(cfg, torch.device("cuda"))

    training = cfg["training"]
    assert isinstance(training, dict)

    assert train._dataloader_kwargs(training) == {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
    }


def test_artifact_cadence_must_be_positive() -> None:
    cfg = {"training": {"artifacts": {"plot_every_epochs": 0}}}

    with pytest.raises(ValueError, match="training.artifacts.plot_every_epochs"):
        train.normalize_speed_config(cfg, torch.device("cpu"))


def test_dataloader_rejects_negative_num_workers() -> None:
    cfg = {"training": {"dataloader": {"num_workers": -1}}}

    with pytest.raises(ValueError, match="num_workers"):
        train.normalize_speed_config(cfg, torch.device("cpu"))


def test_dataloader_rejects_non_positive_prefetch_factor() -> None:
    cfg = {"training": {"dataloader": {"num_workers": 1, "prefetch_factor": 0}}}

    with pytest.raises(ValueError, match="prefetch_factor"):
        train.normalize_speed_config(cfg, torch.device("cpu"))


def test_artifact_default_history_artifacts(tmp_path: Path) -> None:
    history = [{"epoch": 1, "train_loss": 0.25, "psnr": 12.5, "ssim": 0.4, "batches": 2}]

    train._write_history_artifacts(tmp_path, history)

    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "loss_curve.svg").is_file()


def test_amp_dtype_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="training.amp.dtype"):
        train._amp_dtype("float32")


def test_autocast_context_cpu_noop() -> None:
    with train._autocast_context(torch.device("cpu"), {"effective": False, "dtype": "float16"}):
        tensor = torch.ones(1)

    assert tensor.dtype == torch.float32


def test_dataloader_kwargs_can_construct_loader() -> None:
    dataset = TensorDataset(torch.rand(2, 3, 4, 4))
    train_cfg = {
        "batch_size": 1,
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
        },
    }

    loader = torch.utils.data.DataLoader(dataset, **train._dataloader_kwargs(train_cfg))

    assert len(list(loader)) == 2


def test_should_write_epoch_always_writes_final_epoch() -> None:
    assert train._should_write_epoch(epoch=1, total_epochs=3, every_epochs=2) is False
    assert train._should_write_epoch(epoch=2, total_epochs=3, every_epochs=2) is True
    assert train._should_write_epoch(epoch=3, total_epochs=3, every_epochs=10) is True


def test_history_artifacts_can_skip_plot(tmp_path: Path) -> None:
    history = [{"epoch": 1, "train_loss": 0.25, "psnr": 12.5, "ssim": 0.4, "batches": 2}]

    train._write_history_artifacts(tmp_path, history, include_plot=False)

    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "history.json").is_file()
    assert not (tmp_path / "loss_curve.png").exists()


def test_epoch_artifacts_can_write_plot_without_history(tmp_path: Path) -> None:
    history = [{"epoch": 1, "train_loss": 0.25, "psnr": 12.5, "ssim": 0.4, "batches": 2}]

    train._write_epoch_artifacts(tmp_path, history, write_history=False, write_plot=True)

    assert not (tmp_path / "history.csv").exists()
    assert not (tmp_path / "history.json").exists()
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "loss_curve.svg").is_file()


def test_checkpoint_payload_copies_model_state() -> None:
    model = torch.nn.Linear(1, 1)
    payload = train._checkpoint_payload(model, {"epoch": 1})

    assert payload["epoch"] == 1
    assert "model_state" in payload
