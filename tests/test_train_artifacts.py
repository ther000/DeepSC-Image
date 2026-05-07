from __future__ import annotations

import json
from pathlib import Path

import yaml

from deepsc_image import train


def test_experiment_output_dir_includes_safe_run_fields() -> None:
    output_dir = train._experiment_output_dir(
        Path("outputs/train"),
        channel_type="awgn",
        snr_values=[-5.0, 0.5, 10.0],
        semantic_channels=4,
        base_channels=8,
        image_size=32,
        seed=123,
        timestamp="05041200",
    )

    name = output_dir.name

    assert output_dir.parent == Path("outputs/train")
    assert "ts_05041200" in name
    assert "ch_awgn" in name
    assert "snr_m5_0p5_10" in name
    assert "sem_4" in name
    assert "base_8" in name
    assert "img_32" in name
    assert "seed_123" in name


def test_history_artifacts_can_refresh_without_summary(tmp_path: Path) -> None:
    history_one = [
        {"epoch": 1, "train_loss": 0.25, "psnr": 12.5, "ssim": 0.4, "batches": 2},
    ]

    train._write_history_artifacts(tmp_path, history_one)

    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "history.json").is_file()
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "loss_curve.svg").is_file()
    assert not (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == history_one

    history_two = [
        *history_one,
        {"epoch": 2, "train_loss": 0.2, "psnr": 13.0, "ssim": 0.45, "batches": 2},
    ]

    train._write_history_artifacts(tmp_path, history_two)

    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == history_two
    assert len((tmp_path / "history.csv").read_text(encoding="utf-8").splitlines()) == 3
    assert (tmp_path / "loss_curve.png").stat().st_size > 0
    assert (tmp_path / "loss_curve.svg").stat().st_size > 0
    assert not (tmp_path / "summary.json").exists()


def test_training_artifact_writers(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 0.25, "psnr": 12.5, "ssim": 0.4, "batches": 2},
        {"epoch": 2, "train_loss": 0.2, "psnr": 13.0, "ssim": 0.45, "batches": 2},
    ]
    summary = {
        "best_epoch": 2,
        "best_train_loss": 0.2,
        "last_epoch": 2,
        "last_train_loss": 0.2,
        "checkpoint_filenames": {"best": "best_model.pth", "last": "last_model.pth"},
        "output_dir": str(tmp_path),
        "bandwidth_estimate": {"bandwidth_ratio": 0.1},
    }

    train._write_yaml(tmp_path / "config.yaml", {"model": {"semantic_channels": 4}})
    train._write_training_artifacts(tmp_path, history, summary)

    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == {
        "model": {"semantic_channels": 4}
    }
    assert (tmp_path / "history.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "epoch,train_loss,psnr,ssim,batches"
    )
    assert json.loads((tmp_path / "history.json").read_text(encoding="utf-8")) == history
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["best_train_loss"] == 0.2
    assert (tmp_path / "loss_curve.png").is_file()
    assert (tmp_path / "loss_curve.svg").is_file()


def test_loss_curve_title_includes_training_parameters() -> None:
    title = train._format_loss_curve_title(
        {
            "semantic_channels": 32,
            "channel_type": "awgn",
            "snr_values": [-5.0, 0.0, 5.0],
            "image_size": 256,
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
        }
    )

    assert "semantic_channels=32" in title
    assert "channel=AWGN" in title
    assert "train_snr=-5/0/5 dB" in title
    assert "lr=0.001" in title
    assert "image=256x256" not in title
    assert "epochs=100" not in title
    assert "batch_size=64" not in title

