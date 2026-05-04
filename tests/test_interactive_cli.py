from __future__ import annotations

from deepsc_image import interactive_cli


def feed(values: list[str]):
    iterator = iter(values)
    return lambda _: next(iterator)


def test_float_list_accepts_commas_and_spaces() -> None:
    assert interactive_cli.prompt_float_list("SNR", [10.0], feed(["0, 10 20"])) == [0.0, 10.0, 20.0]


def test_int_list_retries_invalid_input() -> None:
    messages: list[str] = []
    assert interactive_cli.prompt_int_list("Semantic channels", [32], feed(["bad", "16,32 64"]), messages.append) == [16, 32, 64]
    assert messages


def test_nested_set_and_get() -> None:
    cfg: dict[str, object] = {}
    interactive_cli.set_nested(cfg, ["training", "epochs"], 10)
    assert interactive_cli.get_nested(cfg, ["training", "epochs"]) == 10
    assert interactive_cli.get_nested(cfg, ["missing"], "fallback") == "fallback"


def test_train_interactive_blank_keeps_defaults() -> None:
    cfg = {
        "seed": 42,
        "device": "auto",
        "dataset": {"name": "cifar10", "root": "datasets/cifar10", "image_size": 32, "download": False},
        "model": {"semantic_channels": [16, 32, 64], "base_channels": 32},
        "channel": {"type": "awgn", "train_snr_db": [0, 5, 10]},
        "training": {"epochs": 100, "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "ssim_weight": 0.2, "output_dir": "outputs/train"},
    }
    interactive_cli.apply_train_interactive_config(cfg, feed([""] * 16), lambda _: None)
    assert cfg["model"]["semantic_channels"] == [16, 32, 64]
    assert cfg["training"]["epochs"] == 100


def test_eval_interactive_updates_selected_values() -> None:
    cfg = {
        "device": "auto",
        "dataset": {"name": "image_folder", "root": "datasets/kodak", "image_size": 256},
        "model": {"semantic_channels": 32, "base_channels": 32},
        "checkpoint": None,
        "channel": {"type": "awgn", "snr_db": [0, 10]},
        "baseline": {"jpeg_quality": 35},
        "evaluation": {"monte_carlo_samples": 1},
        "output_dir": "outputs/eval",
    }
    values = ["cuda", "", "", "", "64", "", "ckpt.pth", "rayleigh", "-5,0,5", "", "3", "outputs/new_eval"]
    interactive_cli.apply_eval_interactive_config(cfg, feed(values), lambda _: None)
    assert cfg["device"] == "cuda"
    assert cfg["model"]["semantic_channels"] == 64
    assert cfg["checkpoint"] == "ckpt.pth"
    assert cfg["channel"] == {"type": "rayleigh", "snr_db": [-5.0, 0.0, 5.0]}
    assert cfg["evaluation"]["monte_carlo_samples"] == 3
    assert cfg["output_dir"] == "outputs/new_eval"
