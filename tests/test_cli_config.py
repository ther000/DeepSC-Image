from __future__ import annotations

from deepsc_image import evaluate, train


def test_train_cli_overrides_nested_config() -> None:
    args = train.parse_args([
        "--config", "configs/train_cifar10_awgn.yaml",
        "--epochs", "2",
        "--semantic-channels", "8", "16",
        "--channel", "rayleigh",
        "--snr-db", "0", "10",
        "--output-dir", "outputs/custom_train",
    ])
    cfg = train.build_config_from_args(args)
    assert cfg["training"]["epochs"] == 2
    assert cfg["model"]["semantic_channels"] == [8, 16]
    assert cfg["channel"] == {"type": "rayleigh", "train_snr_db": [0.0, 10.0]}
    assert cfg["training"]["output_dir"] == "outputs/custom_train"


def test_evaluate_cli_overrides_nested_config() -> None:
    args = evaluate.parse_args([
        "--config", "configs/eval_kodak.yaml",
        "--checkpoint", "best.pth",
        "--semantic-channels", "64",
        "--channel", "rayleigh",
        "--snr-db", "-5", "0", "5",
        "--baseline-codec", "bpg",
        "--bpg-qp", "31",
        "--monte-carlo-samples", "2",
    ])
    cfg = evaluate.build_config_from_args(args)
    assert cfg["checkpoint"] == "best.pth"
    assert cfg["model"]["semantic_channels"] == 64
    assert cfg["channel"] == {"type": "rayleigh", "snr_db": [-5.0, 0.0, 5.0]}
    assert cfg["baseline"]["codec"] == "bpg"
    assert cfg["baseline"]["bpg_qp"] == 31
    assert cfg["evaluation"]["monte_carlo_samples"] == 2
