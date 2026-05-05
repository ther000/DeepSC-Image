from __future__ import annotations

from pathlib import Path

from deepsc_image import evaluate


def _synthetic_rows() -> list[dict[str, float | str | int]]:
    return [
        {
            "snr_db": -5.0,
            "channel": "awgn",
            "deepsc_psnr": 18.2,
            "deepsc_ssim": 0.52,
            "jpeg_psnr": 16.1,
            "jpeg_ssim": 0.44,
            "samples": 2,
            "monte_carlo_samples": 1,
            "repetitions": 2,
            "bandwidth_estimate": 0,
        },
        {
            "snr_db": 0.0,
            "channel": "awgn",
            "deepsc_psnr": 20.3,
            "deepsc_ssim": 0.61,
            "jpeg_psnr": 18.5,
            "jpeg_ssim": 0.54,
            "samples": 2,
            "monte_carlo_samples": 1,
            "repetitions": 2,
            "bandwidth_estimate": 0,
        },
        {
            "snr_db": 10.0,
            "channel": "awgn",
            "deepsc_psnr": 24.8,
            "deepsc_ssim": 0.82,
            "jpeg_psnr": 22.0,
            "jpeg_ssim": 0.77,
            "samples": 2,
            "monte_carlo_samples": 1,
            "repetitions": 2,
            "bandwidth_estimate": 0,
        },
    ]


def test_eval_curve_artifacts_created_and_non_empty(tmp_path: Path) -> None:
    evaluate._write_curve_artifacts(tmp_path, _synthetic_rows())

    psnr_curve = tmp_path / "psnr_vs_snr.png"
    ssim_curve = tmp_path / "ssim_vs_snr.png"

    assert psnr_curve.is_file()
    assert ssim_curve.is_file()
    assert psnr_curve.stat().st_size > 0
    assert ssim_curve.stat().st_size > 0


def test_eval_plot_curves_enabled_defaults_to_true() -> None:
    assert evaluate._plot_curves_enabled({}) is True
    assert evaluate._plot_curves_enabled({"artifacts": {}}) is True


def test_eval_plot_curves_enabled_can_disable() -> None:
    assert evaluate._plot_curves_enabled({"artifacts": {"plot_curves": False}}) is False
    assert evaluate._plot_curves_enabled({"artifacts": {"plot_curves": True}}) is True


def test_eval_output_dir_without_checkpoint_keeps_base_dir() -> None:
    output_dir = evaluate._evaluation_output_dir(Path("outputs/eval_kodak"), None)

    assert output_dir == Path("outputs/eval_kodak")


def test_eval_output_dir_with_empty_checkpoint_keeps_base_dir() -> None:
    output_dir = evaluate._evaluation_output_dir(Path("outputs/eval_kodak"), "")

    assert output_dir == Path("outputs/eval_kodak")


def test_eval_output_dir_matches_training_run_and_checkpoint() -> None:
    checkpoint = (
        Path("outputs/train_cifar10_awgn")
        / "ts_05041200__ch_awgn__snr_0_5_10_15_20__sem_16__base_32__img_32__seed_42"
        / "best_model.pth"
    )

    output_dir = evaluate._evaluation_output_dir(Path("outputs/eval_kodak"), checkpoint)

    assert output_dir == (
        Path("outputs/eval_kodak")
        / "ts_05041200__ch_awgn__snr_0_5_10_15_20__sem_16__base_32__img_32__seed_42__ckpt_best_model"
    )


def test_eval_output_dir_distinguishes_checkpoint_stems() -> None:
    run_dir = Path("outputs/train_cifar10_awgn/ts_05041200__ch_awgn__sem_16")

    best_dir = evaluate._evaluation_output_dir(Path("outputs/eval_kodak"), run_dir / "best_model.pth")
    last_dir = evaluate._evaluation_output_dir(Path("outputs/eval_kodak"), run_dir / "last_model.pth")

    assert best_dir.name.endswith("__ckpt_best_model")
    assert last_dir.name.endswith("__ckpt_last_model")
    assert best_dir != last_dir


def test_eval_output_dir_sanitizes_external_checkpoint_names() -> None:
    output_dir = evaluate._evaluation_output_dir(
        Path("outputs/eval_kodak"),
        Path("external models/Model Run #1/best model!.pth"),
    )

    assert output_dir == Path("outputs/eval_kodak/model-run-1__ckpt_best-model")
