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
