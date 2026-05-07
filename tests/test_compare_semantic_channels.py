from __future__ import annotations

import json
from pathlib import Path

from deepsc_image import compare_semantic_channels


def _write_metrics(path: Path, semantic_channels: int | None, psnr_offset: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for snr in [-5.0, 0.0, 5.0]:
        row = {
            "snr_db": snr,
            "channel": "awgn",
            "deepsc_psnr": 20.0 + snr + psnr_offset,
            "deepsc_ssim": 0.5 + (snr + 5.0) * 0.01,
        }
        if semantic_channels is not None:
            row["semantic_channels"] = semantic_channels
        else:
            row["bandwidth_estimate"] = {"semantic_shape": [16, 64, 64]}
        rows.append(row)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_comparison_rows_uses_metric_field(tmp_path: Path) -> None:
    metrics = _write_metrics(tmp_path / "sem_32_run" / "metrics.json", 32)

    rows = compare_semantic_channels.load_comparison_rows([str(metrics)])

    assert {row["semantic_channels"] for row in rows} == {32}
    assert [row["snr_db"] for row in rows] == [-5.0, 0.0, 5.0]


def test_load_comparison_rows_falls_back_to_semantic_shape(tmp_path: Path) -> None:
    metrics = _write_metrics(tmp_path / "legacy__sem_99" / "metrics.json", None)

    rows = compare_semantic_channels.load_comparison_rows([str(metrics)])

    assert {row["semantic_channels"] for row in rows} == {16}


def test_write_comparison_artifacts(tmp_path: Path) -> None:
    first = _write_metrics(tmp_path / "run__sem_16" / "metrics.json", 16)
    second = _write_metrics(tmp_path / "run__sem_32" / "metrics.json", 32, psnr_offset=2.0)
    rows = compare_semantic_channels.load_comparison_rows([str(first), str(second)])
    output_dir = tmp_path / "compare"

    compare_semantic_channels.write_comparison_artifacts(output_dir, rows)

    assert (output_dir / "semantic_channels_comparison.csv").is_file()
    assert (output_dir / "semantic_channels_psnr_vs_snr.svg").is_file()
    assert (output_dir / "semantic_channels_psnr_vs_snr.png").is_file()
    assert (output_dir / "semantic_channels_ssim_vs_snr.svg").is_file()
    assert (output_dir / "semantic_channels_ssim_vs_snr.png").is_file()
    psnr_svg = (output_dir / "semantic_channels_psnr_vs_snr.svg").read_text(encoding="utf-8")
    ssim_svg = (output_dir / "semantic_channels_ssim_vs_snr.svg").read_text(encoding="utf-8")
    assert "semantic_channels=16" in psnr_svg
    assert "semantic_channels=32" in psnr_svg
    assert "SNR (dB)" in psnr_svg
    assert "PSNR (dB)" in psnr_svg
    assert "SNR (dB)" in ssim_svg
    assert "SSIM" in ssim_svg
    assert "snr_db" not in psnr_svg
    assert "deepsc_psnr" not in psnr_svg
    assert "deepsc_ssim" not in ssim_svg
