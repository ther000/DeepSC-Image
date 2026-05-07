"""Traditional JPEG/BPG baselines with channel-like visual degradation."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable

import torch
from PIL import Image, ImageFilter

from .channels import ChannelConfig
from .utils import pil_to_tensor, tensor_to_pil


VALID_BASELINE_CODECS = ("jpeg", "bpg")


def jpeg_roundtrip(image: Image.Image, quality: int = 35) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bpg_tool(name: str) -> Path | None:
    env_dir = os.environ.get("DEEPSC_BPG_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir) / name)
    candidates.append(_repo_root() / "bpg-0.9.8-win64" / name)
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def bpg_available() -> bool:
    return _bpg_tool("bpgenc.exe") is not None and _bpg_tool("bpgdec.exe") is not None


def _bpg_temp_root() -> Path:
    candidates = []
    env_tmp = os.environ.get("DEEPSC_BPG_TMP_DIR")
    if env_tmp:
        candidates.append(Path(env_tmp))
    candidates.extend([
        Path.cwd() / "outputs" / "bpg_tmp",
        _repo_root() / "outputs" / "bpg_tmp",
        Path(tempfile.gettempdir()) / "deepsc_bpg_tmp",
    ])
    errors = []
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError as exc:
            errors.append(f"{root}: {exc}")
    raise RuntimeError("Unable to create a temporary directory for BPG baseline: " + "; ".join(errors))


def bpg_roundtrip(image: Image.Image, qp: int = 29) -> Image.Image:
    """Run BPG encode/decode through bundled command line tools."""

    encoder = _bpg_tool("bpgenc.exe")
    decoder = _bpg_tool("bpgdec.exe")
    if encoder is None or decoder is None:
        raise RuntimeError(
            "BPG tools not found. Put bpgenc.exe/bpgdec.exe under bpg-0.9.8-win64 "
            "or set DEEPSC_BPG_DIR."
        )

    qp_value = max(0, min(51, int(qp)))
    with tempfile.TemporaryDirectory(prefix="deepsc_bpg_", dir=_bpg_temp_root()) as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.png"
        bpg_path = tmp_dir / "encoded.bpg"
        output_path = tmp_dir / "decoded.png"
        image.convert("RGB").save(input_path, format="PNG")
        subprocess.run(
            [str(encoder), "-q", str(qp_value), "-o", str(bpg_path), str(input_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [str(decoder), "-o", str(output_path), str(bpg_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        decoded = Image.open(output_path).convert("RGB")
        return decoded.copy()


def degrade_tensor_like_channel(tensor: torch.Tensor, config: ChannelConfig) -> torch.Tensor:
    """Approximate modulation/channel artifacts for visual baseline comparison."""

    channel = config.normalized_type()
    if channel == "none":
        return tensor.clamp(0, 1)
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    noise_std = (1.0 / max(snr_linear, 1e-6)) ** 0.5 * 0.18
    noisy = tensor + torch.randn_like(tensor) * noise_std
    if channel == "rayleigh":
        fade_shape = (tensor.shape[0], 1, 1, 1) if tensor.dim() == 4 else (1, 1, 1)
        real = torch.randn(fade_shape, device=tensor.device, dtype=tensor.dtype)
        imag = torch.randn(fade_shape, device=tensor.device, dtype=tensor.dtype)
        fading = torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(0.2) / (2.0 ** 0.5)
        noisy = noisy * fading
    return noisy.clamp(0, 1)


def _roundtrip_baseline_tensor(
    tensor: torch.Tensor,
    config: ChannelConfig,
    roundtrip: Callable[[Image.Image], Image.Image],
    baseline_name: str,
) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"{baseline_name} baseline expects NCHW tensor")
    outputs = []
    for sample in tensor.detach().cpu():
        pil = tensor_to_pil(sample.unsqueeze(0))
        outputs.append(pil_to_tensor(roundtrip(pil)))
    stacked = torch.cat(outputs, dim=0).to(device=tensor.device, dtype=tensor.dtype)
    degraded = degrade_tensor_like_channel(stacked, config)
    if config.normalized_type() == "rayleigh":
        pil_outputs = []
        for sample in degraded.detach().cpu():
            pil_outputs.append(pil_to_tensor(tensor_to_pil(sample.unsqueeze(0)).filter(ImageFilter.GaussianBlur(radius=0.35))))
        degraded = torch.cat(pil_outputs, dim=0).to(device=tensor.device, dtype=tensor.dtype)
    return degraded.clamp(0, 1)


def jpeg_baseline_tensor(tensor: torch.Tensor, config: ChannelConfig, quality: int = 35) -> torch.Tensor:
    """Run an in-memory JPEG encode/decode then add channel-like degradation."""

    return _roundtrip_baseline_tensor(
        tensor,
        config,
        lambda image: jpeg_roundtrip(image, quality=quality),
        "JPEG",
    )


def bpg_baseline_tensor(tensor: torch.Tensor, config: ChannelConfig, qp: int = 29) -> torch.Tensor:
    """Run a BPG encode/decode then add channel-like degradation."""

    return _roundtrip_baseline_tensor(
        tensor,
        config,
        lambda image: bpg_roundtrip(image, qp=qp),
        "BPG",
    )


def baseline_tensor(
    tensor: torch.Tensor,
    config: ChannelConfig,
    *,
    codec: str = "jpeg",
    jpeg_quality: int = 35,
    bpg_qp: int = 29,
) -> torch.Tensor:
    codec_name = codec.lower().strip()
    if codec_name == "jpeg":
        return jpeg_baseline_tensor(tensor, config, quality=jpeg_quality)
    if codec_name == "bpg":
        return bpg_baseline_tensor(tensor, config, qp=bpg_qp)
    raise ValueError(f"Unsupported baseline codec: {codec!r}. Expected one of {VALID_BASELINE_CODECS}.")
