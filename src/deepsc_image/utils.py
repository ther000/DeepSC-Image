"""Shared utilities for configuration, images, checkpoints and reproducibility."""

from __future__ import annotations

import random
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_CHECKPOINT_KEYS = {"model_state", "epoch", "config", "bandwidth_estimate"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pil_to_tensor(image: Image.Image, image_size: int | None = None) -> torch.Tensor:
    image = image.convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def save_checkpoint(path: str | Path, model: torch.nn.Module, extra: dict[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, target)


def _torch_load_weights_only(path: Path, device: torch.device) -> Any:
    """Load trusted project checkpoints with PyTorch's safer tensor-only mode when available."""
    load_kwargs: dict[str, Any] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def _is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()
    )


def _extract_model_state(checkpoint: Any, path: Path) -> Mapping[str, torch.Tensor]:
    if _is_state_dict(checkpoint):
        return checkpoint
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint must be a state_dict or mapping payload: {path}")
    unexpected_keys = set(checkpoint) - PROJECT_CHECKPOINT_KEYS
    if unexpected_keys:
        raise ValueError(f"Checkpoint contains unexpected top-level keys {sorted(unexpected_keys)}: {path}")
    if "model_state" not in checkpoint:
        raise ValueError(f"Project checkpoint payload is missing required 'model_state': {path}")
    state = checkpoint["model_state"]
    if not _is_state_dict(state):
        raise ValueError(f"Checkpoint 'model_state' must map parameter names to tensors: {path}")
    return state


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> None:
    """Load a checkpoint produced by this project or a raw state_dict.

    PyTorch checkpoints are not a general-purpose interchange format. Only load
    checkpoints from trusted project runs or trusted collaborators; this helper
    requests ``weights_only=True`` on PyTorch versions that support it and then
    validates that the payload is either a raw tensor state_dict or this
    project's ``{"model_state": ..., "epoch": ..., "config": ...}`` mapping.
    """
    checkpoint_path = Path(path)
    try:
        checkpoint = _torch_load_weights_only(checkpoint_path, device)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(
            f"Unable to load checkpoint safely: {checkpoint_path}. Only use trusted checkpoints saved by this project."
        ) from exc
    state = _extract_model_state(checkpoint, checkpoint_path)
    model.load_state_dict(state)


def estimate_semantic_bandwidth(
    semantic_channels: int,
    input_shape: tuple[int, int, int] = (3, 32, 32),
    downsampling_factor: int = 4,
) -> dict[str, float | int | tuple[int, int, int]]:
    """Estimate semantic bandwidth and compression ratio from tensor shapes.

    The current encoder downsamples the input image twice by stride 2, so the
    default spatial downsampling factor is 4. The ratio compares raw RGB scalar
    samples (C*H*W) with transmitted semantic channel symbols
    (semantic_channels*ceil(H/4)*ceil(W/4)).
    """
    if semantic_channels <= 0:
        raise ValueError("semantic_channels must be positive")
    if downsampling_factor <= 0:
        raise ValueError("downsampling_factor must be positive")
    channels, height, width = input_shape
    semantic_height = (height + downsampling_factor - 1) // downsampling_factor
    semantic_width = (width + downsampling_factor - 1) // downsampling_factor
    input_scalars = channels * height * width
    semantic_symbols = semantic_channels * semantic_height * semantic_width
    return {
        "input_shape": input_shape,
        "semantic_shape": (semantic_channels, semantic_height, semantic_width),
        "downsampling_factor": downsampling_factor,
        "input_scalars": input_scalars,
        "semantic_symbols": semantic_symbols,
        "bandwidth_ratio": semantic_symbols / input_scalars,
        "compression_ratio": input_scalars / semantic_symbols,
    }


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
