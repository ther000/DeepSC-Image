"""Shared interactive command-line prompt helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

InputFunc = Callable[[str], str]
OutputFunc = Callable[[str], None]


def get_nested(cfg: dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = cfg
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(cfg: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = cfg
    for key in path[:-1]:
        value_at_key = current.setdefault(key, {})
        if not isinstance(value_at_key, dict):
            value_at_key = {}
            current[key] = value_at_key
        current = value_at_key
    current[path[-1]] = value


def _default_text(default: Any) -> str:
    if isinstance(default, list):
        return ", ".join(str(item) for item in default)
    return "" if default is None else str(default)


def _raw(label: str, default: Any, input_func: InputFunc) -> str:
    suffix = f" [{_default_text(default)}]" if default is not None else ""
    return input_func(f"{label}{suffix}: ").strip()


def prompt_string(label: str, default: str | None, input_func: InputFunc = input) -> str | None:
    value = _raw(label, default, input_func)
    return default if value == "" else value


def prompt_choice(label: str, choices: Sequence[str], default: str, input_func: InputFunc = input, output_func: OutputFunc = print) -> str:
    allowed = set(choices)
    while True:
        value = _raw(f"{label} ({'/'.join(choices)})", default, input_func)
        if value == "":
            return default
        if value in allowed:
            return value
        output_func(f"Invalid value. Choose one of: {', '.join(choices)}")


def prompt_bool(label: str, default: bool, input_func: InputFunc = input, output_func: OutputFunc = print) -> bool:
    while True:
        value = _raw(label, "y" if default else "n", input_func).lower()
        if value == "":
            return default
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        output_func("Enter yes or no.")


def prompt_int(label: str, default: int, input_func: InputFunc = input, output_func: OutputFunc = print, *, positive: bool = True) -> int:
    while True:
        value = _raw(label, default, input_func)
        if value == "":
            return int(default)
        try:
            parsed = int(value)
        except ValueError:
            output_func("Enter an integer.")
            continue
        if positive and parsed <= 0:
            output_func("Enter a positive integer.")
            continue
        return parsed


def prompt_non_negative_int(label: str, default: int, input_func: InputFunc = input, output_func: OutputFunc = print) -> int:
    while True:
        value = prompt_int(label, default, input_func, output_func, positive=False)
        if value >= 0:
            return value
        output_func("Enter a non-negative integer.")


def prompt_optional_positive_int(label: str, default: int | None, input_func: InputFunc = input, output_func: OutputFunc = print) -> int | None:
    while True:
        value = _raw(label, default, input_func).lower()
        if value == "":
            return default
        if value in {"none", "null"}:
            return None
        try:
            parsed = int(value)
        except ValueError:
            output_func("Enter a positive integer, none, or null.")
            continue
        if parsed <= 0:
            output_func("Enter a positive integer, none, or null.")
            continue
        return parsed


def prompt_float(label: str, default: float, input_func: InputFunc = input, output_func: OutputFunc = print, *, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        value = _raw(label, default, input_func)
        if value == "":
            return float(default)
        try:
            parsed = float(value)
        except ValueError:
            output_func("Enter a number.")
            continue
        if minimum is not None and parsed < minimum:
            output_func(f"Enter a value >= {minimum}.")
            continue
        if maximum is not None and parsed > maximum:
            output_func(f"Enter a value <= {maximum}.")
            continue
        return parsed


def _split_values(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def prompt_float_list(label: str, default: Sequence[float], input_func: InputFunc = input, output_func: OutputFunc = print) -> list[float]:
    while True:
        value = _raw(label, list(default), input_func)
        if value == "":
            return [float(item) for item in default]
        try:
            parsed = [float(item) for item in _split_values(value)]
        except ValueError:
            output_func("Enter numbers separated by commas or spaces.")
            continue
        if not parsed:
            output_func("Enter at least one number.")
            continue
        return parsed


def prompt_int_list(label: str, default: Sequence[int], input_func: InputFunc = input, output_func: OutputFunc = print, *, positive: bool = True) -> list[int]:
    while True:
        value = _raw(label, list(default), input_func)
        if value == "":
            return [int(item) for item in default]
        try:
            parsed = [int(item) for item in _split_values(value)]
        except ValueError:
            output_func("Enter integers separated by commas or spaces.")
            continue
        if not parsed:
            output_func("Enter at least one integer.")
            continue
        if positive and any(item <= 0 for item in parsed):
            output_func("Enter positive integers.")
            continue
        return parsed


def _as_int_list(value: Any, default: list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None:
        return default
    return [int(value)]


def apply_train_interactive_config(cfg: dict[str, Any], input_func: InputFunc = input, output_func: OutputFunc = print) -> dict[str, Any]:
    output_func("Interactive training setup. Press Enter to keep the value in brackets.")
    set_nested(cfg, ["seed"], prompt_int("Seed", int(get_nested(cfg, ["seed"], 42)), input_func, output_func))
    set_nested(cfg, ["device"], prompt_string("Device", str(get_nested(cfg, ["device"], "auto")), input_func))
    set_nested(cfg, ["dataset", "name"], prompt_choice("Dataset", ("cifar10", "image_folder"), str(get_nested(cfg, ["dataset", "name"], "cifar10")), input_func, output_func))
    set_nested(cfg, ["dataset", "root"], prompt_string("Data root", str(get_nested(cfg, ["dataset", "root"], "datasets/cifar10")), input_func))
    set_nested(cfg, ["dataset", "image_size"], prompt_int("Image size", int(get_nested(cfg, ["dataset", "image_size"], 32)), input_func, output_func))
    set_nested(cfg, ["dataset", "download"], prompt_bool("Download CIFAR-10 if missing", bool(get_nested(cfg, ["dataset", "download"], False)), input_func, output_func))
    set_nested(cfg, ["model", "semantic_channels"], prompt_int_list("Semantic channels", _as_int_list(get_nested(cfg, ["model", "semantic_channels"], [32]), [32]), input_func, output_func))
    set_nested(cfg, ["model", "base_channels"], prompt_int("Base channels", int(get_nested(cfg, ["model", "base_channels"], 32)), input_func, output_func))
    set_nested(cfg, ["channel", "type"], prompt_choice("Channel", ("none", "awgn", "rayleigh"), str(get_nested(cfg, ["channel", "type"], "awgn")), input_func, output_func))
    set_nested(cfg, ["channel", "train_snr_db"], prompt_float_list("Training SNR dB values", get_nested(cfg, ["channel", "train_snr_db"], [10.0]), input_func, output_func))
    set_nested(cfg, ["training", "epochs"], prompt_int("Epochs", int(get_nested(cfg, ["training", "epochs"], 1)), input_func, output_func))
    set_nested(cfg, ["training", "batch_size"], prompt_int("Batch size", int(get_nested(cfg, ["training", "batch_size"], 64)), input_func, output_func))
    set_nested(cfg, ["training", "learning_rate"], prompt_float("Learning rate", float(get_nested(cfg, ["training", "learning_rate"], 1e-3)), input_func, output_func, minimum=0.0))
    set_nested(cfg, ["training", "weight_decay"], prompt_float("Weight decay", float(get_nested(cfg, ["training", "weight_decay"], 1e-4)), input_func, output_func, minimum=0.0))
    set_nested(cfg, ["training", "ssim_weight"], prompt_float("SSIM loss weight", float(get_nested(cfg, ["training", "ssim_weight"], 0.2)), input_func, output_func, minimum=0.0, maximum=1.0))
    set_nested(cfg, ["training", "amp", "enabled"], prompt_bool("Enable CUDA AMP", bool(get_nested(cfg, ["training", "amp", "enabled"], False)), input_func, output_func))
    set_nested(cfg, ["training", "amp", "dtype"], prompt_choice("AMP dtype", ("float16", "bfloat16"), str(get_nested(cfg, ["training", "amp", "dtype"], "float16")), input_func, output_func))
    set_nested(cfg, ["training", "dataloader", "num_workers"], prompt_non_negative_int("DataLoader workers", int(get_nested(cfg, ["training", "dataloader", "num_workers"], 0)), input_func, output_func))
    set_nested(cfg, ["training", "dataloader", "pin_memory"], prompt_bool("DataLoader pin memory", bool(get_nested(cfg, ["training", "dataloader", "pin_memory"], False)), input_func, output_func))
    set_nested(cfg, ["training", "dataloader", "persistent_workers"], prompt_bool("DataLoader persistent workers", bool(get_nested(cfg, ["training", "dataloader", "persistent_workers"], False)), input_func, output_func))
    set_nested(cfg, ["training", "dataloader", "prefetch_factor"], prompt_optional_positive_int("DataLoader prefetch factor", get_nested(cfg, ["training", "dataloader", "prefetch_factor"], None), input_func, output_func))
    set_nested(cfg, ["training", "artifacts", "history_every_epochs"], prompt_int("History refresh every epochs", int(get_nested(cfg, ["training", "artifacts", "history_every_epochs"], 1)), input_func, output_func))
    set_nested(cfg, ["training", "artifacts", "plot_every_epochs"], prompt_int("Loss curve refresh every epochs", int(get_nested(cfg, ["training", "artifacts", "plot_every_epochs"], 1)), input_func, output_func))
    set_nested(cfg, ["training", "artifacts", "checkpoint_every_epochs"], prompt_int("Checkpoint refresh every epochs", int(get_nested(cfg, ["training", "artifacts", "checkpoint_every_epochs"], 1)), input_func, output_func))
    set_nested(cfg, ["training", "output_dir"], prompt_string("Output directory", str(get_nested(cfg, ["training", "output_dir"], "outputs/train")), input_func))
    return cfg


def apply_eval_interactive_config(cfg: dict[str, Any], input_func: InputFunc = input, output_func: OutputFunc = print) -> dict[str, Any]:
    output_func("Interactive evaluation setup. Press Enter to keep the value in brackets.")
    set_nested(cfg, ["device"], prompt_string("Device", str(get_nested(cfg, ["device"], "auto")), input_func))
    set_nested(cfg, ["dataset", "name"], prompt_choice("Dataset", ("cifar10", "image_folder"), str(get_nested(cfg, ["dataset", "name"], "image_folder")), input_func, output_func))
    set_nested(cfg, ["dataset", "root"], prompt_string("Data root", str(get_nested(cfg, ["dataset", "root"], "datasets/kodak")), input_func))
    set_nested(cfg, ["dataset", "image_size"], prompt_int("Image size", int(get_nested(cfg, ["dataset", "image_size"], 256)), input_func, output_func))
    set_nested(cfg, ["model", "semantic_channels"], prompt_int("Semantic channels", int(get_nested(cfg, ["model", "semantic_channels"], 32)), input_func, output_func))
    set_nested(cfg, ["model", "base_channels"], prompt_int("Base channels", int(get_nested(cfg, ["model", "base_channels"], 32)), input_func, output_func))
    set_nested(cfg, ["checkpoint"], prompt_string("Checkpoint path (blank for random model)", get_nested(cfg, ["checkpoint"], None), input_func))
    set_nested(cfg, ["channel", "type"], prompt_choice("Channel", ("none", "awgn", "rayleigh"), str(get_nested(cfg, ["channel", "type"], "awgn")), input_func, output_func))
    set_nested(cfg, ["channel", "snr_db"], prompt_float_list("Evaluation SNR dB values", get_nested(cfg, ["channel", "snr_db"], [10.0]), input_func, output_func))
    set_nested(cfg, ["baseline", "codec"], prompt_choice("Baseline codec", ("jpeg", "bpg"), str(get_nested(cfg, ["baseline", "codec"], "jpeg")), input_func, output_func))
    set_nested(cfg, ["baseline", "jpeg_quality"], prompt_int("JPEG quality", int(get_nested(cfg, ["baseline", "jpeg_quality"], 35)), input_func, output_func))
    set_nested(cfg, ["baseline", "bpg_qp"], prompt_int("BPG QP", int(get_nested(cfg, ["baseline", "bpg_qp"], 29)), input_func, output_func))
    set_nested(cfg, ["evaluation", "monte_carlo_samples"], prompt_int("Monte Carlo samples", int(get_nested(cfg, ["evaluation", "monte_carlo_samples"], 1)), input_func, output_func))
    set_nested(cfg, ["output_dir"], prompt_string("Output directory", str(get_nested(cfg, ["output_dir"], "outputs/eval")), input_func))
    return cfg
