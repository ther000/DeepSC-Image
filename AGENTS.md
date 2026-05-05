# AGENTS.md

## Project Shape

- This is a `src/`-layout Python package (`deepsc_image`) for DeepSC-style image transmission: model/channel/loss/metrics code lives in `src/deepsc_image/`, configs live in `configs/`, and tests live in `tests/`.
- Main executable entrypoints are module CLIs, not top-level scripts: `python -m deepsc_image.train`, `python -m deepsc_image.evaluate`, `python -m deepsc_image.infer`, `python -m deepsc_image.smoke_test`, and `python -m deepsc_image.benchmark_training`.
- The GUI is Streamlit-only: run it with `streamlit run src/deepsc_image/app.py`; it reads `configs/gui.yaml` from the repo root.

## Setup And Imports

- Install editable mode before running module commands or tests: `pip install -r requirements.txt` then `pip install -e .`. Tests need pytest, so use `pip install -e ".[dev]"` if pytest is missing.
- If editable install is skipped, module imports depend on setting `PYTHONPATH=src`; prefer editable install instead of baking path hacks into code.
- `device: auto` resolves to CUDA when available and CPU otherwise; CPU must keep working for smoke tests and GUI/inference pipeline checks.

## Verification Commands

- Fast no-dataset smoke check: `python -m deepsc_image.smoke_test`; success prints `SMOKE_TEST_OK ...`.
- Unit tests: `pytest`. For focused checks use paths such as `pytest tests/test_cli_config.py` or `pytest tests/test_training_speed_config.py`.
- Quick training connectivity uses the real CIFAR config but can be shortened: `python -m deepsc_image.train --config configs/train_cifar10_awgn.yaml --epochs 1`.
- Evaluation writes artifacts with: `python -m deepsc_image.evaluate --config configs/eval_kodak.yaml --checkpoint outputs/train_cifar10_awgn/<experiment_dir>/best_model.pth`.
- Benchmarking requires a single semantic channel when the YAML contains a list: add `--semantic-channels 16` to `python -m deepsc_image.benchmark_training ...`.

## Config And Runtime Gotchas

- Training/evaluation configs are YAML-first; CLI flags override nested YAML values and `--interactive` changes only the in-memory run config, not files in `configs/`.
- Default CIFAR configs use `dataset.download: false`; training expects CIFAR-10 under `datasets/cifar10/cifar-10-batches-py` unless `--download` or YAML is changed.
- `configs/eval_kodak.yaml` expects image files in `datasets/kodak` and writes `metrics.json`, `psnr_vs_snr.png`, and `ssim_vs_snr.png` under `outputs/eval_kodak` by default.
- A list in `model.semantic_channels` triggers one training output subdirectory per value while sharing a timestamp; generated run dirs include channel, SNR, semantic channels, image size, and seed.
- Training writes `best_model.pth`, `last_model.pth`, `config.yaml`, `history.csv`, `history.json`, `loss_curve.png`, and `summary.json`; there are no per-epoch `epoch_*.pth` checkpoints.
- `best_model.pth` is selected by lowest training loss because there is no validation split; do not describe it as validation-best.
- Missing checkpoints are allowed for GUI/infer/evaluate pipeline validation but mean random-initialized model quality; never use those results as DeepSC performance evidence.
- Checkpoint loading is intentionally restricted to raw state dicts or this project's payload keys (`model_state`, `epoch`, `config`, `bandwidth_estimate`); avoid adding unsafe `torch.load` paths.
- AMP is only effective on CUDA even if requested; `persistent_workers` and `prefetch_factor` are only active when `training.dataloader.num_workers > 0`.

## Data And Artifacts

- Do not assume `datasets/` and `outputs/` are untracked just because README says large artifacts are excluded; this repo currently tracks datasets and several output/checkpoint artifacts.
- Before deleting, rewriting, or committing anything under `datasets/` or `outputs/`, check `git ls-files`/`git status` and preserve user-generated experiment results unless explicitly told otherwise.
- `.gitignore` currently ignores caches, virtualenvs, `.streamlit/secrets.toml`, and `.sisyphus/`, but not `datasets/` or `outputs/`.

## Code Conventions

- Keep the JPEG baseline dependency-light in `baseline.py`; the current project intentionally avoids external BPG/system codec requirements for Windows/CPU usability.
- Keep plotting/artifact generation dependency-light: training/evaluation curves use matplotlib with a headless backend.
- When changing model dimensions or checkpoint handling, update GUI loading tests because `app.load_model` reads embedded checkpoint config to construct the matching model before loading weights.
