# Interpretation Notes

## Experiment Setup

This experiment studies the DeepSC semantic bottleneck by varying `semantic_channels` in `{16, 32, 64}` while keeping the evaluation channel, dataset, SNR grid, and metrics fixed. The reported ratio is a semantic bottleneck/channel-use ratio and is not a traditional file-size compression ratio. It should not be interpreted as entropy-coded rate or matched to JPEG by bitrate.

The AWGN checkpoints were selected from the same multi-SNR training family and evaluated on `datasets/kodak` with `configs/eval_kodak.yaml`. The evaluation uses SNR values `[-5, 0, 5, 10, 15, 20]`, `monte_carlo_samples=5`, PSNR, and SSIM. Provenance is recorded in `manifest.json`, `commands.txt`, and `selected_checkpoints.json`.

## Main Trend

The results show a consistent quality increase as the semantic bottleneck widens. At 10 dB, PSNR rises from 31.520 dB for `semantic_channels=16` to 33.941 dB for `semantic_channels=32` and 37.029 dB for `semantic_channels=64`; SSIM rises from 0.9465 to 0.9708 and 0.9843. At 20 dB, PSNR is 32.845, 35.255, and 39.135 dB for 16, 32, and 64 channels respectively.

## Paper-Ready Interpretation

A smaller `semantic_channels` value creates a stronger semantic bottleneck and reduces transmitted semantic symbols, but it also limits reconstruction fidelity. Increasing the bottleneck width improves PSNR and SSIM across the tested AWGN SNR range, with `semantic_channels=64` giving the strongest reconstruction quality and `semantic_channels=16` giving the strongest bottleneck compression. `semantic_channels=32` is the middle tradeoff setting.

## Limitations

These conclusions are limited to this repository implementation, the selected CIFAR-trained AWGN checkpoints, the Kodak evaluation configuration, and the current PSNR/SSIM metrics. The JPEG baseline in this project uses fixed quality 35 and is not matched to JPEG by bitrate. The selected `best_model.pth` files are lowest-training-loss checkpoints, not validation-selected checkpoints. Rayleigh/Rician channels, perceptual metrics, and JPEG quality sweeps are outside this experiment scope.
