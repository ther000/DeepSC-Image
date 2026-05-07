"""Single-image inference CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .baseline import VALID_BASELINE_CODECS
from .channels import ChannelConfig
from .inference import run_inference
from .model import DeepSCImageModel
from .utils import load_checkpoint, pil_to_tensor, resolve_device, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSC reconstruction on one image")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="outputs/infer/reconstruction.png")
    parser.add_argument("--baseline-output", default="outputs/infer/baseline.png")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--channel", default="awgn", choices=["none", "awgn", "rayleigh"])
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--semantic-channels", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--baseline-codec", default="jpeg", choices=VALID_BASELINE_CODECS)
    parser.add_argument("--jpeg-quality", type=int, default=35)
    parser.add_argument("--bpg-qp", type=int, default=29)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = DeepSCImageModel(args.semantic_channels, args.base_channels).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, device)
    else:
        print("No checkpoint supplied; using random initialized model for functional visualization only.")
    tensor = pil_to_tensor(Image.open(args.input), image_size=args.image_size)
    result = run_inference(
        model,
        tensor,
        ChannelConfig(args.channel, args.snr_db),
        jpeg_quality=args.jpeg_quality,
        baseline_codec=args.baseline_codec,
        bpg_qp=args.bpg_qp,
        device=device,
    )
    out_path = Path(args.output)
    base_path = Path(args.baseline_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(result.reconstruction).save(out_path)
    tensor_to_pil(result.baseline).save(base_path)
    print(json.dumps({
        "output": str(out_path),
        "baseline_output": str(base_path),
        "deepsc": result.deepsc_metrics,
        "baseline_codec": result.baseline_codec,
        f"{result.baseline_codec}_baseline": result.baseline_metrics,
        "latency_ms": result.latency_ms,
        "baseline_latency_ms": result.baseline_latency_ms,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
