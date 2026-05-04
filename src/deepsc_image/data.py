"""Dataset builders for training/evaluation without implicit downloads."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import pil_to_tensor


class ImageFolderDataset(Dataset[torch.Tensor]):
    def __init__(self, root: str | Path, image_size: int | None = None) -> None:
        self.root = Path(root)
        self.image_size = image_size
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        self.paths = sorted(path for pattern in patterns for path in self.root.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.root}. Put Kodak/local images there or change config.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return pil_to_tensor(Image.open(self.paths[index]), self.image_size).squeeze(0)


def build_dataset(config: dict, train: bool) -> Dataset[torch.Tensor]:
    name = str(config.get("name", "image_folder")).lower()
    image_size = config.get("image_size")
    if name == "image_folder":
        return ImageFolderDataset(config.get("root", "datasets/images"), image_size=image_size)
    if name == "cifar10":
        try:
            from torchvision.datasets import CIFAR10
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError("CIFAR-10 loading requires torchvision. Install requirements.txt first.") from exc
        transform = transforms.Compose([transforms.Resize((int(image_size or 32), int(image_size or 32))), transforms.ToTensor()])
        return CIFAR10(
            root=str(config.get("root", "datasets/cifar10")),
            train=train,
            transform=transform,
            download=bool(config.get("download", False)),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def unwrap_batch(batch: object) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError(f"Unsupported dataloader batch type: {type(batch)!r}")
