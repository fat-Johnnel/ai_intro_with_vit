from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


def build_transforms(*, img_size: int, is_train: bool, mean, std):
    try:
        from torchvision import transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required. Install torch + torchvision first."
        ) from e

    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _dataset_from_folder(root: Path, transform):
    """Prefer ImageFolder structure: root/class_name/*.jpg"""
    try:
        from torchvision.datasets import ImageFolder
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required. Install torch + torchvision first."
        ) from e

    return ImageFolder(root=str(root), transform=transform)


def build_dataloaders(
    *,
    config: DataConfig,
    mean,
    std,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Tuple[str, ...]]:
    """Build dataloaders expecting the following layout:

    data_dir/
      train/cat/*.jpg
      train/dog/*.jpg
      val/cat/*.jpg
      val/dog/*.jpg
      test/cat/*.jpg (optional)
      test/dog/*.jpg (optional)
    """
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    test_dir = config.data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train/ and val/ under {config.data_dir}. "
            "Run prepare_data first, or supply a directory with that structure."
        )

    train_tf = build_transforms(img_size=config.img_size, is_train=True, mean=mean, std=std)
    eval_tf = build_transforms(img_size=config.img_size, is_train=False, mean=mean, std=std)

    train_ds = _dataset_from_folder(train_dir, train_tf)
    val_ds = _dataset_from_folder(val_dir, eval_tf)

    class_names = tuple(train_ds.classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dir.exists():
        test_ds = _dataset_from_folder(test_dir, eval_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader, class_names
