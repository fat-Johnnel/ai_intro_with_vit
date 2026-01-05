from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def create_resnet_classifier(
    *,
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a ResNet classifier using torchvision backbones.

    Supported: resnet18/resnet34/resnet50/resnet101.
    """
    try:
        from torchvision.models import (
            ResNet18_Weights,
            ResNet34_Weights,
            ResNet50_Weights,
            ResNet101_Weights,
            resnet18,
            resnet34,
            resnet50,
            resnet101,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required. Install torch + torchvision first."
        ) from e

    backbone = backbone.lower().strip()

    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    elif backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    elif backbone == "resnet101":
        weights = ResNet101_Weights.DEFAULT if pretrained else None
        model = resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.fc.in_features
    if dropout and dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, num_classes))  # type: ignore
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def infer_normalization() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """ImageNet normalization used by torchvision ResNet weights."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return mean, std


def create_vit_classifier(
    *,
    backbone: str = "vit_base_patch16_224",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Create a Vision Transformer classifier using timm.

    Example backbones: vit_base_patch16_224, vit_large_patch16_384
    """
    try:
        import timm
    except Exception as e:  # pragma: no cover
        raise RuntimeError("timm is required. Install it via `pip install timm`.") from e

    # timm handles position embedding interpolation internally for common backbones.
    model = timm.create_model(str(backbone), pretrained=bool(pretrained), num_classes=int(num_classes))

    # Some timm models accept drop_rate / drop_path_rate on creation; we keep interface simple.
    if dropout and float(dropout) > 0:
        # Try setting a dropout attribute if available
        try:
            if hasattr(model, "drop_rate"):
                setattr(model, "drop_rate", float(dropout))
        except Exception:
            pass

    return model
