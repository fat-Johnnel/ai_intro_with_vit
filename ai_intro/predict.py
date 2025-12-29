from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, cast

import torch

from .modeling import create_resnet_classifier
from .utils import Checkpoint


def _build_eval_transform(*, img_size: int, mean, std):
    try:
        from torchvision import transforms
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required. Install torch + torchvision first."
        ) from e

    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def predict_image(
    *,
    checkpoint_path: Path,
    image_path: Path,
    device: torch.device,
) -> Tuple[str, float, List[float]]:
    payload = Checkpoint(checkpoint_path).load(map_location=device)

    class_names = tuple(payload["class_names"])
    backbone = str(payload["backbone"])
    img_size = int(payload["img_size"])
    mean = tuple(payload["normalize"]["mean"])
    std = tuple(payload["normalize"]["std"])

    state_dict = payload["model_state_dict"]
    extra = payload.get("extra") or {}
    train_cfg = extra.get("train_cfg") or {}
    dropout = float(train_cfg.get("dropout", 0.0))

    has_sequential_head = any(k.startswith("fc.1.") for k in state_dict.keys())
    has_plain_head = ("fc.weight" in state_dict) or ("fc.bias" in state_dict)
    if has_sequential_head and dropout <= 0:
        dropout = 0.2
    if has_plain_head and dropout > 0:
        dropout = 0.0

    model = create_resnet_classifier(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    from PIL import Image

    tf = _build_eval_transform(img_size=img_size, mean=mean, std=std)

    img = Image.open(image_path).convert("RGB")
    x = cast(torch.Tensor, tf(img)).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    idx = int(torch.argmax(probs).item())
    label = class_names[idx]
    conf = float(probs[idx].item())
    return label, conf, [float(p) for p in probs.tolist()]

