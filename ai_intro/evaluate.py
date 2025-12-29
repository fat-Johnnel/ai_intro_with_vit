from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from .data import DataConfig, build_dataloaders
from .modeling import create_resnet_classifier
from .train import evaluate
from .utils import Checkpoint, get_device


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on val/test")
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/model_best.pt"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args(argv)

    device = get_device(args.device)
    payload = Checkpoint(args.checkpoint).load(map_location=device)

    class_names = tuple(payload["class_names"])
    backbone = str(payload["backbone"])
    img_size = int(payload["img_size"])
    mean = tuple(payload["normalize"]["mean"])
    std = tuple(payload["normalize"]["std"])

    state_dict = payload["model_state_dict"]
    extra = payload.get("extra") or {}
    train_cfg = extra.get("train_cfg") or {}
    dropout = float(train_cfg.get("dropout", 0.0))

    # Backward/forward compatibility:
    # - If checkpoint was trained with Dropout+Linear head, weights are under fc.1.*
    # - If checkpoint was trained with plain Linear head, weights are under fc.*
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

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, val_loader, test_loader, _ = build_dataloaders(
        config=data_cfg, mean=mean, std=std, pin_memory=(device.type == "cuda")
    )

    loader = val_loader if args.split == "val" else test_loader
    if loader is None:
        raise FileNotFoundError(
            f"Split {args.split} not found under {args.data_dir}. "
            "Ensure you have data/test/* or evaluate on --split val."
        )

    criterion = nn.CrossEntropyLoss()

    positive_label = 1
    if "dog" in class_names:
        positive_label = int(class_names.index("dog"))

    stats = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        positive_label=positive_label,
    )

    print(f"split: {args.split}")
    for k in ["loss", "acc", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]:
        print(f"{k}: {stats[k]:.6f}" if k in {"loss", "acc", "precision", "recall", "f1"} else f"{k}: {int(stats[k])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
