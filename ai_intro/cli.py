from __future__ import annotations

import argparse
from typing import List


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ai_intro",
        description="Deep-learning image classification (cats vs dogs) using PyTorch + ResNet",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 准备数据
    sp = sub.add_parser("prepare-data", help="准备 Kaggle Dogs vs Cats 数据集，划分为训练/验证/测试文件夹")
    sp.add_argument("--raw-train-dir", required=True, help="原始图像目录（平铺的 cat.*/dog.* 或 PetImages/ 目录下的 Cat/ + Dog/）")
    sp.add_argument("--out-dir", default="data")
    sp.add_argument("--val-ratio", type=float, default=0.1)
    sp.add_argument("--test-ratio", type=float, default=0.1)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dedupe", action="store_true")
    sp.add_argument("--max-samples", type=int, default=None)
    sp.add_argument("--img-size", type=int, default=224, help="If set, pad to square and resize to this size")
    sp.add_argument("--pad-color", default="0,0,0", help="RGB pad color, e.g. '0,0,0' for black")

    # train
    sp = sub.add_parser("train", help="训练分类模型")
    sp.add_argument("--data-dir", default="data")
    sp.add_argument("--backbone", default="resnet18")
    sp.add_argument("--img-size", type=int, default=224)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--weight-decay", type=float, default=1e-4)
    sp.add_argument("--num-workers", type=int, default=2)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--dropout", type=float, default=0.2)
    sp.add_argument("--device", default=None)
    sp.add_argument("--out-dir", default="artifacts")
    sp.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    sp.add_argument("--wandb-project", default="ai-intro")
    sp.add_argument("--wandb-entity", default=None)
    sp.add_argument("--wandb-run-name", default=None)

    # predict
    sp = sub.add_parser("predict", help="Predict a single image")
    sp.add_argument("--checkpoint", default="artifacts/model_best.pt")
    sp.add_argument("--image", required=True)
    sp.add_argument("--device", default=None)

    # eval
    sp = sub.add_parser("eval", help="Evaluate a checkpoint on val/test")
    sp.add_argument("--checkpoint", default="artifacts/model_best.pt")
    sp.add_argument("--data-dir", default="data")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--batch-size", type=int, default=64)
    sp.add_argument("--num-workers", type=int, default=2)
    sp.add_argument("--device", default=None)

    args = p.parse_args(argv)

    if args.cmd == "prepare-data":
        from pathlib import Path

        from .prepare_data import prepare_kaggle_dogs_vs_cats

        stats = prepare_kaggle_dogs_vs_cats(
            raw_train_dir=Path(args.raw_train_dir),
            out_dir=Path(args.out_dir),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
            dedupe=bool(args.dedupe),
            max_samples=args.max_samples,
            img_size=int(args.img_size) if args.img_size is not None else None,
            pad_color=tuple(int(x) for x in str(args.pad_color).split(",")),
        )
        print("Prepared dataset:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "train":
        from pathlib import Path

        from .train import TrainConfig, train

        cfg = TrainConfig(
            data_dir=Path(args.data_dir),
            backbone=str(args.backbone),
            img_size=int(args.img_size),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            dropout=float(args.dropout),
            device=args.device,
            out_dir=Path(args.out_dir),
            wandb=bool(args.wandb),
            wandb_project=str(args.wandb_project),
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
        )
        path = train(cfg)
        print(f"checkpoint: {path}")
        return 0

    if args.cmd == "predict":
        from pathlib import Path

        from .predict import predict_image
        from .utils import get_device

        device = get_device(args.device)
        label, conf, probs = predict_image(
            checkpoint_path=Path(args.checkpoint),
            image_path=Path(args.image),
            device=device,
        )
        print(f"prediction: {label} (confidence={conf:.4f})")
        print(f"probs: {probs}")
        return 0

    if args.cmd == "eval":
        from .evaluate import main as eval_main

        argv2 = [
            "--checkpoint",
            str(args.checkpoint),
            "--data-dir",
            str(args.data_dir),
            "--split",
            str(args.split),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.device is not None:
            argv2 += ["--device", str(args.device)]
        return eval_main(argv2)

    raise RuntimeError(f"Unknown command: {args.cmd}")
