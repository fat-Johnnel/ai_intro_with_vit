from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .data import DataConfig, build_dataloaders
from .metrics import binary_classification_metrics, confusion_matrix_binary
from .modeling import create_resnet_classifier, infer_normalization
from .utils import Checkpoint, ensure_dir, get_device, set_seed


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    backbone: str = "resnet18"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    dropout: float = 0.2
    device: Optional[str] = None
    out_dir: Path = Path("artifacts")

    # Weights & Biases (wandb)
    wandb: bool = False
    wandb_project: str = "ai_intro"
    wandb_entity: str = "paulkm-huazhong-university-of-science-and-technology"
    wandb_run_name: Optional[str] = None


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())



def train_one_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler], # pyright: ignore[reportPrivateImportUsage]
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            assert scaler is not None
            with torch.amp.autocast('cuda'): # pyright: ignore[reportPrivateImportUsage]
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        running_acc += _accuracy(logits.detach(), y)
        n_batches += 1

    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    positive_label: int = 1,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    all_preds: List[torch.Tensor] = []
    all_true: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_true.append(y.detach().cpu())

    y_pred = torch.cat(all_preds) if all_preds else torch.empty((0,), dtype=torch.int64)
    y_true = torch.cat(all_true) if all_true else torch.empty((0,), dtype=torch.int64)

    m = binary_classification_metrics(y_true, y_pred, positive_label=positive_label)
    cm = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "acc": m.accuracy,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "tp": float(cm["tp"]),
        "tn": float(cm["tn"]),
        "fp": float(cm["fp"]),
        "fn": float(cm["fn"]),
    }


def train(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    out_dir = ensure_dir(cfg.out_dir)
    ckpt_path = out_dir / "model_best.pt"

    mean, std = infer_normalization()

    data_cfg = DataConfig(
        data_dir=cfg.data_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train_loader, val_loader, _, class_names = build_dataloaders(
        config=data_cfg, mean=mean, std=std, pin_memory=(device.type == "cuda")
    )

    model = create_resnet_classifier(
        backbone=cfg.backbone,
        num_classes=len(class_names),
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) # pyright: ignore[reportPrivateImportUsage]

    positive_label = 1
    if "dog" in class_names:
        positive_label = int(class_names.index("dog"))

    best_f1 = -1.0

    wandb_run = None
    wandb_mod = None
    if cfg.wandb:
        try:
            import wandb as wandb_mod
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "wandb is not available. Install it (e.g. `uv add wandb` or `pip install wandb`) "
                "or run without --wandb."
            ) from e

        # Convert config to something JSON-serializable for W&B.
        cfg_dict = asdict(cfg)
        cfg_dict["data_dir"] = str(cfg.data_dir)
        cfg_dict["out_dir"] = str(cfg.out_dir)
        cfg_dict["device"] = str(device)
        cfg_dict["use_amp"] = bool(use_amp)
        cfg_dict["class_names"] = list(class_names)

        wandb_run = wandb_mod.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=cfg_dict,
        )

    try:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler if use_amp else None,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )
            va = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                positive_label=positive_label,
            )
            scheduler.step()

            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"lr={lr:.2e} | "
                f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
                f"val loss={va['loss']:.4f} acc={va['acc']:.4f} "
                f"p={va['precision']:.4f} r={va['recall']:.4f} f1={va['f1']:.4f} | "
                f"{dt:.1f}s"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "time/epoch_sec": dt,
                        "lr": lr,
                        "train/loss": tr["loss"],
                        "train/acc": tr["acc"],
                        "val/loss": va["loss"],
                        "val/acc": va["acc"],
                        "val/precision": va["precision"],
                        "val/recall": va["recall"],
                        "val/f1": va["f1"],
                        "val/tp": va["tp"],
                        "val/tn": va["tn"],
                        "val/fp": va["fp"],
                        "val/fn": va["fn"],
                    },
                    step=epoch,
                )

            if va["f1"] > best_f1:
                best_f1 = float(va["f1"])
                Checkpoint(ckpt_path).save(
                    model=model,
                    class_names=class_names,
                    backbone=cfg.backbone,
                    img_size=cfg.img_size,
                    mean=mean,
                    std=std,
                    extra={
                        "best_val": va,
                        "train_cfg": {
                            "epochs": cfg.epochs,
                            "lr": cfg.lr,
                            "batch_size": cfg.batch_size,
                            "weight_decay": cfg.weight_decay,
                            "dropout": cfg.dropout,
                            "seed": cfg.seed,
                        },
                    },
                )
                print(f"  saved best checkpoint -> {ckpt_path}")

        print(f"Best val f1: {best_f1:.4f}")

        if wandb_run is not None and wandb_mod is not None:
            wandb_run.summary["best_val_f1"] = float(best_f1)
            # Log the (final) best checkpoint as an artifact.
            artifact = wandb_mod.Artifact(
                name="model_best",
                type="model",
                metadata={
                    "backbone": cfg.backbone,
                    "img_size": cfg.img_size,
                    "class_names": list(class_names),
                },
            )
            artifact.add_file(str(ckpt_path))
            wandb_run.log_artifact(artifact)

        return ckpt_path
    finally:
        if wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass

