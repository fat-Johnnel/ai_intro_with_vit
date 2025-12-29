from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def binary_classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, positive_label: int = 1
) -> ClassificationMetrics:
    """Compute metrics for a binary classification problem.

    y_true: shape [N], int64
    y_pred: shape [N], int64
    """
    y_true = y_true.detach().to(torch.int64).view(-1)
    y_pred = y_pred.detach().to(torch.int64).view(-1)

    pos = int(positive_label)
    neg = 1 - pos

    tp = int(((y_true == pos) & (y_pred == pos)).sum().item())
    tn = int(((y_true == neg) & (y_pred == neg)).sum().item())
    fp = int(((y_true == neg) & (y_pred == pos)).sum().item())
    fn = int(((y_true == pos) & (y_pred == neg)).sum().item())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    return ClassificationMetrics(accuracy=acc, precision=prec, recall=rec, f1=f1)


def confusion_matrix_binary(
    y_true: torch.Tensor, y_pred: torch.Tensor, positive_label: int = 1
) -> Dict[str, int]:
    y_true = y_true.detach().to(torch.int64).view(-1)
    y_pred = y_pred.detach().to(torch.int64).view(-1)

    pos = int(positive_label)
    neg = 1 - pos

    return {
        "tp": int(((y_true == pos) & (y_pred == pos)).sum().item()),
        "tn": int(((y_true == neg) & (y_pred == neg)).sum().item()),
        "fp": int(((y_true == neg) & (y_pred == pos)).sum().item()),
        "fn": int(((y_true == pos) & (y_pred == neg)).sum().item()),
    }
