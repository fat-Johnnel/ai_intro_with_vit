from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Checkpoint:
    path: Path

    def save(
        self,
        *,
        model: torch.nn.Module,
        class_names: Tuple[str, ...],
        backbone: str,
        img_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "class_names": list(class_names),
            "backbone": backbone,
            "img_size": img_size,
            "normalize": {"mean": list(mean), "std": list(std)},
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, self.path)

    def load(self, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
        return torch.load(self.path, map_location=map_location)
