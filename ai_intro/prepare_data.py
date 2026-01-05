from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def _is_image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _label_from_filename(name: str) -> str:
    # Kaggle Dogs vs Cats naming: cat.0.jpg / dog.123.jpg
    lower = name.lower()
    if lower.startswith("cat"):
        return "cat"
    if lower.startswith("dog"):
        return "dog"
    raise ValueError(f"Cannot infer label from filename: {name}")


def _label_from_path(path: Path) -> str:
    """Infer label from either parent folder or filename.

    Supports both layouts:
      - PetImages/Cat/0001.jpg, PetImages/Dog/0001.jpg
      - cat.0.jpg, dog.1.jpg
    """
    parent = path.parent.name.lower()
    if parent in {"cat", "dog"}:
        return parent
    return _label_from_filename(path.name)


def _collect_raw_images(raw_dir: Path) -> List[Path]:
    """Collect image files from supported raw dataset layouts."""
    raw_dir = raw_dir.resolve()

    # Prefer PetImages-style folders if present.
    class_dirs = [
        p
        for p in raw_dir.iterdir()
        if p.is_dir() and p.name.lower() in {"cat", "dog"}
    ]
    if class_dirs:
        files: List[Path] = []
        for d in sorted(class_dirs, key=lambda p: p.name.lower()):
            files.extend([p for p in d.iterdir() if p.is_file()])
        return sorted(files)

    # Fallback: flat directory of images.
    return sorted([p for p in raw_dir.iterdir() if p.is_file()])


def _copy_to_split(
    *,
    files: List[Path],
    out_dir: Path,
    split: str,
    img_size: int | None = None,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    for src in files:
        label = _label_from_path(src)
        dst_dir = out_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if img_size is None:
            shutil.copy2(src, dst)
        else:
            _process_and_save(src=src, dst=dst, img_size=img_size, pad_color=pad_color)


def _process_and_save(
    *,
    src: Path,
    dst: Path,
    img_size: int | None = None,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Open image, pad to square with pad_color if needed, resize to img_size, and save to dst."""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w != h:
                m = max(w, h)
                # create square canvas and paste centered
                canvas = Image.new("RGB", (m, m), pad_color)
                paste_x = (m - w) // 2
                paste_y = (m - h) // 2
                canvas.paste(im, (paste_x, paste_y))
                im = canvas

            if img_size is not None:
                im = im.resize((img_size, img_size), resample=Image.BILINEAR)

            im.save(dst)
    except Exception:
        # If processing fails, skip copying and do a direct copy as fallback
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def prepare_kaggle_dogs_vs_cats(
    *,
    raw_train_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dedupe: bool = False,
    max_samples: int | None = None,
    img_size: int | None = None,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, int]:
    """Prepare folder structure suitable for ImageFolder.

        Input:
            raw_train_dir:
                - directory containing images like cat.0.jpg, dog.1.jpg, OR
                - directory containing Cat/ and Dog/ subfolders (PetImages layout)

    Output:
      out_dir/train/{cat,dog}/*.jpg
      out_dir/val/{cat,dog}/*.jpg
      out_dir/test/{cat,dog}/*.jpg
    """
    raw_train_dir = raw_train_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = _collect_raw_images(raw_train_dir)

    # basic cleaning: unreadable images
    ok_files: List[Path] = [p for p in all_files if _is_image_ok(p)]

    if dedupe:
        seen: Dict[str, Path] = {}
        deduped: List[Path] = []
        for p in ok_files:
            h = _file_sha256(p)
            if h in seen:
                continue
            seen[h] = p
            deduped.append(p)
        ok_files = deduped

    if max_samples is not None:
        ok_files = ok_files[: int(max_samples)]

    random.seed(seed)
    random.shuffle(ok_files)

    n = len(ok_files)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_files = ok_files[:n_test]
    val_files = ok_files[n_test : n_test + n_val]
    train_files = ok_files[n_test + n_val :]

    _copy_to_split(files=train_files, out_dir=out_dir, split="train", img_size=img_size, pad_color=pad_color)
    _copy_to_split(files=val_files, out_dir=out_dir, split="val", img_size=img_size, pad_color=pad_color)
    _copy_to_split(files=test_files, out_dir=out_dir, split="test", img_size=img_size, pad_color=pad_color)

    return {
        "total": n,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }

