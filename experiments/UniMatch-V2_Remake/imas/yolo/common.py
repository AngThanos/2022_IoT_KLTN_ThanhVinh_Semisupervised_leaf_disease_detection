from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BANANA_ROOT = REPO_ROOT / "data" / "banana_data"

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(data: dict, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
    return target


def read_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def load_class_names(meta_dir: str | Path | None = None) -> list[str]:
    meta_dir = Path(meta_dir) if meta_dir is not None else BANANA_ROOT / "meta"
    classes_file = meta_dir / "classes.txt"
    if not classes_file.exists():
        return []
    return read_lines(classes_file)


def discover_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    images = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(images)


def image_to_label_path(image_path: str | Path, label_root_name: str = "labels") -> Path:
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = label_root_name
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def relative_to_banana(path: str | Path) -> Path:
    path = Path(path)
    try:
        return path.relative_to(BANANA_ROOT)
    except ValueError:
        return path


def try_hardlink_or_copy(src: str | Path, dst: str | Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    ensure_dir(dst.parent)
    if dst.exists():
        return dst
    try:
        src.parent.mkdir(parents=True, exist_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)
    return dst


def write_csv(rows: Iterable[dict], path: str | Path, fieldnames: list[str]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target
