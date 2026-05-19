"""
YOLO detection data loading adapted from iMAS_Remake.
Provides:
  - YoloDetDataset: loads images + YOLO-format labels with augmentation
  - build_yolo_semi_loaders: creates separate labeled / unlabeled / val loaders
    with labeled oversampling to match unlabeled count (iMAS-style).
"""
from __future__ import annotations

import math
import random as _rng
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .det_augmentations import (
    Compose, Resize, RandomFlip, Crop, ToTensorNorm, StrongImgAug,
)

# Avoid crashes on truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _discover_images(directory: Path) -> list[Path]:
    """Recursively discover all image files in a directory."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read_yolo_label(path: Path, allow_missing: bool = False) -> np.ndarray:
    """Read YOLO label file: class cx cy w h (normalized). Returns (N, 5)."""
    if not path.exists():
        if allow_missing:
            return np.zeros((0, 5), dtype=np.float32)
        raise FileNotFoundError(str(path))
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls, xc, yc, w, h = map(float, parts[:5])
                    rows.append([cls, xc, yc, w, h])
                except ValueError:
                    continue
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _image_to_label_path(image_path: Path, labels_dir: Optional[Path] = None) -> Path:
    """Convert image path to label path: .../images/x.jpg → .../labels/x.txt."""
    if labels_dir is not None:
        return labels_dir / (image_path.stem + ".txt")
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _make_to_tensor():
    """Just ToTensor for YOLO: [0,255] → [0,1], no ImageNet normalization."""
    return transforms.ToTensor()


# ============================================================================
# Collate functions
# ============================================================================

def collate_labeled(batch):
    """Labeled batch: (index, image_tensor, boxes_tensor)."""
    indices = [b[0] for b in batch]
    images = torch.stack([b[1] for b in batch], dim=0)
    boxes = [b[2] for b in batch]
    return indices, images, boxes


def collate_unlabeled(batch):
    """Unlabeled dual-view batch: (index, weak_tensor, strong_tensor, boxes_tensor)."""
    indices = [b[0] for b in batch]
    weak = torch.stack([b[1] for b in batch], dim=0)
    strong = torch.stack([b[2] for b in batch], dim=0)
    boxes = [b[3] for b in batch]
    return indices, weak, strong, boxes


# ============================================================================
# Dataset
# ============================================================================

class YoloDetDataset(Dataset):
    """
    Dataset that loads images and YOLO-format bounding-box labels.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing label .txt files (optional,
                     defaults to sibling 'labels' folder).
        allow_missing_labels: If True, missing label files return empty boxes.
        return_dual_view: If True, return (weak_img, strong_img) for
                          semi-supervised unlabeled training.
        geometric_aug: Compose of (image, boxes) → (image, boxes) transforms.
        strong_aug: image-only strong augmentation for the second view.
        color_jitter: Apply random ColorJitter on labeled images.
    """

    def __init__(
        self,
        images_dir,
        labels_dir=None,
        allow_missing_labels=False,
        return_dual_view=False,
        geometric_aug=None,
        strong_aug=None,
        color_jitter=False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.allow_missing = allow_missing_labels
        self.dual_view = return_dual_view
        self.to_tensor = _make_to_tensor()
        self.geo_aug = geometric_aug
        self.strong_aug = strong_aug
        self.jitter = (transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                               saturation=0.3, hue=0.02)
                       if color_jitter else None)
        self.image_paths = _discover_images(self.images_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        lbl_path = _image_to_label_path(img_path, self.labels_dir)
        image = Image.open(img_path).convert("RGB")
        boxes = _read_yolo_label(lbl_path, allow_missing=self.allow_missing)

        # Geometric augmentation (modifies image + boxes)
        if self.geo_aug is not None:
            image, boxes = self.geo_aug(image, boxes)

        # Color jitter for labeled data
        if self.jitter is not None:
            image = self.jitter(image)

        boxes_tensor = torch.from_numpy(boxes).float() if isinstance(boxes, np.ndarray) \
                       else boxes.clone() if isinstance(boxes, torch.Tensor) \
                       else torch.zeros((0, 5), dtype=torch.float32)

        if self.dual_view:
            img_weak = self.to_tensor(image)  # weak view
            if self.strong_aug is not None:
                img_strong_pil, _ = self.strong_aug(image, None)
                img_strong = self.to_tensor(img_strong_pil)
            else:
                img_strong = self.to_tensor(image)
            return index, img_weak, img_strong, boxes_tensor

        img_tensor = self.to_tensor(image)
        return index, img_tensor, boxes_tensor


# ============================================================================
# Loader builder (iMAS-style separate loaders + oversampling)
# ============================================================================

def build_yolo_semi_loaders(cfg):
    """
    Build labeled, unlabeled, and val DataLoaders from config dict.

    Expected config keys:
        root_path: base directory of the banana_data folder
        img_size: image size (default 1024)
        batch_size_lb: labeled batch size per iteration
        batch_size_ulb: unlabeled batch size per iteration
        num_workers: dataloader workers
        num_strong_augs: number of strong augs applied per unlabeled image

    Data directory layout (under root_path):
        train/labeled/images/  train/labeled/labels/  train/labeled/list.txt
        train/unlabeled/images/                        train/unlabeled/list.txt
        val/images/  val/labels/  val/list.txt
    """
    root = Path(cfg["root_path"])
    imgsz = int(cfg.get("img_size", 1024))
    bs_lb = int(cfg.get("batch_size_lb", 4))
    bs_ulb = int(cfg.get("batch_size_ulb", 4))
    workers = int(cfg.get("num_workers", 4))
    n_strong = int(cfg.get("num_strong_augs", 2))

    # Geometric augmentation (iMAS-style: Resize → Flip → Crop)
    geo_aug = Compose([
        Resize(base_size=[imgsz, imgsz], rand_resize=[0.5, 2.0]),
        RandomFlip(prob=0.5),
        Crop(crop_size=[imgsz, imgsz], crop_type="rand"),
    ])

    # Strong augmentation for unlabeled second view
    strong_aug = StrongImgAug(num_augs=n_strong)

    # --- Labeled dataset ---
    labeled_dir = root / "train" / "labeled"
    labeled_ds = YoloDetDataset(
        images_dir=labeled_dir / "images",
        labels_dir=labeled_dir / "labels",
        allow_missing_labels=False,
        geometric_aug=geo_aug,
        color_jitter=True,
    )

    # --- Unlabeled dataset (dual-view) ---
    unlabeled_dir = root / "train" / "unlabeled"
    unlabeled_ds = YoloDetDataset(
        images_dir=unlabeled_dir / "images",
        labels_dir=None,
        allow_missing_labels=True,
        return_dual_view=True,
        geometric_aug=geo_aug,
        strong_aug=strong_aug,
    )

    # --- Val dataset ---
    val_dir = root / "val"
    val_ds = YoloDetDataset(
        images_dir=val_dir / "images",
        labels_dir=val_dir / "labels",
        allow_missing_labels=True,
    )

    # Oversample labeled to match unlabeled (iMAS-style)
    n_lb_orig = len(labeled_ds)
    n_ulb = len(unlabeled_ds)
    if n_lb_orig > 0 and n_ulb > 0 and n_lb_orig < n_ulb:
        num_repeat = math.ceil(n_ulb / n_lb_orig)
        full_indices = list(range(n_lb_orig)) * num_repeat
        oversampled = _rng.sample(full_indices, n_ulb)
        labeled_ds = Subset(labeled_ds, oversampled)
        print(f"[DATA] Oversample labeled {n_lb_orig} → {len(labeled_ds)} "
              f"(repeat x{num_repeat}) to match {n_ulb} unlabeled")

    print(f"[DATA] labeled={len(labeled_ds)}, unlabeled={len(unlabeled_ds)}, val={len(val_ds)}")

    loader_lb = DataLoader(
        labeled_ds, batch_size=bs_lb, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_worker, collate_fn=collate_labeled,
    )
    loader_ulb = DataLoader(
        unlabeled_ds, batch_size=bs_ulb, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_worker, collate_fn=collate_unlabeled,
    )
    loader_val = DataLoader(
        val_ds, batch_size=bs_lb, shuffle=False,
        num_workers=workers, pin_memory=False, drop_last=False,
        worker_init_fn=_seed_worker, collate_fn=collate_labeled,
    )
    return loader_lb, loader_ulb, loader_val
