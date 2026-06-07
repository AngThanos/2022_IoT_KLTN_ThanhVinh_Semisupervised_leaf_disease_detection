"""
YOLO Semi-Supervised Dataset for UniMatch V2.

Provides dual-strong-view data loading for object detection,
following UniMatch V2's dual-stream augmentation strategy.

Imports data loading utilities from iMAS_Remake.
"""
from __future__ import annotations

import math
import random as _rng
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Add iMAS_Remake to path for data loading utilities
_IMAS_ROOT = Path(__file__).resolve().parents[2] / "iMAS_Remake"
if str(_IMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMAS_ROOT))

from imas.yolo.common import discover_images, image_to_label_path
from imas.dataset.augmentations import (
    Compose as AugCompose,
    Resize as AugResize,
    RandomFlip,
    Crop as AugCrop,
    ColorJitter as AugColorJitter,
    GaussianBlur as AugGaussianBlur,
    RandomGrayscale as AugRandomGrayscale,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Paper-faithful strong augmentation (UniMatch V2 Origin dataset/semi.py):
#   - ColorJitter(0.5, 0.5, 0.5, 0.25)  with p=0.8
#   - RandomGrayscale                   with p=0.2
#   - GaussianBlur (sigma 0.1-2.0)      with p=0.5
# Color-only ops preserve geometry, which is critical for detection.
# ---------------------------------------------------------------------------

class UniMatchStrongAug:
    """Origin UniMatch V2 strong augmentation pipeline."""

    def __init__(
        self,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_hue: float = 0.25,
        gray_prob: float = 0.2,
        blur_prob: float = 0.5,
        blur_radius: tuple = (0.1, 2.0),
    ):
        self.cj_prob = cj_prob
        self.color_jitter = AugColorJitter(
            brightness=cj_strength,
            contrast=cj_strength,
            saturation=cj_strength,
            hue=cj_hue,
        )
        self.gray = AugRandomGrayscale(p=gray_prob)
        self.blur = AugGaussianBlur(radius_range=blur_radius, prob=blur_prob)

    def __call__(self, image, label=None):
        if _rng.random() < self.cj_prob:
            image, _ = self.color_jitter(image, None)
        image, _ = self.gray(image, None)
        image, _ = self.blur(image, None)
        return image, label


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_image_transform():
    """YOLO expects [0,1] range (just /255). No ImageNet normalization."""
    return transforms.Compose([transforms.ToTensor()])


def _read_yolo_label(label_path: Path, allow_missing: bool = False) -> np.ndarray:
    if not label_path.exists():
        if allow_missing:
            return np.zeros((0, 5), dtype=np.float32)
        raise FileNotFoundError(str(label_path))
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls, xc, yc, w, h = map(float, parts[:5])
            except ValueError:
                continue
            rows.append([cls, xc, yc, w, h])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class YoloLabeledDataset(Dataset):
    """Labeled detection dataset with geometric augmentation only.

    Paper-faithful to UniMatch V2 (Origin dataset/semi.py `train_l` path):
    resize + crop + hflip, NO color jitter on labeled.
    Optional `color_jitter` flag for iMAS-style experiments.
    """

    def __init__(self, images_dir, labels_dir, geometric_aug=None, color_jitter=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = _make_image_transform()
        self.geometric_aug = geometric_aug
        self.color_jitter = (
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)
            if color_jitter else None
        )
        self.image_paths = [p for p in discover_images(self.images_dir)
                           if "checkpoint" not in p.name]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        rel = img_path.relative_to(self.images_dir)
        lbl_path = self.labels_dir / rel.with_suffix(".txt")

        image = Image.open(img_path).convert("RGB")
        boxes = _read_yolo_label(lbl_path, allow_missing=False)

        if self.geometric_aug is not None:
            image, boxes = self.geometric_aug(image, boxes)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image_tensor = self.transform(image)
        target_tensor = torch.from_numpy(boxes)
        return index, image_tensor, target_tensor


class YoloDualStrongDataset(Dataset):
    """Unlabeled detection dataset returning 1 weak + 2 strong views.

    Follows UniMatch V2: both strong views are independently sampled
    from the same weakly-augmented image through the strong augmentation pool.
    """

    def __init__(self, images_dir, labels_dir=None, geometric_aug=None, strong_aug=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.transform = _make_image_transform()
        self.geometric_aug = geometric_aug
        self.strong_aug = strong_aug
        self.image_paths = [p for p in discover_images(self.images_dir)
                           if "checkpoint" not in p.name]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        if self.labels_dir is not None:
            rel = img_path.relative_to(self.images_dir)
            lbl_path = self.labels_dir / rel.with_suffix(".txt")
        else:
            lbl_path = image_to_label_path(img_path)
        image = Image.open(img_path).convert("RGB")
        boxes = _read_yolo_label(lbl_path, allow_missing=True)

        # Geometric augmentation (shared across all views)
        if self.geometric_aug is not None:
            image, boxes = self.geometric_aug(image, boxes)

        target_tensor = torch.from_numpy(boxes)

        # Weak view: geometric aug only
        img_weak = self.transform(image)

        # Strong view 1: independent random strong augmentation
        if self.strong_aug is not None:
            img_s1_pil, _ = self.strong_aug(image, None)
        else:
            img_s1_pil = image
        img_strong1 = self.transform(img_s1_pil)

        # Strong view 2: independent random strong augmentation
        if self.strong_aug is not None:
            img_s2_pil, _ = self.strong_aug(image, None)
        else:
            img_s2_pil = image
        img_strong2 = self.transform(img_s2_pil)

        return index, img_weak, img_strong1, img_strong2, target_tensor


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_labeled(batch):
    indices = [b[0] for b in batch]
    images = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return indices, images, targets


def collate_unlabeled_dual(batch):
    indices = [b[0] for b in batch]
    weak = torch.stack([b[1] for b in batch])
    s1 = torch.stack([b[2] for b in batch])
    s2 = torch.stack([b[3] for b in batch])
    targets = [b[4] for b in batch]
    return indices, weak, s1, s2, targets


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_unimatch_yolo_loaders(cfg: dict, seed: int = 0):
    """Build labeled, unlabeled, and validation data loaders.

    Config keys under 'data':
        root, labeled_images, labeled_labels, unlabeled_images,
        pseudo_labels, val_images, val_labels, names
    Config keys under 'train':
        imgsz, batch, workers
    """
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train", {})
    root = Path(data_cfg["root"])

    imgsz = int(train_cfg.get("imgsz", 640))
    batch_size = int(train_cfg.get("batch", 4))
    workers = int(train_cfg.get("workers", 4))

    # Geometric augmentation (iMAS style: Resize -> Flip -> Crop)
    geo_aug = AugCompose([
        AugResize(base_size=[imgsz, imgsz], rand_resize=[0.5, 2.0]),
        RandomFlip(prob=0.5),
        AugCrop(crop_size=[imgsz, imgsz], crop_type="rand"),
    ])

    # Strong augmentation pool (UniMatch V2 paper-faithful: color-only).
    strong_aug = UniMatchStrongAug()

    # Pseudo labels dir (optional)
    pseudo_dir = data_cfg.get("pseudo_labels")
    pseudo_dir = (root / pseudo_dir) if pseudo_dir else None

    labeled_ds = YoloLabeledDataset(
        images_dir=root / data_cfg["labeled_images"],
        labels_dir=root / data_cfg["labeled_labels"],
        geometric_aug=geo_aug,
        color_jitter=bool(train_cfg.get("labeled_color_jitter", False)),
    )
    unlabeled_ds = YoloDualStrongDataset(
        images_dir=root / data_cfg["unlabeled_images"],
        labels_dir=pseudo_dir,
        geometric_aug=geo_aug,
        strong_aug=strong_aug,
    )
    val_ds = YoloLabeledDataset(
        images_dir=root / data_cfg["val_images"],
        labels_dir=root / data_cfg["val_labels"],
        geometric_aug=None,
        color_jitter=False,
    )

    # Oversample labeled to match unlabeled (same as iMAS original)
    n_labeled = len(labeled_ds)
    n_unlabeled = len(unlabeled_ds)
    if n_labeled < n_unlabeled:
        num_repeat = math.ceil(n_unlabeled / n_labeled)
        full_idx = list(range(n_labeled)) * num_repeat
        oversampled_idx = _rng.sample(full_idx, n_unlabeled)
        labeled_ds = Subset(labeled_ds, oversampled_idx)
        print(f"[DATA] Oversample labeled {n_labeled} -> {len(labeled_ds)} to match unlabeled {n_unlabeled}")

    loader_l = DataLoader(
        labeled_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=False, drop_last=True, worker_init_fn=seed_worker,
        collate_fn=collate_labeled,
    )
    loader_u = DataLoader(
        unlabeled_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=False, drop_last=True, worker_init_fn=seed_worker,
        collate_fn=collate_unlabeled_dual,
    )
    loader_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=False, drop_last=False, worker_init_fn=seed_worker,
        collate_fn=collate_labeled,
    )
    return loader_l, loader_u, loader_val
