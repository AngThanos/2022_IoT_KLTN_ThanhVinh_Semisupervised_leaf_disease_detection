from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from ..yolo.common import BANANA_ROOT, discover_images, image_to_label_path
from .augmentations import (
    Compose as AugCompose,
    Resize as AugResize,
    RandomFlip,
    Crop as AugCrop,
    ColorJitter as AugColorJitter,
    strong_img_aug,
)

# Avoid dataloader worker crashes on minor JPEG truncation/corruption.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    seed += worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_image_transform(mean=None, std=None):
    # Ultralytics YOLO expects [0,1] input (just /255). No ImageNet normalization.
    # OLD: mean = mean or [0.485, 0.456, 0.406]; std = std or [0.229, 0.224, 0.225]
    # OLD: return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ops = [transforms.ToTensor()]
    if mean is not None and std is not None:
        ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def _read_yolo_label(label_path: Path, allow_missing: bool = False) -> np.ndarray:
    if not label_path.exists():
        if allow_missing:
            return np.zeros((0, 5), dtype=np.float32)
        raise FileNotFoundError(str(label_path))

    rows = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
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


def _collate_detection_batch(batch):
    indices = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch], dim=0)
    targets = [item[2] for item in batch]
    return indices, images, targets


def _collate_unlabeled_dual_view_batch(batch):
    indices = [item[0] for item in batch]
    images_weak = torch.stack([item[1] for item in batch], dim=0)
    images_strong = torch.stack([item[2] for item in batch], dim=0)
    targets = [item[3] for item in batch]
    return indices, images_weak, images_strong, targets


def yolo_collate_fn(batch):
    """Compatibility alias for older code paths."""
    return _collate_detection_batch(batch)


def _make_sampler(dataset, shuffle: bool):
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None, mean=None, std=None,
                 allow_missing_labels=False, return_dual_view=False,
                 geometric_aug=None, strong_aug=None, color_jitter=False):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) if labels_dir is not None else None
        self.allow_missing_labels = allow_missing_labels
        self.return_dual_view = bool(return_dual_view)
        self.transform = _make_image_transform(mean=mean, std=std)
        self.geometric_aug = geometric_aug    # AugCompose([RandomFlip, ...]) — (image, label) → (image, label)
        self.strong_aug = strong_aug          # strong_img_aug — (image) → (image, label)
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02) if color_jitter else None
        self.image_paths = discover_images(self.images_dir)

    def __len__(self):
        return len(self.image_paths)

    def _resolve_label_path(self, image_path: Path) -> Path:
        if self.labels_dir is not None:
            rel = image_path.relative_to(self.images_dir)
            return self.labels_dir / rel.with_suffix(".txt")
        return image_to_label_path(image_path)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self._resolve_label_path(image_path)
        with open(image_path, "rb") as handle:
            image = Image.open(handle).convert("RGB")
        boxes = _read_yolo_label(label_path, allow_missing=self.allow_missing_labels)

        # Geometric augmentation (modifies both image and boxes)
        if self.geometric_aug is not None:
            image, boxes = self.geometric_aug(image, boxes)

        # Color jitter for labeled data (image only)
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        target_tensor = torch.from_numpy(boxes)

        if self.return_dual_view:
            # Weak view: geometrically augmented PIL → tensor (for teacher)
            image_weak = self.transform(image)
            # Strong view: geometric + strong_img_aug → tensor (for student)
            if self.strong_aug is not None:
                image_strong_pil, _ = self.strong_aug(image, None)
                image_strong = self.transform(image_strong_pil)
            else:
                image_strong = self.transform(image)
            return index, image_weak, image_strong, target_tensor

        image_tensor = self.transform(image)
        return index, image_tensor, target_tensor


def _get_data_cfg(cfg):
    return cfg.get("data", cfg.get("dataset", {}))


def _get_train_cfg(cfg):
    return cfg.get("train", cfg.get("model", {}))


def build_yolo_loader(split, all_cfg, seed=0):
    data_cfg = _get_data_cfg(all_cfg)
    train_cfg = _get_train_cfg(all_cfg)
    root = Path(data_cfg.get("root", BANANA_ROOT))

    if split == "train":
        images_dir = root / data_cfg["train_images"]
        labels_dir = root / data_cfg["train_labels"]
    else:
        images_dir = root / data_cfg["val_images"]
        labels_dir = root / data_cfg["val_labels"]

    dataset = YoloDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        allow_missing_labels=(split != "train"),
    )
    batch_size = int(train_cfg.get("batch", train_cfg.get("batch_size", 1)))
    workers = int(train_cfg.get("workers", 4))
    sampler = _make_sampler(dataset, shuffle=(split == "train"))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and split == "train",
        num_workers=workers,
        pin_memory=False,
        drop_last=(split == "train"),
        worker_init_fn=seed_worker,
        collate_fn=_collate_detection_batch,
    )
    return loader


def build_yololoader(split, all_cfg, seed=0):
    """Compatibility alias for older caller naming."""
    return build_yolo_loader(split, all_cfg, seed=seed)


def build_yolo_semi_loader(split, all_cfg, seed=0):
    data_cfg = _get_data_cfg(all_cfg)
    train_cfg = _get_train_cfg(all_cfg)
    semi_cfg = all_cfg.get("semi", {})
    root = Path(data_cfg.get("root", BANANA_ROOT))

    pseudo_labels = data_cfg.get("pseudo_labels", data_cfg.get("unlabeled_labels"))
    pseudo_labels = (root / pseudo_labels) if pseudo_labels else None

    # Geometric augmentation (iMAS original: Resize → Flip → Crop)
    # iMAS VOC config: rand_resize=[0.5, 2.0], resize_base_size=[512,512], flip=True, crop=rand 512x512
    # Adapted for detection: base_size=imgsz (1024), crop back to imgsz after scale.
    imgsz = int(train_cfg.get("imgsz", 1024))
    geo_aug = AugCompose([
        AugResize(base_size=[imgsz, imgsz], rand_resize=[0.5, 2.0]),
        RandomFlip(prob=0.5),
        AugCrop(crop_size=[imgsz, imgsz], crop_type="rand"),
    ])

    # Strong augmentation for unlabeled: 2 random ops from pool (like iMAS original)
    strong_aug = strong_img_aug(num_augs=2)

    labeled_dataset = YoloDataset(
        images_dir=root / data_cfg["labeled_images"],
        labels_dir=root / data_cfg["labeled_labels"],
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        allow_missing_labels=False,
        geometric_aug=geo_aug,
        color_jitter=True,
    )
    unlabeled_dataset = YoloDataset(
        images_dir=root / data_cfg["unlabeled_images"],
        labels_dir=pseudo_labels,
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        allow_missing_labels=True,
        return_dual_view=True,
        geometric_aug=geo_aug,
        strong_aug=strong_aug,
    )
    val_dataset = YoloDataset(
        images_dir=root / data_cfg["val_images"],
        labels_dir=root / data_cfg["val_labels"],
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
        allow_missing_labels=True,
    )

    # Oversample labeled to match unlabeled count (iMAS original: len(loader_l)==len(loader_u))
    # iMAS gốc (pascal_voc.py line 50-51):
    #   num_repeat = math.ceil(n_sup / len(self.list_sample))
    #   self.list_sample = self.list_sample * num_repeat
    #   self.list_sample_new = random.sample(self.list_sample, n_sup)
    # → Repeat labeled list đủ lớn, rồi random.sample đúng bằng len(unlabeled).
    # Mỗi epoch DataLoader shuffle lại → thứ tự khác nhau → giảm overfit.
    import math, random as _rng
    n_labeled_orig = len(labeled_dataset)
    n_unlabeled = len(unlabeled_dataset)
    if n_labeled_orig < n_unlabeled:
        num_repeat = math.ceil(n_unlabeled / n_labeled_orig)
        full_indices = list(range(n_labeled_orig)) * num_repeat          # [0,1,...,73, 0,1,...,73, ...]
        oversampled_indices = _rng.sample(full_indices, n_unlabeled)     # random sample đúng = n_unlabeled
        labeled_dataset = Subset(labeled_dataset, oversampled_indices)
        print(f"[DATA] Oversample labeled {n_labeled_orig} -> {len(labeled_dataset)} "
              f"(repeat x{num_repeat}, sample {n_unlabeled}) to match unlabeled")

    batch_size = int(train_cfg.get("batch", train_cfg.get("batch_size", 1)))
    workers = int(train_cfg.get("workers", 4))

    labeled_sampler = _make_sampler(labeled_dataset, shuffle=True)
    unlabeled_sampler = _make_sampler(unlabeled_dataset, shuffle=True)
    val_sampler = _make_sampler(val_dataset, shuffle=False)

    loader_sup = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        sampler=labeled_sampler,
        shuffle=labeled_sampler is None,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        collate_fn=_collate_detection_batch,
    )
    loader_unsup = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        sampler=unlabeled_sampler,
        shuffle=unlabeled_sampler is None,
        num_workers=workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        collate_fn=_collate_unlabeled_dual_view_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=seed_worker,
        collate_fn=_collate_detection_batch,
    )
    return loader_sup, loader_unsup, val_loader


def build_semi_yololoader(split, all_cfg, seed=0):
    """Compatibility alias for older caller naming."""
    return build_yolo_semi_loader(split, all_cfg, seed=seed)
