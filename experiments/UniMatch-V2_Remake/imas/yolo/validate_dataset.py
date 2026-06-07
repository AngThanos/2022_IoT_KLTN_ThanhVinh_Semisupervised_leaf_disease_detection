from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imas.yolo.common import BANANA_ROOT, discover_images, load_class_names, relative_to_banana, write_csv


def parse_label_file(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}: expected 5 columns, found {len(parts)}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return np.asarray(rows, dtype=np.float32)


def check_pair(image_path: Path, label_path: Path, num_classes: int) -> tuple[bool, str, int]:
    if not image_path.exists():
        return False, f"missing image: {image_path}", 0
    if not label_path.exists():
        return False, f"missing label: {label_path}", 0

    labels = parse_label_file(label_path)
    if labels.size == 0:
        return True, "empty label file", 0

    if labels.shape[1] != 5:
        return False, f"invalid label shape: {labels.shape}", 0

    cls = labels[:, 0]
    box = labels[:, 1:]
    if np.any(cls < 0) or np.any(cls >= num_classes):
        return False, f"class id out of range in {label_path}", int(len(labels))
    if np.any(box < -1e-6) or np.any(box > 1.0 + 1e-6):
        return False, f"normalized box out of range in {label_path}", int(len(labels))
    if np.any(box[:, 2] <= 0) or np.any(box[:, 3] <= 0):
        return False, f"non-positive width/height in {label_path}", int(len(labels))
    return True, "ok", int(len(labels))


def build_expected_pairs(root: Path):
    labeled_images = discover_images(root / "train" / "labeled" / "images")
    val_images = discover_images(root / "val" / "images")
    unlabeled_images = discover_images(root / "train" / "unlabeled" / "images")
    return labeled_images, val_images, unlabeled_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate banana_data YOLO structure")
    parser.add_argument("--root", type=str, default=str(BANANA_ROOT))
    parser.add_argument("--output", type=str, default="./runs/yolo_det/data_check/latest")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(root / "meta")
    num_classes = len(class_names) if class_names else 1

    labeled_images, val_images, unlabeled_images = build_expected_pairs(root)
    rows = []
    failures = []
    total_boxes = 0

    for split, images, label_root, require_labels in (
        ("train_labeled", labeled_images, root / "train" / "labeled" / "labels", True),
        ("val", val_images, root / "val" / "labels", True),
        ("train_unlabeled", unlabeled_images, None, False),
    ):
        for image_path in images:
            if require_labels:
                image_base = root / "train" / "labeled" / "images" if split == "train_labeled" else root / "val" / "images"
                rel = image_path.relative_to(image_base)
                label_path = label_root / rel.with_suffix(".txt")
                ok, status, box_count = check_pair(image_path, label_path, num_classes)
                total_boxes += box_count
                if not ok:
                    failures.append(status)
            else:
                rel = image_path.relative_to(root / "train" / "unlabeled" / "images")
                label_path = root / "train" / "unlabeled" / "labels_pseudo" / rel.with_suffix(".txt")
                status = "image only"
                box_count = 0
            rows.append(
                {
                    "split": split,
                    "relative_path": str(relative_to_banana(image_path)).replace("\\", "/"),
                    "has_label": int(require_labels),
                    "label_path": str(relative_to_banana(label_path)).replace("\\", "/") if label_path else "",
                    "status": status,
                }
            )

    labeled_keys = {str(p.relative_to(root / "train" / "labeled" / "images")).replace("\\", "/") for p in labeled_images}
    val_keys = {str(p.relative_to(root / "val" / "images")).replace("\\", "/") for p in val_images}
    unlabeled_keys = {str(p.relative_to(root / "train" / "unlabeled" / "images")).replace("\\", "/") for p in unlabeled_images}
    overlap = (labeled_keys & val_keys) | (labeled_keys & unlabeled_keys) | (val_keys & unlabeled_keys)
    if overlap:
        failures.append(f"split overlap: {sorted(list(overlap))[:20]}")

    write_csv(rows, output_dir / "dataset_audit.csv", ["split", "relative_path", "has_label", "label_path", "status"])
    summary = {
        "root": str(root),
        "num_classes": num_classes,
        "labeled_images": len(labeled_images),
        "val_images": len(val_images),
        "unlabeled_images": len(unlabeled_images),
        "total_boxes_in_labeled_and_val": total_boxes,
        "failures": failures,
        "ok": len(failures) == 0,
    }
    with open(output_dir / "dataset_audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
