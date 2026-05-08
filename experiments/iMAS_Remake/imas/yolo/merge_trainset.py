from __future__ import annotations

import argparse
import json
from pathlib import Path

from imas.yolo.common import BANANA_ROOT, discover_images, ensure_dir, load_yaml, save_yaml, try_hardlink_or_copy, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge labeled and pseudo-labeled YOLO data into a staged trainset")
    parser.add_argument("--config", type=str, default="./exps/yolo_det/config_semi.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["data"]["root"])
    run_root = Path(cfg["project"]["output_root"]) / "merge" / "latest"
    ensure_dir(run_root)

    stage_root = run_root / "dataset_stage"
    labeled_images = discover_images(root / cfg["data"]["labeled_images"])
    unlabeled_images = discover_images(root / cfg["data"]["unlabeled_images"])
    val_images = discover_images(root / cfg["data"]["val_images"])
    train_images_dir = stage_root / "train" / "images"
    train_labels_dir = stage_root / "train" / "labels"
    val_images_dir = stage_root / "val" / "images"
    val_labels_dir = stage_root / "val" / "labels"
    for path in (train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
        ensure_dir(path)

    for image_path in labeled_images:
        rel = image_path.relative_to(root / cfg["data"]["labeled_images"])
        try_hardlink_or_copy(image_path, train_images_dir / "labeled" / rel)
        label_path = root / cfg["data"]["labeled_labels"] / rel.with_suffix(".txt")
        if label_path.exists():
            try_hardlink_or_copy(label_path, train_labels_dir / "labeled" / rel.with_suffix(".txt"))

    for image_path in unlabeled_images:
        rel = image_path.relative_to(root / cfg["data"]["unlabeled_images"])
        try_hardlink_or_copy(image_path, train_images_dir / "unlabeled" / rel)
        pseudo_label = root / cfg["data"]["pseudo_labels"] / rel.with_suffix(".txt")
        if pseudo_label.exists():
            try_hardlink_or_copy(pseudo_label, train_labels_dir / "unlabeled" / rel.with_suffix(".txt"))

    for image_path in val_images:
        rel = image_path.relative_to(root / cfg["data"]["val_images"])
        try_hardlink_or_copy(image_path, val_images_dir / rel)
        label_path = root / cfg["data"]["val_labels"] / rel.with_suffix(".txt")
        if label_path.exists():
            try_hardlink_or_copy(label_path, val_labels_dir / rel.with_suffix(".txt"))

    dataset_yaml = stage_root / "dataset.yaml"
    save_yaml(
        {
            "path": str(stage_root),
            "train": "train/images",
            "val": "val/images",
            "names": cfg["data"]["names"],
        },
        dataset_yaml,
    )
    summary = {
        "config": str(Path(args.config).resolve()),
        "stage_root": str(stage_root.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_images": len(labeled_images) + len(unlabeled_images),
        "val_images": len(val_images),
    }
    with open(run_root / "merge_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(
        [
            {"split": "train_labeled", "count": len(labeled_images)},
            {"split": "train_unlabeled", "count": len(unlabeled_images)},
            {"split": "val", "count": len(val_images)},
        ],
        run_root / "merge_stats.csv",
        ["split", "count"],
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())