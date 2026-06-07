from __future__ import annotations

import argparse
from pathlib import Path

from imas.yolo.common import BANANA_ROOT, discover_images, relative_to_banana, write_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Build split manifest for banana_data")
    parser.add_argument("--root", type=str, default=str(BANANA_ROOT))
    parser.add_argument("--output", type=str, default=str(BANANA_ROOT / "meta" / "split_manifest.csv"))
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for split, image_dir, has_label in (
        ("train_labeled", root / "train" / "labeled" / "images", True),
        ("train_unlabeled", root / "train" / "unlabeled" / "images", False),
        ("val", root / "val" / "images", True),
    ):
        for image_path in discover_images(image_dir):
            label_path = ""
            if has_label:
                rel = image_path.relative_to(image_dir)
                label_file = image_dir.parent / "labels" / rel.with_suffix(".txt")
                label_path = str(relative_to_banana(label_file)).replace("\\", "/")
            rows.append(
                {
                    "split": split,
                    "relative_path": str(relative_to_banana(image_path)).replace("\\", "/"),
                    "has_label": int(has_label),
                    "label_path": label_path,
                    "status": "ready" if has_label else "unlabeled",
                }
            )

    write_csv(rows, args.output, ["split", "relative_path", "has_label", "label_path", "status"])
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
