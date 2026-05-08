from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from imas.yolo.common import BANANA_ROOT, discover_images, ensure_dir, load_yaml, read_lines, write_csv
from imas.yolo.ultralytics_bridge import get_yolo_class


def boxes_to_yolo_lines(result, class_whitelist: set[int] | None = None) -> tuple[list[str], dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], {"box_count": 0, "conf_mean": 0.0, "class_hist": {}}

    xywhn = boxes.xywhn.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    lines = []
    kept_conf = []
    class_hist = Counter()
    for (x, y, w, h), c, k in zip(xywhn, conf, cls):
        if class_whitelist and int(k) not in class_whitelist:
            continue
        lines.append(f"{int(k)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept_conf.append(float(c))
        class_hist[int(k)] += 1

    stats = {
        "box_count": len(lines),
        "conf_mean": float(np.mean(kept_conf)) if kept_conf else 0.0,
        "class_hist": dict(class_hist),
    }
    return lines, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo labels for banana_data unlabeled set")
    parser.add_argument("--config", type=str, default="./exps/yolo_det/config_pseudo.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    root = Path(data_cfg.get("root", BANANA_ROOT))
    run_id = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = root / data_cfg["unlabeled_images"]
    out_dir = root / data_cfg["pseudo_labels_out"]
    ensure_dir(out_dir)
    run_root = Path(cfg.get("project", {}).get("output_root", "./runs/yolo_det/pseudo"))
    stats_dir = run_root / run_id
    ensure_dir(stats_dir)
    latest_dir = run_root / "latest"
    ensure_dir(latest_dir)

    class_names = read_lines(root / "meta" / "classes.txt")
    class_whitelist = set(data_cfg.get("class_whitelist", [])) if data_cfg.get("class_whitelist") else None
    yolo_cls = get_yolo_class()
    model = yolo_cls(cfg["teacher"]["pt"])

    rows = []
    for image_path in discover_images(image_dir):
        rel = image_path.relative_to(image_dir)
        results = model.predict(
            source=str(image_path),
            imgsz=int(cfg["teacher"]["imgsz"]),
            device=cfg["teacher"]["device"],
            conf=float(cfg["infer"]["conf_thres"]),
            iou=float(cfg["infer"]["iou_thres_nms"]),
            max_det=int(cfg["infer"]["max_det"]),
            verbose=False,
        )
        result = results[0]
        lines, stats = boxes_to_yolo_lines(result, class_whitelist=class_whitelist)
        min_boxes = int(cfg.get("filter", {}).get("min_boxes_per_image", 0))
        max_boxes = int(cfg.get("filter", {}).get("max_boxes_per_image", 0))
        if len(lines) < min_boxes:
            lines = []
            stats["box_count"] = 0
            stats["conf_mean"] = 0.0
            stats["class_hist"] = {}
        if max_boxes > 0 and len(lines) > max_boxes:
            lines = lines[:max_boxes]
            stats["box_count"] = len(lines)
        txt_path = out_dir / rel.with_suffix(".txt")
        ensure_dir(txt_path.parent)
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if lines:
                handle.write("\n")
        rows.append(
            {
                "image": str(image_path.relative_to(root)).replace("\\", "/"),
                "label": str(txt_path.relative_to(root)).replace("\\", "/"),
                "box_count": stats["box_count"],
                "conf_mean": stats["conf_mean"],
                "class_hist": json.dumps(stats["class_hist"]),
            }
        )

    if cfg.get("save", {}).get("export_stats_csv", True):
        write_csv(rows, stats_dir / "pseudo_stats.csv", ["image", "label", "box_count", "conf_mean", "class_hist"])
    summary = {
        "image_dir": str(image_dir),
        "output_dir": str(out_dir),
        "stats_dir": str(stats_dir),
        "latest_dir": str(latest_dir),
        "num_images": len(rows),
        "classes": class_names,
    }
    with open(stats_dir / "pseudo_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if cfg.get("save", {}).get("export_stats_csv", True):
        write_csv(rows, latest_dir / "pseudo_stats.csv", ["image", "label", "box_count", "conf_mean", "class_hist"])
    with open(latest_dir / "pseudo_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
