from __future__ import annotations

import argparse
import json
from pathlib import Path

from imas.yolo.common import BANANA_ROOT, load_yaml, save_yaml, write_csv
from imas.yolo.ultralytics_bridge import get_yolo_class


def resolve_dataset_yaml(cfg: dict) -> Path:
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("root", BANANA_ROOT))
    candidate = root / "meta" / "dataset_supervised.yaml"
    return candidate if candidate.exists() else BANANA_ROOT / "meta" / "dataset_supervised.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on banana_data val set")
    parser.add_argument("--config", type=str, default="./exps/yolo_det/config_sup.yaml")
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_yaml = resolve_dataset_yaml(cfg)
    run_id = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    yolo_cls = get_yolo_class()
    model = yolo_cls(args.weights)

    metrics = model.val(
        data=str(dataset_yaml),
        imgsz=int(cfg["model"]["imgsz"]),
        device=cfg["model"]["device"],
        batch=int(cfg["model"]["batch"]),
        verbose=False,
    )

    out_dir = Path(cfg["project"]["output_root"]) / "eval" / run_id
    latest_dir = Path(cfg["project"]["output_root"]) / "eval" / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = getattr(metrics, "results_dict", {}) if metrics is not None else {}
    payload = {"weights": args.weights, "dataset": str(dataset_yaml), "metrics": metrics_dict}
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with open(latest_dir / "eval_report.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    if metrics_dict:
        csv_rows = [{"metric": key, "value": value} for key, value in metrics_dict.items()]
        write_csv(csv_rows, out_dir / "eval_metrics.csv", ["metric", "value"])
        write_csv(csv_rows, latest_dir / "eval_metrics.csv", ["metric", "value"])
    save_yaml(cfg, out_dir / "resolved_config.yaml")
    save_yaml(cfg, latest_dir / "resolved_config.yaml")
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
