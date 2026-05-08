from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from imas.yolo.common import ensure_dir, load_yaml, save_yaml
from imas.yolo.ultralytics_bridge import get_yolo_class


def _nested_get(data: dict, *keys, default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_data_cfg(cfg: dict) -> dict:
    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict) and data_cfg:
        return data_cfg
    bridge_cfg = cfg.get("yolo_data")
    if isinstance(bridge_cfg, dict) and bridge_cfg:
        return bridge_cfg
    raise ValueError(
        "Missing YOLO data config. Provide either 'data' or 'yolo_data' with keys: "
        "root, train_images, val_images, names."
    )


def _build_runtime_cfg(cfg: dict, args) -> dict:
    data_cfg = _resolve_data_cfg(cfg)

    model_cfg = cfg.get("model", {})
    trainer_cfg = cfg.get("trainer", {})
    optim_cfg = cfg.get("optim", {})

    output_root = Path(_nested_get(cfg, "project", "output_root", default="./runs/yolo_det/sup"))

    epochs = int(model_cfg.get("epochs", trainer_cfg.get("epochs", 100)))
    imgsz = int(model_cfg.get("imgsz", _nested_get(cfg, "train", "imgsz", default=640)))
    batch = int(model_cfg.get("batch", _nested_get(cfg, "train", "batch", default=16)))
    workers = int(model_cfg.get("workers", _nested_get(cfg, "train", "workers", default=8)))
    device = model_cfg.get("device", _nested_get(cfg, "train", "device", default=0))

    # Optimizer compatibility: modern optim.* or legacy trainer.optimizer.kwargs
    lr0 = float(optim_cfg.get("lr0", _nested_get(trainer_cfg, "optimizer", "kwargs", "lr", default=0.001)))
    weight_decay = float(
        optim_cfg.get("weight_decay", _nested_get(trainer_cfg, "optimizer", "kwargs", "weight_decay", default=0.0005))
    )
    momentum = float(_nested_get(trainer_cfg, "optimizer", "kwargs", "momentum", default=0.937))

    # LR scheduler compatibility from legacy poly mode.
    lr_mode = str(_nested_get(trainer_cfg, "lr_scheduler", "mode", default="cosine")).lower()
    lrf = float(optim_cfg.get("lrf", 0.01 if lr_mode in {"poly", "cosine"} else 0.1))
    warmup_epochs = float(optim_cfg.get("warmup_epochs", 3.0))
    cos_lr = bool(lr_mode == "cosine")

    runtime = {
        "output_root": output_root,
        "data_root": Path(data_cfg["root"]),
        "train_images": str(data_cfg["train_images"]),
        "val_images": str(data_cfg["val_images"]),
        "train_labels": str(data_cfg.get("train_labels", "")),
        "val_labels": str(data_cfg.get("val_labels", "")),
        "names": data_cfg["names"],
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "save_period": int(_nested_get(cfg, "save", "save_period", default=-1)),
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "warmup_epochs": warmup_epochs,
        "cos_lr": cos_lr,
        "hsv_h": float(_nested_get(cfg, "aug", "hsv_h", default=0.015)),
        "hsv_s": float(_nested_get(cfg, "aug", "hsv_s", default=0.7)),
        "hsv_v": float(_nested_get(cfg, "aug", "hsv_v", default=0.4)),
        "fliplr": float(_nested_get(cfg, "aug", "fliplr", default=0.5)),
        "mosaic": float(_nested_get(cfg, "aug", "mosaic", default=1.0)),
        "model_pt": str(args.model or model_cfg.get("init_pt", "")),
    }

    if not runtime["model_pt"]:
        raise ValueError("Missing model checkpoint. Set model.init_pt in config or pass --model.")

    return runtime


def _build_dataset_yaml(rt: dict, run_dir: Path) -> Path:
    dataset_yaml = run_dir / "dataset_sup.yaml"
    save_yaml(
        {
            "path": str(rt["data_root"]),
            "train": str(rt["train_images"]),
            "val": str(rt["val_images"]),
            "names": rt["names"],
        },
        dataset_yaml,
    )
    return dataset_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO supervised training (full compatibility mode)")
    parser.add_argument("--config", type=str, default="./exps/yolo_det/config_sup.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Optional override for model checkpoint path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rt = _build_runtime_cfg(cfg, args)

    run_id = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = rt["output_root"]
    run_dir = output_root / run_id
    latest_dir = output_root / "latest"
    ensure_dir(run_dir)
    ensure_dir(latest_dir)

    dataset_yaml = _build_dataset_yaml(rt, run_dir)

    yolo_cls = get_yolo_class()
    model = yolo_cls(rt["model_pt"])

    train_kwargs = {
        "data": str(dataset_yaml),
        "imgsz": rt["imgsz"],
        "epochs": rt["epochs"],
        "batch": rt["batch"],
        "device": rt["device"],
        "workers": rt["workers"],
        "project": str(output_root),
        "name": run_id,
        "exist_ok": True,
        "save": True,
        "save_period": rt["save_period"],
        "lr0": rt["lr0"],
        "lrf": rt["lrf"],
        "weight_decay": rt["weight_decay"],
        "momentum": rt["momentum"],
        "warmup_epochs": rt["warmup_epochs"],
        "cos_lr": rt["cos_lr"],
        "hsv_h": rt["hsv_h"],
        "hsv_s": rt["hsv_s"],
        "hsv_v": rt["hsv_v"],
        "fliplr": rt["fliplr"],
        "mosaic": rt["mosaic"],
    }
    if args.seed is not None:
        train_kwargs["seed"] = int(args.seed)

    result = model.train(**train_kwargs)
    train_run_dir = Path(model.trainer.save_dir)
    weights_dir = train_run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        src = weights_dir / name
        if src.exists():
            shutil.copy2(src, train_run_dir / name)
            shutil.copy2(src, latest_dir / name)

    save_yaml(cfg, train_run_dir / "resolved_config.yaml")
    shutil.copy2(train_run_dir / "resolved_config.yaml", latest_dir / "resolved_config.yaml")

    summary = {
        "config": str(Path(args.config).resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "run_dir": str(train_run_dir.resolve()),
        "latest_dir": str(latest_dir.resolve()),
        "runtime": {
            "epochs": rt["epochs"],
            "imgsz": rt["imgsz"],
            "batch": rt["batch"],
            "device": rt["device"],
            "lr0": rt["lr0"],
            "lrf": rt["lrf"],
            "weight_decay": rt["weight_decay"],
            "momentum": rt["momentum"],
            "cos_lr": rt["cos_lr"],
        },
        "result_type": type(result).__name__,
    }
    with open(train_run_dir / "sup_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    shutil.copy2(train_run_dir / "sup_summary.json", latest_dir / "sup_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
