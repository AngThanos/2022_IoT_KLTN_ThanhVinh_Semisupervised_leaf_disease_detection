from __future__ import annotations

import argparse
import copy
import json
import math
import random as _random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from imas.yolo.common import discover_images, ensure_dir, load_yaml, save_yaml
from imas.dataset.yolodata import build_yolo_semi_loader
from imas.yolo.hardness import DetectionHardnessConfig, compute_detection_hardness
from imas.yolo.ultralytics_bridge import get_yolo_class


class _StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _setup_run_logging(run_dir: Path, latest_dir: Path) -> dict[str, Any]:
    log_dir = run_dir / "log"
    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_path = log_dir / f"seg_{ts}.log"
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

    state = {
        "stdout": sys.stdout,
        "stderr": sys.stderr,
        "fp": log_fp,
        "log_path": log_path,
        "latest_dir": latest_dir,
    }

    sys.stdout = _StreamTee(sys.stdout, log_fp)
    sys.stderr = _StreamTee(sys.stderr, log_fp)
    print(f"[LOG] File logging enabled: {log_path}")
    return state


def _finalize_run_logging(state: dict[str, Any] | None):
    if not state:
        return
    try:
        sys.stdout = state["stdout"]
        sys.stderr = state["stderr"]
    finally:
        fp = state.get("fp")
        if fp:
            fp.flush()
            fp.close()
    log_path: Path = state["log_path"]
    latest_dir: Path = state["latest_dir"]
    ensure_dir(latest_dir)
    shutil.copy2(log_path, latest_dir / "seg_latest.log")


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
        "root, labeled_images, labeled_labels, unlabeled_images, pseudo_labels, val_images, val_labels, names."
    )


def _build_runtime_cfg(cfg: dict, args) -> dict:
    data_cfg = _resolve_data_cfg(cfg)

    train_cfg = cfg.get("train", {})
    trainer_cfg = cfg.get("trainer", {})
    unsup_cfg = trainer_cfg.get("unsupervised", {})
    pseudo_cfg = cfg.get("pseudo", {})
    loss_cfg = cfg.get("loss", {})
    optim_cfg = cfg.get("optim", {})
    hard_cfg = cfg.get("hardness", {})
    semi_cfg = cfg.get("semi", {})

    legacy_hardness_aware = _nested_get(cfg, "dataset", "train", "hardness_aware", default={})

    output_root = Path(_nested_get(cfg, "project", "output_root", default="./runs/yolo_det/semi"))

    total_epochs = int(train_cfg.get("epochs", trainer_cfg.get("epochs", 120)))
    sup_only_epoch = int(trainer_cfg.get("sup_only_epoch", train_cfg.get("sup_only_epoch", 0)))
    refresh_every = int(pseudo_cfg.get("refresh_every_epochs", max(1, total_epochs)))
    refresh_every = max(1, refresh_every)

    threshold_legacy = float(unsup_cfg.get("threshold", 0.7))
    conf_start = float(pseudo_cfg.get("conf_start", threshold_legacy))
    conf_end = float(pseudo_cfg.get("conf_end", threshold_legacy))

    lambda_u = float(loss_cfg.get("lambda_u", unsup_cfg.get("loss_weight", 1.0)))
    lambda_u_schedule = str(loss_cfg.get("lambda_u_schedule", "ramp")).lower()

    # Hardness behavior: keep all pseudo labels then reweight/sampling in train stage.
    hard_enabled = bool(hard_cfg.get("enabled", bool(legacy_hardness_aware)))
    hard_train_strategy = str(hard_cfg.get("train_strategy", "easy_focus")).lower()
    hard_gamma = float(hard_cfg.get("gamma", 1.0))
    hard_min_weight = float(hard_cfg.get("min_weight", 0.1))
    hard_max_weight = float(hard_cfg.get("max_weight", 1.5))

    rt = {
        "output_root": output_root,
        "imgsz": int(train_cfg.get("imgsz", 640)),
        "batch": int(train_cfg.get("batch", 16)),
        "workers": int(train_cfg.get("workers", 8)),
        "device": train_cfg.get("device", 0),
        "total_epochs": total_epochs,
        "sup_only_epoch": sup_only_epoch,
        "conf_start": conf_start,
        "conf_end": conf_end,
        "iou_thres_nms": float(pseudo_cfg.get("iou_thres_nms", 0.6)),
        "max_det": int(pseudo_cfg.get("max_det", 300)),
        "lambda_u": lambda_u,
        "lambda_u_schedule": lambda_u_schedule,
        # FIX 2: Lower default LR from 0.001 to 0.0001 to preserve pretrained features.
        "lr0": float(optim_cfg.get("lr0", _nested_get(trainer_cfg, "optimizer", "kwargs", "lr", default=0.0001))),
        "weight_decay": float(
            optim_cfg.get("weight_decay", _nested_get(trainer_cfg, "optimizer", "kwargs", "weight_decay", default=0.0005))
        ),
        "momentum": float(_nested_get(trainer_cfg, "optimizer", "kwargs", "momentum", default=0.937)),
        "hard_enabled": hard_enabled,
        "hard_iou": float(hard_cfg.get("iou_match_thres", 0.5)),
        "hard_alpha": float(hard_cfg.get("alpha_unmatched", 0.5)),
        "hard_weight_map": str(hard_cfg.get("weight_map", "linear")),
        "hard_train_strategy": hard_train_strategy,
        "hard_gamma": hard_gamma,
        "hard_min_weight": hard_min_weight,
        "hard_max_weight": hard_max_weight,
        "hard_online_weighted_loss": bool(hard_cfg.get("online_weighted_loss", True)),
        "student_pt": str(args.student or _nested_get(cfg, "student", "init_pt", default="")),
        "teacher_pt": str(args.teacher or _nested_get(cfg, "teacher", "pt", default="")),
        "ema_decay": float(_nested_get(cfg, "teacher", "ema_decay", default=_nested_get(cfg, "net", "ema_decay", default=0.996))),
        # iMAS Contribution 2 & 3: adaptive augmentation and CutMix
        "adaptive_aug": bool(_nested_get(cfg, "semi", "adaptive_aug", default=False)),
        "adaptive_cutmix": bool(_nested_get(cfg, "semi", "adaptive_cutmix", default=False)),
        "cutmix_scale": list(_nested_get(cfg, "semi", "cutmix_scale", default=[0.3, 0.33])),
        # Resume training
        "resume": str(_nested_get(cfg, "train", "resume", default="")),
    }

    if not rt["student_pt"]:
        raise ValueError("Missing student checkpoint. Set student.init_pt in config or pass --student.")
    if not rt["teacher_pt"]:
        rt["teacher_pt"] = rt["student_pt"]

    return rt


def _clone_preds(preds):
    if isinstance(preds, torch.Tensor):
        return preds.clone()
    if isinstance(preds, list):
        return [_clone_preds(x) for x in preds]
    if isinstance(preds, tuple):
        return tuple(_clone_preds(x) for x in preds)
    if isinstance(preds, dict):
        return {k: _clone_preds(v) for k, v in preds.items()}
    return preds


def _extract_loss_scalar(loss_out) -> torch.Tensor:
    if isinstance(loss_out, torch.Tensor):
        return loss_out if loss_out.ndim == 0 else loss_out.sum()
    if isinstance(loss_out, (list, tuple)) and len(loss_out) > 0:
        first = loss_out[0]
        if isinstance(first, torch.Tensor):
            return first if first.ndim == 0 else first.sum()
    raise RuntimeError("Unable to extract scalar loss from YOLO loss output.")


def _safe_yolo_loss(yolo_model, batch: dict, preds) -> torch.Tensor:
    if hasattr(yolo_model, "init_criterion"):
        criterion = yolo_model.init_criterion()
        # Some Ultralytics builds expose criterion.hyp as dict, while loss code expects
        # attribute access (e.g. self.hyp.box). Normalize once before calling loss().
        if hasattr(criterion, "hyp"):
            if isinstance(criterion.hyp, dict):
                hyp_dict = dict(criterion.hyp)
            else:
                hyp_dict = dict(vars(criterion.hyp))

            # Keep this resilient across minor Ultralytics builds where some fields
            # may be missing from criterion.hyp.
            hyp_defaults = {
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "pose": 12.0,
                "kobj": 1.0,
                "label_smoothing": 0.0,
                "fl_gamma": 0.0,
            }
            for key, value in hyp_defaults.items():
                hyp_dict.setdefault(key, value)

            criterion.hyp = SimpleNamespace(**hyp_dict)
        yolo_model.criterion = criterion
    return _extract_loss_scalar(yolo_model.loss(batch, _clone_preds(preds)))


def _prediction_tensor(preds):
    if isinstance(preds, torch.Tensor):
        return preds
    if isinstance(preds, (list, tuple)) and len(preds) > 0:
        if isinstance(preds[0], torch.Tensor):
            return preds[0]
    return preds


def _apply_nms_from_preds(preds, conf_thres: float, iou_thres: float, max_det: int):
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    pred = _prediction_tensor(preds)
    if not isinstance(pred, torch.Tensor):
        return []
    return non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)


def _det_to_cls_xywhn_conf(det: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    if det is None or det.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    x1 = det[:, 0].clamp(0, img_w)
    y1 = det[:, 1].clamp(0, img_h)
    x2 = det[:, 2].clamp(0, img_w)
    y2 = det[:, 3].clamp(0, img_h)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    xywhn = torch.stack(
        [
            cx / float(img_w),
            cy / float(img_h),
            w / float(img_w),
            h / float(img_h),
        ],
        dim=1,
    )
    cls = det[:, 5:6]
    conf = det[:, 4:5]
    arr = torch.cat([cls, xywhn, conf], dim=1).detach().cpu().numpy().astype(np.float32)
    return arr


def _build_supervised_batch(images: torch.Tensor, targets: list[torch.Tensor], device: torch.device) -> dict:
    batch_idx_list = []
    cls_list = []
    bbox_list = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        t = t.to(device)
        batch_idx_list.append(torch.full((t.shape[0],), i, dtype=torch.long, device=device))
        cls_list.append(t[:, 0:1].float())
        bbox_list.append(t[:, 1:5].float())

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        bboxes = torch.cat(bbox_list, dim=0)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.long, device=device)
        cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)

    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }


def _build_pseudo_batch_from_teacher(
    teacher_preds,
    images: torch.Tensor,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> tuple[dict, float, float, float]:
    detections = _apply_nms_from_preds(teacher_preds, conf_thres=0.001, iou_thres=iou_thres, max_det=max_det)
    if not detections:
        empty = _build_supervised_batch(images, [], images.device)
        return empty, 0.0, 0.0, 0.0

    bsz, _, img_h, img_w = images.shape
    batch_idx_list = []
    cls_list = []
    bbox_list = []
    conf_list = []
    selected_images = 0

    for bi, det in enumerate(detections):
        if det is None or det.shape[0] == 0:
            continue
        det = det[det[:, 4] >= float(conf_thres)]
        if det.shape[0] == 0:
            continue

        selected_images += 1
        arr = _det_to_cls_xywhn_conf(det, img_w=img_w, img_h=img_h)
        t = torch.from_numpy(arr).to(images.device)
        batch_idx_list.append(torch.full((t.shape[0],), bi, dtype=torch.long, device=images.device))
        cls_list.append(t[:, 0:1].float())
        bbox_list.append(t[:, 1:5].float())
        conf_list.append(t[:, 4:5].float())  # confidence scores

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        bboxes = torch.cat(bbox_list, dim=0)
        mean_conf = float(torch.cat(conf_list, dim=0).mean().item())
    else:
        batch_idx = torch.zeros((0,), dtype=torch.long, device=images.device)
        cls = torch.zeros((0, 1), dtype=torch.float32, device=images.device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=images.device)
        mean_conf = 0.0

    pseudo_batch = {
        "img": images,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
    }
    pseudo_ratio = float(selected_images) / float(max(1, bsz))
    pseudo_boxes_per_img = float(batch_idx.numel()) / float(max(1, bsz))
    return pseudo_batch, pseudo_ratio, pseudo_boxes_per_img, mean_conf


def _teacher_dets_per_image(teacher_preds, iou_thres: float, max_det: int, img_w: int, img_h: int) -> list[np.ndarray]:
    """Get per-image teacher detections as list of (N, 6) arrays: [cls, cx, cy, w, h, conf]."""
    raw_dets = _apply_nms_from_preds(teacher_preds, conf_thres=0.001, iou_thres=iou_thres, max_det=max_det)
    result = []
    for det in raw_dets:
        arr = _det_to_cls_xywhn_conf(det, img_w=img_w, img_h=img_h)
        result.append(arr)
    return result


def _per_image_confs(per_img_dets: list[np.ndarray]) -> list[float]:
    """Mean teacher confidence per image."""
    return [float(d[:, 5].mean()) if d.shape[0] > 0 else 0.0 for d in per_img_dets]


def _build_pseudo_batch_from_dets(
    per_img_dets: list[np.ndarray],
    images: torch.Tensor,
    conf_thres: float,
    device: torch.device,
) -> tuple[dict, float, float, float]:
    """Build YOLO batch from pre-computed per-image detections with conf filtering."""
    bsz = images.shape[0]
    batch_idx_list, cls_list, bbox_list, conf_list = [], [], [], []
    selected = 0
    for bi, arr in enumerate(per_img_dets):
        if arr.shape[0] == 0:
            continue
        mask = arr[:, 5] >= conf_thres
        arr_f = arr[mask]
        if arr_f.shape[0] == 0:
            continue
        selected += 1
        t = torch.from_numpy(arr_f).to(device)
        batch_idx_list.append(torch.full((t.shape[0],), bi, dtype=torch.long, device=device))
        cls_list.append(t[:, 0:1])
        bbox_list.append(t[:, 1:5])
        conf_list.append(t[:, 5:6])

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list)
        cls = torch.cat(cls_list)
        bboxes = torch.cat(bbox_list)
        mean_conf = float(torch.cat(conf_list).mean())
    else:
        batch_idx = torch.zeros((0,), dtype=torch.long, device=device)
        cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        mean_conf = 0.0

    pseudo_batch = {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}
    pseudo_ratio = float(selected) / float(max(1, bsz))
    pbox = float(batch_idx.numel()) / float(max(1, bsz))
    return pseudo_batch, pseudo_ratio, pbox, mean_conf


def _detection_cutmix(
    images: torch.Tensor,
    per_img_dets: list[np.ndarray],
    hardness_values: list[float],
    scale: tuple[float, float] = (0.3, 0.33),
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """CutMix for detection with hard-easy pairing (iMAS Contribution 3).

    Hard samples get patches from easy samples and vice versa.
    Bounding boxes are reassigned based on center location.
    """
    B, C, H, W = images.shape
    h_arr = np.array(hardness_values[:B])
    # Hard-easy pairing: ascending ↔ descending
    asc = np.argsort(h_arr)
    desc = asc[::-1].copy()
    pair_map = {int(asc[i]): int(desc[i]) for i in range(B)}

    v = _random.uniform(min(scale), max(scale))
    cut_w = int(W * v)
    cut_h = int(H * v)

    mixed_images = images.clone()
    mixed_dets = []

    for i in range(B):
        j = pair_map[i]
        cx = _random.randint(max(1, W // 16), max(2, W - 1))
        cy = _random.randint(max(1, H // 16), max(2, H - 1))
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(W, cx + cut_w // 2), min(H, cy + cut_h // 2)

        mixed_images[i, :, y1:y2, x1:x2] = images[j, :, y1:y2, x1:x2]

        nx1, ny1 = x1 / max(1, W), y1 / max(1, H)
        nx2, ny2 = x2 / max(1, W), y2 / max(1, H)

        # Keep boxes from image i with center OUTSIDE cut region
        dets_i = per_img_dets[i]
        if dets_i.shape[0] > 0:
            ci_x, ci_y = dets_i[:, 1], dets_i[:, 2]
            outside = ~((ci_x >= nx1) & (ci_x <= nx2) & (ci_y >= ny1) & (ci_y <= ny2))
            keep_i = dets_i[outside]
        else:
            keep_i = np.zeros((0, 6), dtype=np.float32)

        # Add boxes from image j with center INSIDE cut region
        dets_j = per_img_dets[j]
        if dets_j.shape[0] > 0:
            cj_x, cj_y = dets_j[:, 1], dets_j[:, 2]
            inside = (cj_x >= nx1) & (cj_x <= nx2) & (cj_y >= ny1) & (cj_y <= ny2)
            add_j = dets_j[inside]
        else:
            add_j = np.zeros((0, 6), dtype=np.float32)

        parts = [p for p in (keep_i, add_j) if p.shape[0] > 0]
        mixed_dets.append(np.concatenate(parts, axis=0) if parts else np.zeros((0, 6), dtype=np.float32))

    return mixed_images, mixed_dets


def _sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


def _save_online_ckpt(
    student_model, teacher_model, epoch: int, out_pt: Path, imgsz: int = 640,
    optimizer=None, scheduler=None, best_val_map50: float = -1.0, epochs_no_improve: int = 0,
) -> None:
    ensure_dir(out_pt.parent)
    train_args = {"task": "detect", "imgsz": imgsz}
    ckpt = {
        "epoch": int(epoch),
        "model": copy.deepcopy(student_model).eval(),
        "ema": copy.deepcopy(teacher_model).eval(),
        "updates": int(epoch),
        "train_args": train_args,
        "best_val_map50": float(best_val_map50),
        "epochs_no_improve": int(epochs_no_improve),
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state"] = scheduler.state_dict()
    torch.save(ckpt, out_pt)


def _load_resume(ckpt_path: Path, student_model, teacher_model, optimizer, scheduler, device) -> tuple[int, float, int]:
    """Load resume checkpoint. Returns (start_epoch, best_val_map50, epochs_no_improve)."""
    print(f"[RESUME] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        saved_student = ckpt["model"].float()
        student_model.load_state_dict(saved_student.state_dict())
    if "ema" in ckpt:
        saved_teacher = ckpt["ema"].float()
        teacher_model.load_state_dict(saved_teacher.state_dict())
    if "optimizer_state" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    last_epoch = ckpt.get("epoch", -1)
    best_val = ckpt.get("best_val_map50", -1.0)
    epochs_no_imp = ckpt.get("epochs_no_improve", 0)
    print(f"[RESUME] Resuming from epoch {last_epoch + 1}, best_val_mAP50={best_val:.4f}")
    return last_epoch + 1, best_val, epochs_no_imp


def _train_full_online(rt: dict, cfg: dict, args, run_dir: Path, latest_dir: Path) -> tuple[Path, dict]:
    yolo_cls = get_yolo_class()
    student = yolo_cls(rt["student_pt"])
    teacher = yolo_cls(rt["teacher_pt"])
    student_model = student.model
    teacher_model = teacher.model

    if isinstance(rt["device"], str) and rt["device"].lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(rt['device'])}" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    # Some exported checkpoints come with all params frozen; re-enable grads for student training.
    for p in student_model.parameters():
        p.requires_grad = True

    for p in teacher_model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        for t_params, s_params in zip(teacher_model.parameters(), student_model.parameters()):
            t_params.copy_(s_params)

    train_loader_sup, train_loader_unsup, _ = build_yolo_semi_loader("train", cfg, seed=args.seed or 0)
    # iMAS original (train_semi.py line 353-375):
    #   assert len(loader_l) == len(loader_u), "imbalance!"
    #   for step in range(len(loader_l)):
    # Labeled dataset đã được oversample trong build_yolo_semi_loader cho bằng unlabeled.
    assert abs(len(train_loader_sup) - len(train_loader_unsup)) <= 1, (
        f"loader length mismatch: sup={len(train_loader_sup)}, unsup={len(train_loader_unsup)}. "
        f"Labeled dataset should be oversampled to match unlabeled in build_yolo_semi_loader."
    )
    steps_per_epoch = len(train_loader_sup)

    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=float(rt["lr0"]),
        momentum=float(rt["momentum"]),
        weight_decay=float(rt["weight_decay"]),
    )

    total_steps = max(1, rt["total_epochs"] * steps_per_epoch)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda it: 0.5 * (1.0 + math.cos(math.pi * min(1.0, it / total_steps))))

    online_dir = run_dir / "online"
    ensure_dir(online_dir)
    losses = []
    best_val_map50 = -1.0
    best_pt = online_dir / "best.pt"
    last_pt = online_dir / "last.pt"

    # Build a temporary val YAML for Ultralytics validator
    data_cfg = _resolve_data_cfg(cfg)
    _val_yaml_path = online_dir / "_val_data.yaml"
    _val_root = Path(data_cfg.get("root", "./data/banana_data")).resolve()
    _val_yaml_content = {
        "path": str(_val_root),
        "train": str(data_cfg["val_images"]),  # Ultralytics requires 'train' key; reuse val path
        "val": str(data_cfg["val_images"]),
        "names": data_cfg.get("names", {0: "early_sigatoka", 1: "late_sigatoka"}),
    }
    save_yaml(_val_yaml_content, _val_yaml_path)
    # Validate every N epochs (save time); always validate last epoch.
    _val_every = max(1, int(_nested_get(cfg, "save", "val_every", default=5)))
    _early_stop_patience = int(_nested_get(cfg, "save", "early_stop_patience", default=0))
    _epochs_no_improve = 0

    sup_only_epoch = int(rt["sup_only_epoch"])

    # Resume from checkpoint if requested
    start_epoch = 0
    resume_flag = rt.get("resume", "")
    if resume_flag:
        if resume_flag == "auto":
            resume_path = last_pt
        else:
            resume_path = Path(resume_flag)
        if resume_path.exists():
            start_epoch, best_val_map50, _epochs_no_improve = _load_resume(
                resume_path, student_model, teacher_model, optimizer, scheduler, device
            )
        else:
            print(f"[RESUME] Checkpoint not found: {resume_path}, starting from scratch.")

    for epoch in range(start_epoch, rt["total_epochs"]):
        if hasattr(train_loader_sup, "sampler") and hasattr(train_loader_sup.sampler, "set_epoch"):
            train_loader_sup.sampler.set_epoch(epoch)
        if hasattr(train_loader_unsup, "sampler") and hasattr(train_loader_unsup.sampler, "set_epoch"):
            train_loader_unsup.sampler.set_epoch(epoch)

        student_model.train()
        # FIX 1: Freeze BN running stats — prevent 74-sample overwrite of pretrained BN.
        for m in student_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        teacher_model.eval()

        sup_meter = 0.0
        uns_meter = 0.0
        pseudo_ratio_meter = 0.0
        pbox_meter = 0.0
        hardness_meter = 0.0
        hard_weight_meter = 0.0
        hard_steps = 0
        step_time_meter = 0.0

        print_freq = max(1, steps_per_epoch // 8)
        print_freq_steps = set(i * print_freq for i in range(1, 8))
        print_freq_steps.add(max(0, steps_per_epoch - 1))

        print(
            f"Epoch/Start [{epoch + 1}/{rt['total_epochs']}], "
            f"steps={steps_per_epoch}, batch={rt['batch']}, imgsz={rt['imgsz']}"
        )

        sup_iter = iter(train_loader_sup)
        uns_iter = iter(train_loader_unsup)

        for step in range(steps_per_epoch):
            step_start = time.time()
            try:
                _, image_l, target_l = next(sup_iter)
            except StopIteration:
                sup_iter = iter(train_loader_sup)
                _, image_l, target_l = next(sup_iter)
            try:
                unsup_batch = next(uns_iter)
            except StopIteration:
                uns_iter = iter(train_loader_unsup)
                unsup_batch = next(uns_iter)
            # Dataloader returns (index, weak_tensor, strong_tensor, targets)
            _, image_u_weak, image_u_strong, _ = unsup_batch
            image_l = image_l.to(device, non_blocking=True)
            image_u_weak = image_u_weak.to(device, non_blocking=True)
            image_u_strong = image_u_strong.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Keep supervised/unsupervised loss forwards on training path.
            student_model.train()
            for m in student_model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
            batch_l = _build_supervised_batch(image_l, target_l, device)
            pred_l = student_model(batch_l["img"])
            sup_loss = _safe_yolo_loss(student_model, batch_l, pred_l)

            unsup_loss = torch.tensor(0.0, device=device)
            pseudo_ratio = 0.0
            pseudo_boxes_per_img = 0.0
            mean_pseudo_conf = 0.0
            step_hardness_avg = 0.5
            step_hard_weight = 1.0

            if epoch >= sup_only_epoch:
                with torch.no_grad():
                    teacher_preds = teacher_model(image_u_weak)

                _, _, img_h_px, img_w_px = image_u_weak.shape
                lambda_eff = float(rt["lambda_u"])
                conf_thres = _scheduled_conf(float(rt["conf_start"]), float(rt["conf_end"]), epoch, max(1, rt["total_epochs"]))

                # Get all teacher detections (low conf for NMS), then filter for pseudo labels
                per_img_dets = _teacher_dets_per_image(
                    teacher_preds, float(rt["iou_thres_nms"]), int(rt["max_det"]), img_w_px, img_h_px
                )

                # --- iMAS: Hardness from FILTERED detections (conf >= threshold) ---
                # Using raw dets (conf=0.001) gave mean_conf~0.04 → hardness~0.96 → CutMix always triggers → NaN
                per_img_conf_filtered = []
                for d in per_img_dets:
                    if d.shape[0] > 0:
                        mask = d[:, 5] >= conf_thres
                        d_f = d[mask]
                        per_img_conf_filtered.append(float(d_f[:, 5].mean()) if d_f.shape[0] > 0 else 0.0)
                    else:
                        per_img_conf_filtered.append(0.0)
                hardness_values = [1.0 - c for c in per_img_conf_filtered]
                step_hardness_avg = float(np.mean(hardness_values)) if hardness_values else 0.5

                # iMAS Contribution 2: Adaptive augmentation intensity
                # Hard images → more strong aug, easy images → more weak aug
                if rt["adaptive_aug"]:
                    gamma = torch.tensor(hardness_values, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
                    image_u_strong = image_u_strong * gamma + image_u_weak * (1.0 - gamma)

                # iMAS Contribution 3: Adaptive CutMix with hard-easy pairing
                if rt["adaptive_cutmix"]:
                    # Clamp trigger prob to [0, 0.5] — iMAS original uses 0.5 as default trigger
                    trigger_prob = min(step_hardness_avg, 0.5)
                    if _random.random() < trigger_prob:
                        image_u_strong, per_img_dets = _detection_cutmix(
                            image_u_strong, per_img_dets, hardness_values, scale=tuple(rt["cutmix_scale"])
                        )

                # Build pseudo batch from per-image detections with conf filtering
                pseudo_batch, pseudo_ratio, pseudo_boxes_per_img, mean_pseudo_conf = _build_pseudo_batch_from_dets(
                    per_img_dets, image_u_strong, conf_thres, device
                )

                if pseudo_batch["batch_idx"].numel() > 0:
                    student_model.train()
                    for m in student_model.modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                            m.eval()
                    pred_u = student_model(pseudo_batch["img"])
                    pseudo_loss = _safe_yolo_loss(student_model, pseudo_batch, pred_u)
                    # iMAS Contribution 1: Confidence-weighted loss (easy_focus = 1 - hardness ≈ conf)
                    conf_weight = max(0.05, mean_pseudo_conf)
                    unsup_loss = pseudo_loss * float(lambda_eff) * conf_weight
                    # Safety clamp: prevent gradient explosion from noisy pseudo labels
                    max_unsup = float(sup_loss.detach().item()) * 2.0
                    if float(unsup_loss.detach().item()) > max_unsup and max_unsup > 0:
                        unsup_loss = unsup_loss * (max_unsup / float(unsup_loss.detach().item()))

            loss = sup_loss + unsup_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if epoch >= sup_only_epoch:
                    global_step = (epoch - sup_only_epoch) * steps_per_epoch + step
                    ema_decay = min(1.0 - 1.0 / float(global_step + 1), float(rt["ema_decay"]))
                else:
                    ema_decay = 0.0
                for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
                    p_t.copy_(p_t * ema_decay + p_s.detach() * (1.0 - ema_decay))
                # iMAS original: only EMA-update float buffers (BN stats);
                # copy non-float buffers (e.g. num_batches_tracked) directly.
                for b_s, b_t in zip(student_model.buffers(), teacher_model.buffers()):
                    if b_t.dtype in (torch.float32, torch.float64, torch.float16):
                        b_t.copy_(b_t * ema_decay + b_s.detach() * (1.0 - ema_decay))
                    else:
                        b_t.copy_(b_s)

            sup_meter += float(sup_loss.detach().item())
            uns_meter += float(unsup_loss.detach().item())
            pseudo_ratio_meter += float(pseudo_ratio)
            pbox_meter += float(pseudo_boxes_per_img)
            conf_meter = 0.0
            if epoch >= sup_only_epoch:
                hardness_meter += float(step_hardness_avg)
                hard_weight_meter += float(step_hard_weight)
                conf_meter += float(mean_pseudo_conf)
                hard_steps += 1

            step_time_meter += max(0.0, time.time() - step_start)
            if step in print_freq_steps:
                done = step + 1
                avg_sup_now = sup_meter / float(max(1, done))
                avg_uns_now = uns_meter / float(max(1, done))
                avg_pseudo_now = pseudo_ratio_meter / float(max(1, done))
                avg_time_now = step_time_meter / float(max(1, done))
                avg_hard_now = hardness_meter / float(max(1, hard_steps))
                lr_now = optimizer.param_groups[0].get("lr", 0.0)
                print(
                    f"Epoch/Iter [{rt['total_epochs']}:{epoch:3d}/{step:3d}]  "
                    f"AvgHard:{avg_hard_now:.3f}  "
                    f"ConfW:{float(mean_pseudo_conf):.3f}  "
                    f"Sup:{float(sup_loss.detach().item()):.3f}({avg_sup_now:.3f})  "
                    f"Uns:{float(unsup_loss.detach().item()):.3f}({avg_uns_now:.3f})  "
                    f"Pseudo:{float(pseudo_ratio):.3f}({avg_pseudo_now:.3f})  "
                    f"Time:{avg_time_now:.2f}  LR:{lr_now:.5f}"
                )

        avg_sup = sup_meter / float(max(1, steps_per_epoch))
        avg_uns = uns_meter / float(max(1, steps_per_epoch))
        avg_total = avg_sup + avg_uns
        losses.append(
            {
                "epoch": int(epoch),
                "sup_loss": float(avg_sup),
                "unsup_loss": float(avg_uns),
                "total_loss": float(avg_total),
                "pseudo_ratio": float(pseudo_ratio_meter / float(max(1, steps_per_epoch))),
                "pseudo_boxes_per_img": float(pbox_meter / float(max(1, steps_per_epoch))),
                "avg_hardness": float(hardness_meter / float(max(1, hard_steps))),
                "avg_hardness_weight": float(hard_weight_meter / float(max(1, hard_steps))),
            }
        )

        _save_online_ckpt(student_model, teacher_model, epoch, last_pt, imgsz=rt["imgsz"],
                         optimizer=optimizer, scheduler=scheduler,
                         best_val_map50=best_val_map50, epochs_no_improve=_epochs_no_improve)

        # Validation: run val on teacher (EMA) model every _val_every epochs + last epoch.
        val_map50 = -1.0
        is_last_epoch = (epoch == rt["total_epochs"] - 1)
        if (epoch + 1) % _val_every == 0 or is_last_epoch:
            try:
                _tmp_val_pt = online_dir / "_tmp_val.pt"
                _save_online_ckpt(student_model, teacher_model, epoch, _tmp_val_pt, imgsz=rt["imgsz"])
                val_yolo = yolo_cls(str(_tmp_val_pt))
                val_results = val_yolo.val(
                    data=str(_val_yaml_path),
                    imgsz=int(rt["imgsz"]),
                    batch=int(rt["batch"]),
                    conf=0.1,
                    iou=0.1,
                    verbose=False,
                )
                val_map50 = float(val_results.results_dict.get("metrics/mAP50(B)", -1.0))
                if _tmp_val_pt.exists():
                    _tmp_val_pt.unlink()
            except Exception as e:
                print(f"[VAL] Warning: validation failed at epoch {epoch+1}: {e}")
                val_map50 = -1.0

        if val_map50 > best_val_map50:
            best_val_map50 = val_map50
            shutil.copy2(last_pt, best_pt)
            _epochs_no_improve = 0
        elif val_map50 >= 0:
            _epochs_no_improve += 1

        print(
            f"Epoch/Done [{epoch + 1}/{rt['total_epochs']}]  "
            f"Sup:{avg_sup:.4f}  Uns:{avg_uns:.4f}  Total:{avg_total:.4f}  "
            f"Pseudo:{pseudo_ratio_meter / float(max(1, steps_per_epoch)):.4f}  "
            f"Val_mAP50:{val_map50:.4f}  Best_mAP50:{best_val_map50:.4f}"
        )

        # Early stopping: only activate after sup_only phase
        if _early_stop_patience > 0 and epoch >= sup_only_epoch and _epochs_no_improve >= _early_stop_patience:
            print(f"[EARLY STOP] No val mAP50 improvement for {_epochs_no_improve} epochs. Stopping at epoch {epoch + 1}.")
            break

    shutil.copy2(best_pt, latest_dir / "best.pt")
    shutil.copy2(last_pt, latest_dir / "last.pt")

    summary = {
        "mode": "online",
        "online_type": "full_step",
        "epochs": int(rt["total_epochs"]),
        "steps_per_epoch": int(steps_per_epoch),
        "best_val_map50": float(best_val_map50),
        "best_pt": str(best_pt.resolve()),
        "last_pt": str(last_pt.resolve()),
        "history": losses,
    }
    with open(online_dir / "online_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    shutil.copy2(online_dir / "online_summary.json", latest_dir / "online_summary.json")
    return online_dir, summary


def _scheduled_conf(conf_start: float, conf_end: float, round_idx: int, num_rounds: int) -> float:
    if num_rounds <= 1:
        return conf_end
    t = float(round_idx) / float(num_rounds - 1)
    return conf_start + (conf_end - conf_start) * t


def _hardness_to_weight(hardness: float, rt: dict) -> float:
    h = float(np.clip(hardness, 0.0, 1.0))
    gamma = max(1e-6, float(rt["hard_gamma"]))
    mode = rt["hard_train_strategy"]

    if mode == "hard_focus":
        weight = pow(h, gamma)
    elif mode == "balanced":
        weight = 1.0
    else:
        # default: easy_focus (consistent with pseudo quality weighting)
        weight = pow(1.0 - h, gamma)

    weight = float(np.clip(weight, rt["hard_min_weight"], rt["hard_max_weight"]))
    return weight


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO semi-supervised training (iMAS online)")
    parser.add_argument("--config", type=str, default="./exps/yolo_det/config_semi.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--student", type=str, default=None)
    parser.add_argument("--teacher", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rt = _build_runtime_cfg(cfg, args)

    run_id = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = rt["output_root"]
    run_dir = output_root / run_id
    latest_dir = output_root / "latest"
    ensure_dir(run_dir)
    ensure_dir(latest_dir)
    log_state = _setup_run_logging(run_dir, latest_dir)

    try:
        online_dir, online_summary = _train_full_online(rt, cfg, args, run_dir, latest_dir)
        save_yaml(cfg, run_dir / "resolved_config.yaml")
        shutil.copy2(run_dir / "resolved_config.yaml", latest_dir / "resolved_config.yaml")
        summary = {
            "config": str(Path(args.config).resolve()),
            "run_root": str(run_dir.resolve()),
            "latest_dir": str(latest_dir.resolve()),
            "online_dir": str(online_dir.resolve()),
            "ema_decay": rt["ema_decay"],
            "online": online_summary,
        }
        with open(run_dir / "semi_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        shutil.copy2(run_dir / "semi_summary.json", latest_dir / "semi_summary.json")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _finalize_run_logging(log_state)


if __name__ == "__main__":
    raise SystemExit(main())
