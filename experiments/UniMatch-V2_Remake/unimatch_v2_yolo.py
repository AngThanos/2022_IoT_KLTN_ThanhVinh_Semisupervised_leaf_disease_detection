"""
UniMatch V2 for YOLOv11 Object Detection
=========================================
Semi-supervised object detection following the UniMatch V2 algorithm
(arXiv:2410.10777 — Yang et al., TPAMI).

Core algorithm:
    1. EMA teacher produces pseudo labels on weakly-augmented images
    2. Two strongly-augmented views are generated from each weak image
    3. CutMix is applied independently to each strong view (batch-flip pairing)
    4. Complementary channel-wise Dropout is applied at YOLO backbone features
    5. Student learns from both streams; loss = (loss_l + loss_u) / 2
    6. EMA teacher is updated: γ = min(1 - 1/(iter+1), 0.996)

Data loading and model are imported from iMAS_Remake.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from imas.yolo.ultralytics_bridge import get_yolo_class
from imas.yolo.common import ensure_dir, load_yaml, save_yaml

from dataset.yolo_semi import build_unimatch_yolo_loaders


# ===================================================================
# Complementary Channel-Wise Dropout for YOLO
# (Section 3.3.2 of UniMatch V2)
# ===================================================================

class CompDropContext:
    """Manages complementary dropout masks for dual-stream forward passes.

    On the first stream, generates binomial masks at each backbone save point.
    On the second stream, reuses the complementary (1 - mask) masks.

    Following UniMatch V2 (DPT implementation):
        mask1 = binomial.sample((B, C)) * 2.0          → values 0 or 2
        mask2 = 2.0 - mask1                             → complementary
        For ~50% of batch items, masks are set to 1.0   → no dropout (stability)
    """

    def __init__(self, dropout_prob: float = 0.5):
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        self.dropout_prob = dropout_prob
        self.masks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.stream: int = 0  # 0 = first stream, 1 = second (complementary)
        self.enabled: bool = False

    def get_mask(self, layer_idx: int, B: int, C: int, device: torch.device) -> torch.Tensor:
        """Get or create dropout mask for a given layer."""
        if layer_idx not in self.masks:
            mask1 = self.binomial.sample((B, C)).to(device) * 2.0
            mask2 = 2.0 - mask1
            # Keep some batch items without dropout for stability
            num_kept = max(1, int(B * (1 - self.dropout_prob)))
            kept = torch.randperm(B)[:num_kept]
            mask1[kept, :] = 1.0
            mask2[kept, :] = 1.0
            self.masks[layer_idx] = (mask1, mask2)
        return self.masks[layer_idx][self.stream]

    def reset(self):
        """Clear all masks (call between iterations)."""
        self.masks.clear()


def _find_backbone_save_indices(det_model) -> set[int]:
    """Identify backbone layers whose outputs are saved for the neck.

    These are the layers where complementary dropout should be applied
    (encoder-decoder boundary, Section 3.3.2).
    """
    neck_start = None
    for i, m in enumerate(det_model.model):
        name = type(m).__name__
        if name in ("Upsample", "Concat"):
            neck_start = i
            break
    if neck_start is None:
        neck_start = len(det_model.model)
    return {s for s in det_model.save if s < neck_start}


def yolo_forward_comp_drop(
    det_model: nn.Module,
    x: torch.Tensor,
    backbone_saves: set[int],
    comp_ctx: CompDropContext | None = None,
) -> Any:
    """Run YOLO forward pass with optional complementary dropout.

    Replicates the inner loop of ultralytics DetectionModel._predict_once(),
    injecting complementary channel-wise dropout at backbone save points.
    """
    y: list[Any] = []
    for m in det_model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)

        # Apply complementary dropout at backbone feature save points
        if (
            comp_ctx is not None
            and comp_ctx.enabled
            and m.i in backbone_saves
            and isinstance(x, torch.Tensor)
        ):
            B, C, H, W = x.shape
            mask = comp_ctx.get_mask(m.i, B, C, x.device)
            x = x * mask.unsqueeze(-1).unsqueeze(-1)

        y.append(x if m.i in det_model.save else None)
    return x


# ===================================================================
# YOLO Loss Utilities (from iMAS_Remake)
# ===================================================================

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
    """Compute YOLO detection loss (box + cls + dfl)."""
    if hasattr(yolo_model, "init_criterion"):
        criterion = yolo_model.init_criterion()
        if hasattr(criterion, "hyp"):
            if isinstance(criterion.hyp, dict):
                hyp_dict = dict(criterion.hyp)
            else:
                hyp_dict = dict(vars(criterion.hyp))
            hyp_defaults = {
                "box": 7.5, "cls": 0.2, "dfl": 1.5,
                "pose": 12.0, "kobj": 1.0,
                "label_smoothing": 0.0, "fl_gamma": 0.0,
            }
            for k, v in hyp_defaults.items():
                hyp_dict.setdefault(k, v)
            criterion.hyp = SimpleNamespace(**hyp_dict)
        # Anti-collapse for BCE: pos_weight=4 amplifies the rare positive anchors
        # so they are not drowned out by ~10k negative anchors. Combined with
        # cls=0.2 above, this prevents the "predict-all-background" absorbing state.
        if hasattr(criterion, "bce") and isinstance(criterion.bce, torch.nn.BCEWithLogitsLoss) \
                and getattr(criterion.bce, "pos_weight", None) is None and hasattr(criterion, "nc"):
            dev = getattr(criterion, "device", next(yolo_model.parameters()).device)
            pos_w = torch.full((criterion.nc,), 4.0, device=dev)
            criterion.bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w)
        yolo_model.criterion = criterion
    return _extract_loss_scalar(yolo_model.loss(batch, _clone_preds(preds)))


def _prediction_tensor(preds):
    if isinstance(preds, torch.Tensor):
        return preds
    if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], torch.Tensor):
        return preds[0]
    return preds


# ===================================================================
# Pseudo Label Generation (from teacher detections)
# ===================================================================

def _apply_nms(preds, conf_thres: float, iou_thres: float, max_det: int):
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    pred = _prediction_tensor(preds)
    if not isinstance(pred, torch.Tensor):
        return []
    return non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)


def _det_to_array(det: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    """Convert NMS detections to [cls, cx_n, cy_n, w_n, h_n, conf] arrays."""
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
    arr = torch.stack([
        det[:, 5],               # cls
        cx / float(img_w),       # cx_norm
        cy / float(img_h),       # cy_norm
        w / float(img_w),        # w_norm
        h / float(img_h),        # h_norm
        det[:, 4],               # conf
    ], dim=1)
    return arr.detach().cpu().numpy().astype(np.float32)


def _teacher_per_image_dets(
    teacher_preds, iou_thres: float, max_det: int, img_w: int, img_h: int
) -> list[np.ndarray]:
    """Get teacher detections per image: list of (N, 6) arrays."""
    raw = _apply_nms(teacher_preds, conf_thres=0.001, iou_thres=iou_thres, max_det=max_det)
    return [_det_to_array(d, img_w, img_h) for d in raw]


def _build_yolo_batch(images: torch.Tensor, targets: list, device: torch.device) -> dict:
    """Build YOLO batch dict from images and per-image target tensors/arrays."""
    batch_idx_list, cls_list, bbox_list = [], [], []
    for i, t in enumerate(targets):
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        t = t.to(device)
        if t.numel() == 0:
            continue
        batch_idx_list.append(torch.full((t.shape[0],), i, dtype=torch.long, device=device))
        cls_list.append(t[:, 0:1].float())
        bbox_list.append(t[:, 1:5].float())
    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list)
        cls = torch.cat(cls_list)
        bboxes = torch.cat(bbox_list)
    else:
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)
        cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
    return {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


def _build_pseudo_batch(
    per_img_dets: list[np.ndarray],
    images: torch.Tensor,
    conf_thresh: float,
    device: torch.device,
) -> tuple[dict, float, float]:
    """Build YOLO batch from teacher detections, filtering by confidence.

    Returns: (batch_dict, pseudo_ratio, mean_conf)
    """
    B = images.shape[0]
    batch_idx_list, cls_list, bbox_list, conf_list = [], [], [], []
    selected = 0
    for bi, arr in enumerate(per_img_dets):
        if arr.shape[0] == 0:
            continue
        mask = arr[:, 5] >= conf_thresh
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
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)
        cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
        bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        mean_conf = 0.0
    batch = {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}
    return batch, float(selected) / max(1, B), mean_conf


# ===================================================================
# Detection CutMix (UniMatch V2 paper-faithful: obtain_cutmix_box)
# Mirrors UniMatch-V2_Origin/dataset/transform.py::obtain_cutmix_box:
#   - prob p=0.5 (50% of samples actually receive CutMix)
#   - size: uniform in [size_min, size_max] of image area
#   - aspect ratio: uniform in [ratio_1, ratio_2]
# Adapted to detection with batch-flip pairing.
# ===================================================================

def _obtain_cutmix_box(
    H: int,
    W: int,
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_1: float = 0.3,
    ratio_2: float = 1 / 0.3,
) -> tuple[int, int, int, int] | None:
    """Sample a CutMix region following Origin's `obtain_cutmix_box`.

    Returns (y1, x1, y2, x2) in pixels, or None when no CutMix is applied.
    """
    if random.random() > p:
        return None
    img_area = float(H * W)
    size = random.uniform(size_min, size_max) * img_area
    for _ in range(20):
        ratio = random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        if cutmix_w <= 0 or cutmix_h <= 0:
            continue
        if cutmix_w >= W or cutmix_h >= H:
            continue
        x1 = random.randint(0, W - cutmix_w)
        y1 = random.randint(0, H - cutmix_h)
        return y1, x1, y1 + cutmix_h, x1 + cutmix_w
    return None


def apply_detection_cutmix(
    images: torch.Tensor,
    per_img_dets: list[np.ndarray],
    cutmix_prob: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Apply CutMix to detection images with batch-flip pairing.

    Paper-faithful: each image independently receives CutMix with prob ``p``,
    with variable region size and aspect ratio (Origin obtain_cutmix_box).
    Boxes are reassigned by center location for swapped regions.
    """
    B, C, H, W = images.shape
    mixed = images.clone()
    flipped_images = images.flip(0)
    flipped_dets = per_img_dets[::-1]
    mixed_dets = []

    for i in range(B):
        box = _obtain_cutmix_box(H, W, p=cutmix_prob, size_min=size_min, size_max=size_max)
        if box is None:
            # No CutMix for this sample: keep original image and boxes
            mixed_dets.append(per_img_dets[i])
            continue
        y1, x1, y2, x2 = box
        mixed[i, :, y1:y2, x1:x2] = flipped_images[i, :, y1:y2, x1:x2]

        nx1, ny1 = x1 / max(1, W), y1 / max(1, H)
        nx2, ny2 = x2 / max(1, W), y2 / max(1, H)

        dets_i = per_img_dets[i]
        if dets_i.shape[0] > 0:
            cx, cy = dets_i[:, 1], dets_i[:, 2]
            outside = ~((cx >= nx1) & (cx <= nx2) & (cy >= ny1) & (cy <= ny2))
            keep_i = dets_i[outside]
        else:
            keep_i = np.zeros((0, 6), dtype=np.float32)

        dets_j = flipped_dets[i]
        if dets_j.shape[0] > 0:
            cx, cy = dets_j[:, 1], dets_j[:, 2]
            inside = (cx >= nx1) & (cx <= nx2) & (cy >= ny1) & (cy <= ny2)
            add_j = dets_j[inside]
        else:
            add_j = np.zeros((0, 6), dtype=np.float32)

        parts = [p for p in (keep_i, add_j) if p.shape[0] > 0]
        mixed_dets.append(np.concatenate(parts) if parts else np.zeros((0, 6), dtype=np.float32))

    return mixed, mixed_dets


# ===================================================================
# Training Loop
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UniMatch V2: Semi-Supervised Object Detection with YOLOv11"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--student", type=str, default=None, help="Student model checkpoint")
    parser.add_argument("--teacher", type=str, default=None, help="Teacher model checkpoint")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg["data"]

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Paths
    student_pt = args.student or cfg.get("student", {}).get("init_pt", "")
    teacher_pt = args.teacher or cfg.get("teacher", {}).get("pt", student_pt)
    if not student_pt:
        raise ValueError("Student checkpoint required: --student or student.init_pt in config")
    save_path = Path(args.save_path or cfg.get("project", {}).get("output_root", "./runs/unimatch_yolo"))
    ensure_dir(save_path)

    # Hyperparameters
    imgsz = int(train_cfg.get("imgsz", 640))
    batch_size = int(train_cfg.get("batch", 4))
    epochs = int(train_cfg.get("epochs", 50))
    lr_encoder = float(train_cfg.get("lr", 5e-6))
    # Pseudo-label confidence threshold (AugSeg-style schedule):
    #   - Static : only `conf_thresh` is set.
    #   - Dynamic: both `conf_thresh` (start) and `conf_thresh_end` (end) set;
    #              linearly interpolated across epochs.
    conf_thresh_start = float(cfg.get("conf_thresh", 0.5))
    conf_thresh_end = float(cfg.get("conf_thresh_end", conf_thresh_start))
    ema_decay_max = float(cfg.get("teacher", {}).get("ema_decay", 0.996))
    # EMA floor: protect fully-pretrained YOLO teacher from iter-0 wipe.
    ema_floor = float(cfg.get("teacher", {}).get("ema_floor", 0.996))
    # CutMix paper-faithful params (Origin obtain_cutmix_box).
    cutmix_prob = float(cfg.get("cutmix_prob", 0.5))
    cutmix_size_min = float(cfg.get("cutmix_size_min", 0.02))
    cutmix_size_max = float(cfg.get("cutmix_size_max", 0.4))
    iou_thres_nms = float(cfg.get("iou_thres_nms", 0.6))
    max_det = int(cfg.get("max_det", 300))
    val_every = int(cfg.get("val_every", 5))

    is_dynamic_conf = conf_thresh_start != conf_thresh_end
    if is_dynamic_conf:
        print(f"[CONFIG] conf_thresh: DYNAMIC linear {conf_thresh_start} -> {conf_thresh_end} over {epochs} epochs")
    else:
        print(f"[CONFIG] conf_thresh: STATIC {conf_thresh_start}")

    device_id = train_cfg.get("device", 0)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ==========================
    # Model setup
    # ==========================
    yolo_cls = get_yolo_class()
    student_yolo = yolo_cls(student_pt)
    teacher_yolo = yolo_cls(teacher_pt)

    student_model = student_yolo.model.to(device)
    teacher_model = teacher_yolo.model.to(device)

    # Enable gradients for student, freeze teacher
    for p in student_model.parameters():
        p.requires_grad = True
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Initialize teacher from student (exact copy)
    with torch.no_grad():
        for t_p, s_p in zip(teacher_model.parameters(), student_model.parameters()):
            t_p.copy_(s_p)

    # Find backbone save indices for complementary dropout
    backbone_saves = _find_backbone_save_indices(student_model)
    print(f"[MODEL] Backbone save indices for CompDrop: {backbone_saves}")

    # Complementary dropout context
    comp_ctx = CompDropContext(dropout_prob=0.5)

    # ==========================
    # Optimizer: SGD+momentum for YOLO CNN detection.
    # Paper UniMatch V2 uses AdamW for DPT/ViT semseg, but for YOLO11 CNN with
    # tiny labeled set (74 imgs) AdamW overfits val (val_72 same distribution as
    # labeled_74) while losing generalization. SGD with momentum 0.937 generalizes
    # better — matches iMAS / AD-MT / AugSeg setup which all beat the supervised
    # baseline on this dataset.
    # ==========================
    opt_name = str(train_cfg.get("optimizer", cfg.get("optimizer", "sgd"))).lower()
    wd = float(train_cfg.get("weight_decay", cfg.get("weight_decay", 0.0005)))
    momentum_v = float(train_cfg.get("momentum", cfg.get("momentum", 0.937)))
    if opt_name == "adamw":
        optimizer = AdamW(
            student_model.parameters(),
            lr=lr_encoder,
            betas=(0.9, 0.999),
            weight_decay=wd,
        )
    else:
        optimizer = SGD(
            student_model.parameters(),
            lr=lr_encoder,
            momentum=momentum_v,
            weight_decay=wd,
            nesterov=True,
        )
    print(f"[OPTIM] {opt_name.upper()}  lr={lr_encoder}  wd={wd}"
          + (f"  momentum={momentum_v}" if opt_name != "adamw" else ""))

    # ==========================
    # Data loaders
    # ==========================
    loader_l, loader_u, loader_val = build_unimatch_yolo_loaders(cfg, seed=args.seed)
    steps_per_epoch = min(len(loader_l), len(loader_u))
    total_iters = steps_per_epoch * epochs
    print(f"[DATA] Labeled: {len(loader_l.dataset)}, Unlabeled: {len(loader_u.dataset)}, "
          f"Val: {len(loader_val.dataset)}")
    print(f"[TRAIN] Epochs: {epochs}, Steps/epoch: {steps_per_epoch}, Total iters: {total_iters}")

    # Build validation YAML for ultralytics
    val_yaml_path = save_path / "_val_data.yaml"
    val_root = Path(data_cfg["root"]).resolve()
    save_yaml({
        "path": str(val_root),
        "train": str(data_cfg["val_images"]),
        "val": str(data_cfg["val_images"]),
        "names": data_cfg.get("names", {0: "object"}),
    }, val_yaml_path)

    # ==========================
    # Resume
    # ==========================
    start_epoch = 0
    best_val_map50 = -1.0
    latest_path = save_path / "latest.pth"
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        student_model.load_state_dict(ckpt["model"])
        teacher_model.load_state_dict(ckpt["model_ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_map50 = ckpt.get("best_val_map50", -1.0)
        print(f"[RESUME] From epoch {start_epoch}, best_mAP50={best_val_map50:.4f}")

    # ==========================
    # Training
    # ==========================
    history = []

    for epoch in range(start_epoch, epochs):
        # Compute conf_thresh for this epoch (linear interp, AugSeg pattern)
        if is_dynamic_conf and epochs > 1:
            t = epoch / float(epochs - 1)
            conf_thresh = conf_thresh_start + t * (conf_thresh_end - conf_thresh_start)
        else:
            conf_thresh = conf_thresh_start
        print(f"[EPOCH {epoch}] conf_thresh = {conf_thresh:.4f}")

        student_model.train()
        # Freeze BN to preserve pretrained statistics
        for m in student_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        teacher_model.eval()

        meter_loss = 0.0
        meter_loss_x = 0.0
        meter_loss_u = 0.0
        meter_mask_ratio = 0.0

        iter_l = iter(loader_l)
        iter_u = iter(loader_u)

        print_freq = max(1, steps_per_epoch // 8)

        for step in range(steps_per_epoch):
            step_start = time.time()

            # Get data
            try:
                _, img_x, target_x = next(iter_l)
            except StopIteration:
                iter_l = iter(loader_l)
                _, img_x, target_x = next(iter_l)
            try:
                _, img_u_w, img_u_s1, img_u_s2, _ = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                _, img_u_w, img_u_s1, img_u_s2, _ = next(iter_u)

            img_x = img_x.to(device)
            img_u_w = img_u_w.to(device)
            img_u_s1 = img_u_s1.to(device)
            img_u_s2 = img_u_s2.to(device)

            # ============================================
            # Step 1: Teacher generates pseudo labels
            # (from weakly-augmented images, Eq. 3-4)
            # ============================================
            with torch.no_grad():
                teacher_preds = teacher_model(img_u_w)

            _, _, img_h, img_w = img_u_w.shape
            per_img_dets = _teacher_per_image_dets(
                teacher_preds, iou_thres_nms, max_det, img_w, img_h
            )

            # ============================================
            # Step 2: CutMix on strong views (Eq. 2, 9)
            # Each stream gets independent CutMix
            # ============================================
            img_u_s1, dets_cutmixed1 = apply_detection_cutmix(
                img_u_s1, per_img_dets,
                cutmix_prob=cutmix_prob,
                size_min=cutmix_size_min,
                size_max=cutmix_size_max,
            )
            img_u_s2, dets_cutmixed2 = apply_detection_cutmix(
                img_u_s2, per_img_dets,
                cutmix_prob=cutmix_prob,
                size_min=cutmix_size_min,
                size_max=cutmix_size_max,
            )

            # Build pseudo batches with confidence filtering (𝟙(conf >= τ))
            pseudo_batch_s1, p_ratio1, p_conf1 = _build_pseudo_batch(
                dets_cutmixed1, img_u_s1, conf_thresh, device
            )
            pseudo_batch_s2, p_ratio2, p_conf2 = _build_pseudo_batch(
                dets_cutmixed2, img_u_s2, conf_thresh, device
            )

            # ============================================
            # Step 3: Supervised loss (Eq. 1)
            # ============================================
            student_model.train()
            for m in student_model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()

            batch_l = _build_yolo_batch(img_x, target_x, device)
            pred_x = student_model(batch_l["img"])
            loss_x = _safe_yolo_loss(student_model, batch_l, pred_x)

            # ============================================
            # Step 4: Unsupervised loss with comp dropout
            # (Eq. 14-17 — complementary channel-wise dropout)
            # ============================================
            loss_u_s1 = torch.tensor(0.0, device=device)
            loss_u_s2 = torch.tensor(0.0, device=device)

            comp_ctx.reset()
            comp_ctx.enabled = True

            # Stream 1: forward with mask M
            if pseudo_batch_s1["batch_idx"].numel() > 0:
                comp_ctx.stream = 0
                preds_s1 = yolo_forward_comp_drop(
                    student_model, img_u_s1, backbone_saves, comp_ctx
                )
                loss_u_s1 = _safe_yolo_loss(student_model, pseudo_batch_s1, preds_s1)

            # Stream 2: forward with complementary mask (1 - M)
            if pseudo_batch_s2["batch_idx"].numel() > 0:
                comp_ctx.stream = 1
                preds_s2 = yolo_forward_comp_drop(
                    student_model, img_u_s2, backbone_saves, comp_ctx
                )
                loss_u_s2 = _safe_yolo_loss(student_model, pseudo_batch_s2, preds_s2)

            comp_ctx.enabled = False

            # Dual-stream unsupervised loss (Eq. 17)
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            # Total loss (Eq. 5, paper-faithful): L = (L_l + L_u) / 2
            loss = (loss_x + loss_u_s) / 2.0

            # ============================================
            # Step 5: Backward + update
            # ============================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ============================================
            # Step 6: EMA teacher update (Eq. 18)
            # γ = min(1 - 1/(iter+1), 0.996), clamped by ema_floor.
            # YOLO teacher is initialized from a fully-trained best.pt (unlike
            # Origin where the decoder is randomly initialized), so the iter-0
            # warmup value of 0 would wipe out the pretrained weights. The
            # floor keeps γ above a safe threshold from step 0.
            # ============================================
            iters = epoch * steps_per_epoch + step
            ema_ratio = max(ema_floor, min(1 - 1 / (iters + 1), ema_decay_max))

            with torch.no_grad():
                for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
                    p_t.copy_(p_t * ema_ratio + p_s.detach() * (1 - ema_ratio))
                for b_s, b_t in zip(student_model.buffers(), teacher_model.buffers()):
                    if b_t.dtype in (torch.float32, torch.float64, torch.float16):
                        b_t.copy_(b_t * ema_ratio + b_s.detach() * (1 - ema_ratio))
                    else:
                        b_t.copy_(b_s)

            # Learning rate schedule: poly decay (UniMatch V2 Section 4.2)
            lr = lr_encoder * (1 - iters / total_iters) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Logging
            meter_loss += loss.item()
            meter_loss_x += loss_x.item()
            meter_loss_u += loss_u_s.item()
            mask_ratio = (p_ratio1 + p_ratio2) / 2.0
            meter_mask_ratio += mask_ratio

            if step % print_freq == 0 or step == steps_per_epoch - 1:
                done = step + 1
                elapsed = time.time() - step_start
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{step+1}/{steps_per_epoch}]  "
                    f"Loss: {loss.item():.4f}  L_x: {loss_x.item():.4f}  "
                    f"L_u: {loss_u_s.item():.4f}  "
                    f"PseudoRatio: {mask_ratio:.3f}  "
                    f"EMA: {ema_ratio:.4f}  LR: {lr:.2e}  "
                    f"Time: {elapsed:.2f}s"
                )

        # End of epoch
        avg_loss = meter_loss / max(1, steps_per_epoch)
        avg_loss_x = meter_loss_x / max(1, steps_per_epoch)
        avg_loss_u = meter_loss_u / max(1, steps_per_epoch)
        avg_mask = meter_mask_ratio / max(1, steps_per_epoch)

        # ============================================
        # Validation (EMA teacher — paper Section 4.2)
        # ============================================
        val_map50 = -1.0
        is_last = (epoch == epochs - 1)
        if (epoch + 1) % val_every == 0 or is_last:
            try:
                tmp_pt = save_path / "_tmp_val.pt"
                # Place TEACHER in the 'model' slot so YOLO().val() evaluates EMA
                _save_ckpt_teacher_for_val(teacher_model, epoch, tmp_pt, imgsz)
                val_yolo = yolo_cls(str(tmp_pt))
                # Val protocol matches iMAS_Remake / AD-MT_Remake (deploy-like):
                # conf=0.1 iou=0.1 — selects best.pt by test-relevant metric instead of
                # paper-style conf=0.001 iou=0.6 (which inflates mAP and picks wrong ckpt).
                val_results = val_yolo.val(
                    data=str(val_yaml_path),
                    imgsz=imgsz,
                    batch=batch_size,
                    conf=0.1,
                    iou=0.1,
                    verbose=False,
                )
                val_map50 = float(val_results.results_dict.get("metrics/mAP50(B)", -1.0))
                if tmp_pt.exists():
                    tmp_pt.unlink()
            except Exception as e:
                print(f"[VAL] Warning: {e}")

        # Save checkpoints
        is_best = val_map50 > best_val_map50
        if val_map50 >= 0:
            best_val_map50 = max(best_val_map50, val_map50)

        ckpt = {
            "model": student_model.state_dict(),
            "model_ema": teacher_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_map50": best_val_map50,
        }
        torch.save(ckpt, latest_path)
        if is_best:
            torch.save(ckpt, save_path / "best.pth")
            _save_ckpt(student_model, teacher_model, epoch, save_path / "best.pt", imgsz)

        # Also save ultralytics-compatible checkpoint
        _save_ckpt(student_model, teacher_model, epoch, save_path / "last.pt", imgsz)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "loss_x": avg_loss_x,
            "loss_u": avg_loss_u,
            "pseudo_ratio": avg_mask,
            "val_mAP50": val_map50,
            "best_mAP50": best_val_map50,
        })

        print(
            f"Epoch [{epoch+1}/{epochs}] DONE  "
            f"Loss: {avg_loss:.4f}  L_x: {avg_loss_x:.4f}  L_u: {avg_loss_u:.4f}  "
            f"PseudoRatio: {avg_mask:.3f}  "
            f"Val_mAP50: {val_map50:.4f}  Best: {best_val_map50:.4f}"
        )

    # Save final summary
    summary = {
        "config": str(Path(args.config).resolve()),
        "student_pt": student_pt,
        "teacher_pt": teacher_pt,
        "epochs": epochs,
        "best_val_map50": best_val_map50,
        "history": history,
    }
    with open(save_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Best mAP50: {best_val_map50:.4f}")
    print(f"[DONE] Checkpoints saved to: {save_path}")


def _save_ckpt(student_model, teacher_model, epoch, path, imgsz):
    """Save ultralytics-compatible checkpoint (student in 'model' slot)."""
    ensure_dir(Path(path).parent)
    ckpt = {
        "epoch": epoch,
        "model": copy.deepcopy(student_model).eval(),
        "ema": copy.deepcopy(teacher_model).eval(),
        "updates": epoch,
        "train_args": {"task": "detect", "imgsz": imgsz},
    }
    torch.save(ckpt, path)


def _save_ckpt_teacher_for_val(teacher_model, epoch, path, imgsz):
    """Save a tmp checkpoint with TEACHER in the 'model' slot.

    Ultralytics YOLO(path).val() loads ckpt['model'] for evaluation. UniMatch V2
    selects the best checkpoint via the EMA teacher (paper Section 4.2), so we
    place the teacher there for validation. Used ONLY for tmp val ckpts.
    """
    ensure_dir(Path(path).parent)
    ckpt = {
        "epoch": epoch,
        "model": copy.deepcopy(teacher_model).eval(),
        "ema": copy.deepcopy(teacher_model).eval(),
        "updates": epoch,
        "train_args": {"task": "detect", "imgsz": imgsz},
    }
    torch.save(ckpt, path)


if __name__ == "__main__":
    main()
