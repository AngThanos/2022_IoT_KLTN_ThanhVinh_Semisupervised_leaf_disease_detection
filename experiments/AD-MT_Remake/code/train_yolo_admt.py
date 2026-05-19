"""
AD-MT (Alternate Diverse Teaching) for semi-supervised YOLO detection.

References:
  - AD-MT paper:  https://arxiv.org/abs/2311.17325  (ECCV 2024)
  - iMAS_Remake:  model loading, data pipeline, loss helpers

Components:
  1. Model:  YOLO loaded via Ultralytics (setup_yolo_model)
  2. Data:   iMAS-style separate labeled/unlabeled loaders with
             geometric augmentation + strong augmentation
  3. Loss:   YOLO detection loss (box/cls/dfl) with autograd graph
  4. Algo:   AD-MT  →  RPA (Random Periodic Alternate EMA updating)
                     +  CCM (Conflict-Combating Module for pseudo boxes)
"""
import argparse
import copy
import logging
import math
import os
import os.path as osp
import pprint
import random
import sys

import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloaders.yolo_loader import build_yolo_semi_loaders, collate_labeled
from train_utils import AlternateUpdate
from utils import ramps
from utils.util import time_str, AverageMeter


# ═══════════════════════════════════════════════════════════════════
#  I.  Helper functions
# ═══════════════════════════════════════════════════════════════════

def get_current_consistency_weight(epoch, args):
    """Sigmoid ramp-up for consistency loss weight."""
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def get_current_conf_threshold(iter_num, args):
    """Progressive confidence threshold warmup."""
    target = args["conf_threshold"]
    start = args.get("conf_threshold_min", target * 0.3)
    warmup = args.get("conf_threshold_warmup", 2000)
    if iter_num >= warmup:
        return target
    return start + (target - start) * (iter_num / warmup)


def update_ema_variables(model, ema_model, alpha, global_step, args):
    """EMA update with warmup schedule (AD-MT paper)."""
    if global_step < args["consistency_rampup"]:
        alpha = 0.0
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(1 - alpha, p.data)
    for buf_t, buf_s in zip(ema_model.buffers(), model.buffers()):
        buf_t.data = buf_t.data * alpha + buf_s.data * (1 - alpha)


# ───────────────────────────────────────────────────────────────────
#  I-a.  YOLO model helpers
# ───────────────────────────────────────────────────────────────────

def setup_yolo_model(pretrain_path):
    """Load YOLO architecture + weights from Ultralytics."""
    from ultralytics import YOLO
    from types import SimpleNamespace
    yolo_wrapper = YOLO(pretrain_path)
    model = yolo_wrapper.model
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    if hasattr(model, "criterion"):
        model.criterion = None
    return model


def save_yolo_pt(model, save_path, pretrain_path):
    """Save model as Ultralytics-compatible .pt checkpoint.

    This creates a checkpoint that can be loaded directly via
    ``YOLO(save_path)`` for inference or further training.
    """
    from ultralytics import YOLO
    # Load a fresh wrapper to get the original checkpoint structure
    ref = YOLO(pretrain_path)
    # Overwrite weights with trained ones
    ref_model = ref.model
    ref_model.load_state_dict(model.state_dict())
    # Build ultralytics-style checkpoint dict
    try:
        train_args = vars(ref.model.args)
    except TypeError:
        train_args = dict(ref.model.args.__dict__) if hasattr(ref.model.args, "__dict__") else {}
    ckpt = {
        "model": copy.deepcopy(ref_model).half(),
        "train_args": train_args,
    }
    torch.save(ckpt, save_path)
    del ref  # free GPU/CPU memory


def yolo_get_pseudo_boxes(model, imgs, conf_thresh=0.25, iou_thresh=0.45, img_size=640):
    """Teacher inference → NMS → list of (N_i, 5) [cls, cx, cy, w, h] per image."""
    from ultralytics.utils.nms import non_max_suppression
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs)
        preds = out[0] if isinstance(out, (list, tuple)) else out
        results = non_max_suppression(preds, conf_thres=conf_thresh,
                                      iou_thres=iou_thresh, max_det=300)
    if was_training:
        model.train()

    boxes_list, conf_list = [], []
    for dets in results:
        if len(dets) == 0:
            boxes_list.append(torch.zeros(0, 5, device=imgs.device))
            conf_list.append(torch.zeros(0, device=imgs.device))
            continue
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        conf, cls = dets[:, 4], dets[:, 5]
        cx = (x1 + x2) / 2 / img_size
        cy = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        boxes_list.append(torch.stack([cls, cx, cy, w, h], dim=1))
        conf_list.append(conf)
    return boxes_list, conf_list


def boxes_list_to_batch(boxes_list, imgs):
    """List of per-image (N_i, 5) boxes → dict for YOLO criterion."""
    cls_all, bboxes_all, bidx_all = [], [], []
    device = imgs.device
    for b, boxes in enumerate(boxes_list):
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.to(device)
        cls_all.append(boxes[:, 0].float())
        bboxes_all.append(boxes[:, 1:].float())
        bidx_all.append(torch.full((len(boxes),), b, dtype=torch.float32, device=device))
    if cls_all:
        return {"cls": torch.cat(cls_all), "bboxes": torch.cat(bboxes_all),
                "batch_idx": torch.cat(bidx_all), "img": imgs}
    return {"cls": torch.zeros(0, device=device),
            "bboxes": torch.zeros(0, 4, device=device),
            "batch_idx": torch.zeros(0, device=device), "img": imgs}


def yolo_detection_loss_with_grad(model, preds, batch):
    """Compute YOLO loss while preserving autograd graph."""
    from ultralytics.utils.tal import make_anchors

    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    crit = model.criterion

    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat(
        [xi.view(feats[0].shape[0], crit.no, -1) for xi in feats], 2
    ).split((crit.reg_max * 4, crit.nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    bs = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=crit.device, dtype=dtype) * crit.stride[0]
    anchor_points, stride_tensor = make_anchors(feats, crit.stride, 0.5)

    targets = torch.cat((batch["batch_idx"].view(-1, 1),
                          batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = crit.preprocess(targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    pred_bboxes = crit.bbox_decode(anchor_points, pred_distri)
    _, target_bboxes, target_scores, fg_mask, _ = crit.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
    )

    target_scores_sum = target_scores.sum().clamp(min=1.0)
    loss_cls = crit.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
    loss_box = loss_dfl = torch.tensor(0.0, device=crit.device)
    if fg_mask.sum():
        loss_box, loss_dfl = crit.bbox_loss(
            pred_distri, pred_bboxes, anchor_points,
            target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask,
        )

    loss_box *= crit.hyp.box;  loss_cls *= crit.hyp.cls;  loss_dfl *= crit.hyp.dfl
    total_loss = (loss_box + loss_cls + loss_dfl) * bs
    return total_loss, torch.stack((loss_box.detach(), loss_cls.detach(), loss_dfl.detach()))


# ───────────────────────────────────────────────────────────────────
#  I-b.  CCM (Conflict-Combating Module) for detection
# ───────────────────────────────────────────────────────────────────

def _boxes_to_xyxy(boxes, img_size):
    if len(boxes) == 0:
        return torch.zeros(0, 4, device=boxes.device)
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    return torch.stack([(cx - w / 2) * img_size, (cy - h / 2) * img_size,
                        (cx + w / 2) * img_size, (cy + h / 2) * img_size], dim=1)


def ccm_split_teachers(boxes1, conf1, boxes2, conf2, img_size=640, match_iou=0.5,
                       ensemble_temp=1.0, entropy_weighted=True):
    """
    AD-MT CCM for detection (Eq.8 paper): split two teachers' predictions into
      consistent (IoU-matched, same class) and conflicting (unmatched).

    For consistent boxes, uses entropy-weighted fusion:
      w_k = exp(-H_k) / sum(exp(-H_k))   where H_k ≈ -log(conf_k)
    Then fuses confidence: w1*c1 + w2*c2 and picks class from higher-weight teacher.
    Falls back to argmax(conf) when entropy_weighted=False.

    Returns: consist_boxes, consist_conf, conflict_boxes, conflict_conf,
             conflict_ratio, conflict_src  (list of 1/2 indicating source teacher)
    """
    from torchvision.ops import box_iou as tv_box_iou
    device = boxes1.device
    empty5 = torch.zeros(0, 5, device=device)
    empty1 = torch.zeros(0, device=device)
    empty_int = torch.zeros(0, dtype=torch.long, device=device)

    if len(boxes1) == 0 and len(boxes2) == 0:
        return empty5, empty1, empty5, empty1, torch.tensor(0.0, device=device), empty_int
    if len(boxes1) == 0:
        src = torch.full((len(boxes2),), 2, dtype=torch.long, device=device)
        return empty5, empty1, boxes2, conf2, torch.tensor(1.0, device=device), src
    if len(boxes2) == 0:
        src = torch.full((len(boxes1),), 1, dtype=torch.long, device=device)
        return empty5, empty1, boxes1, conf1, torch.tensor(1.0, device=device), src

    xyxy1 = _boxes_to_xyxy(boxes1, img_size)
    xyxy2 = _boxes_to_xyxy(boxes2, img_size)
    iou_mat = tv_box_iou(xyxy1, xyxy2)

    matched1, matched2 = set(), set()
    consist_b, consist_c = [], []
    flat_order = torch.argsort(iou_mat.reshape(-1), descending=True)
    for idx in flat_order:
        i = (idx // iou_mat.shape[1]).item()
        j = (idx % iou_mat.shape[1]).item()
        if iou_mat[i, j].item() < match_iou:
            break
        if i in matched1 or j in matched2:
            continue
        if int(boxes1[i, 0]) != int(boxes2[j, 0]):
            continue
        matched1.add(i);  matched2.add(j)

        c1_val = conf1[i].item()
        c2_val = conf2[j].item()

        if entropy_weighted:
            # Eq.8: entropy-weighted fusion  H_k ≈ -log(conf_k)
            h1 = -math.log(max(c1_val, 1e-6))
            h2 = -math.log(max(c2_val, 1e-6))
            w1 = math.exp(-h1 / max(ensemble_temp, 1e-6))
            w2 = math.exp(-h2 / max(ensemble_temp, 1e-6))
            w_sum = w1 + w2 + 1e-8
            w1, w2 = w1 / w_sum, w2 / w_sum
            # Fused confidence
            fused_conf = w1 * c1_val + w2 * c2_val
            # Pick box geometry from higher-weight teacher
            if w1 >= w2:
                fused_box = boxes1[i].clone()
            else:
                fused_box = boxes2[j].clone()
            consist_b.append(fused_box)
            consist_c.append(torch.tensor(fused_conf, device=device))
        else:
            # Fallback: argmax confidence
            if c1_val >= c2_val:
                consist_b.append(boxes1[i]);  consist_c.append(conf1[i])
            else:
                consist_b.append(boxes2[j]);  consist_c.append(conf2[j])

    conflict_b, conflict_c, conflict_src = [], [], []
    for i in range(len(boxes1)):
        if i not in matched1:
            conflict_b.append(boxes1[i]);  conflict_c.append(conf1[i])
            conflict_src.append(1)
    for j in range(len(boxes2)):
        if j not in matched2:
            conflict_b.append(boxes2[j]);  conflict_c.append(conf2[j])
            conflict_src.append(2)

    nc, nx = len(consist_b), len(conflict_b)
    cr = torch.tensor(nx / max(nc + nx, 1), device=device, dtype=torch.float32)
    return (torch.stack(consist_b) if consist_b else empty5,
            torch.stack(consist_c) if consist_c else empty1,
            torch.stack(conflict_b) if conflict_b else empty5,
            torch.stack(conflict_c) if conflict_c else empty1, cr,
            torch.tensor(conflict_src, dtype=torch.long, device=device) if conflict_src else empty_int)


# ───────────────────────────────────────────────────────────────────
#  I-b2. CCM: Student arbitration for conflict boxes (Eq.9 paper)
#         Mirrors get_compromise_pseudo_btw_tea_stu from Origin
# ───────────────────────────────────────────────────────────────────

def ccm_student_arbitration(conflict_boxes, conflict_conf, conflict_src,
                            stu_boxes, stu_conf,
                            mode="pixel_confidence", stu_use_more=False,
                            img_size=640, match_iou=0.3):
    """
    AD-MT CCM Eq.9: use student prediction to arbitrate teacher conflicts.

    For each conflict box from teacher, find matching student box (IoU + class).
    Depending on mode, decide whether to keep teacher box or replace with student:
      - "pixel_confidence": keep whichever has higher confidence
      - "random": 50% chance replace with student
      - "always_tea": always keep teacher
      - "always_stu": always replace with student (if match exists)

    Args:
        conflict_boxes: (N, 5) [cls, cx, cy, w, h] from teachers
        conflict_conf:  (N,) confidence
        conflict_src:   (N,) which teacher (1 or 2)
        stu_boxes:      (M, 5) student predictions
        stu_conf:       (M,) student confidence
        mode:           conflict resolution mode
        stu_use_more:   if True, student can override even non-conflict boxes
                        when confidence difference exceeds threshold
        img_size:       image size for IoU computation
        match_iou:      minimum IoU to consider a match

    Returns:
        final_boxes:    (K, 5) resolved pseudo-labels
        final_conf:     (K,)
        n_from_student: int, how many boxes came from student
    """
    from torchvision.ops import box_iou as tv_box_iou
    device = conflict_boxes.device

    if len(conflict_boxes) == 0:
        return conflict_boxes, conflict_conf, 0

    if len(stu_boxes) == 0:
        # No student predictions → keep all teacher conflicts as-is
        return conflict_boxes, conflict_conf, 0

    # Compute IoU between conflict boxes and student boxes
    xyxy_conf = _boxes_to_xyxy(conflict_boxes, img_size)
    xyxy_stu = _boxes_to_xyxy(stu_boxes, img_size)
    iou_mat = tv_box_iou(xyxy_conf, xyxy_stu)  # (N, M)

    final_boxes = conflict_boxes.clone()
    final_conf = conflict_conf.clone()
    n_from_student = 0

    used_stu = set()
    for i in range(len(conflict_boxes)):
        # Find best matching student box with same class
        best_j, best_iou = -1, match_iou
        for j in range(len(stu_boxes)):
            if j in used_stu:
                continue
            if int(conflict_boxes[i, 0]) != int(stu_boxes[j, 0]):
                continue
            if iou_mat[i, j].item() > best_iou:
                best_iou = iou_mat[i, j].item()
                best_j = j

        if best_j < 0:
            continue  # No student match → keep teacher box

        used_stu.add(best_j)
        tea_c = conflict_conf[i].item()
        stu_c = stu_conf[best_j].item()

        replace_with_student = False
        if mode == "always_stu":
            replace_with_student = True
        elif mode == "always_tea":
            replace_with_student = False
        elif mode == "random":
            replace_with_student = (random.random() < 0.5)
        elif mode == "pixel_confidence":
            replace_with_student = (stu_c > tea_c)
        else:
            replace_with_student = (stu_c > tea_c)  # fallback

        if replace_with_student:
            final_boxes[i] = stu_boxes[best_j]
            final_conf[i] = stu_conf[best_j]
            n_from_student += 1

    # NOTE: `stu_use_more` (origin: mtx_teacher_conflict=None) means student
    # is ALLOWED to override even non-conflict (consist) boxes when its
    # prediction is more confident — handled at caller side by passing
    # consist boxes through this function too. Here we keep the conflict
    # arbitration logic identical regardless of `stu_use_more`; the caller
    # is responsible for widening the arbitration scope.
    _ = stu_use_more  # accepted for API compat; logic moved to caller

    return final_boxes, final_conf, n_from_student


# ───────────────────────────────────────────────────────────────────
#  I-c.  Detection CutMix
# ───────────────────────────────────────────────────────────────────

def detection_cutmix(imgs_a, boxes_a, imgs_b, boxes_b):
    """CutMix for detection: paste random patch from B onto A.

    Mirrors origin `cut_mix`: shuffle B internally so that each image
    pastes a patch from a DIFFERENT image (avoid no-op when A==B).
    """
    B, C, H, W = imgs_a.shape
    mixed_imgs = imgs_a.clone()
    mixed_boxes = []
    # Shuffle source indices: each i pastes from b_idx[i] != i (when possible)
    perm = torch.randperm(B, device=imgs_a.device)
    # Avoid identity at any position if A and B are the same tensor
    if imgs_a.data_ptr() == imgs_b.data_ptr() and B > 1:
        for i in range(B):
            if perm[i].item() == i:
                j = (i + 1) % B
                perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    for i in range(B):
        src = perm[i].item()
        lam = np.random.beta(1.0, 1.0)
        cut_r = np.sqrt(1.0 - lam)
        cw, ch = int(W * cut_r), int(H * cut_r)
        cx_r, cy_r = np.random.randint(W), np.random.randint(H)
        x1 = max(cx_r - cw // 2, 0);  x2 = min(cx_r + cw // 2, W)
        y1 = max(cy_r - ch // 2, 0);  y2 = min(cy_r + ch // 2, H)
        mixed_imgs[i, :, y1:y2, x1:x2] = imgs_b[src, :, y1:y2, x1:x2]

        kept = []
        bxa, bxb = boxes_a[i], boxes_b[src]
        if len(bxa) > 0:
            bcx, bcy = bxa[:, 1] * W, bxa[:, 2] * H
            outside = ~((bcx >= x1) & (bcx <= x2) & (bcy >= y1) & (bcy <= y2))
            kept.append(bxa[outside])
        if len(bxb) > 0:
            bcx, bcy = bxb[:, 1] * W, bxb[:, 2] * H
            inside = (bcx >= x1) & (bcx <= x2) & (bcy >= y1) & (bcy <= y2)
            kept.append(bxb[inside])
        mixed_boxes.append(torch.cat(kept, 0) if kept
                           else torch.zeros(0, 5, device=imgs_a.device))
    return mixed_imgs, mixed_boxes


# ───────────────────────────────────────────────────────────────────
#  I-d.  Validation metrics
# ───────────────────────────────────────────────────────────────────

def compute_val_map(model, pretrain_path, dataset_yaml, img_size, batch_size=4,
                    snapshot_path="/tmp", root_path=None):
    """Run Ultralytics val() to get mAP50, mAP50-95, P, R.

    Saves a temp .pt, loads via YOLO(), runs .val(), returns dict.
    """
    from ultralytics import YOLO
    tmp_pt = osp.join(snapshot_path, "_tmp_val.pt")
    # Create a dataset yaml with absolute path to avoid Ultralytics path issues
    tmp_yaml = osp.join(snapshot_path, "_tmp_dataset.yaml")
    try:
        with open(dataset_yaml, "r") as f:
            ds_cfg = yaml.safe_load(f)
        # Force absolute path: use root_path from config if provided
        if root_path and osp.isdir(root_path):
            ds_cfg["path"] = osp.abspath(root_path)
        elif not osp.isabs(ds_cfg.get("path", "")):
            ds_cfg["path"] = osp.dirname(dataset_yaml)
        with open(tmp_yaml, "w") as f:
            yaml.dump(ds_cfg, f)

        save_yolo_pt(model, tmp_pt, pretrain_path)
        val_yolo = YOLO(tmp_pt)
        results = val_yolo.val(
            data=tmp_yaml,
            imgsz=int(img_size),
            batch=int(batch_size),
            conf=0.1,
            iou=0.1,
            verbose=True,
        )
        rd = results.results_dict if results is not None else {}
        metrics = {
            "mAP50":    float(rd.get("metrics/mAP50(B)", -1.0)),
            "mAP50_95": float(rd.get("metrics/mAP50-95(B)", -1.0)),
            "P":        float(rd.get("metrics/precision(B)", -1.0)),
            "R":        float(rd.get("metrics/recall(B)", -1.0)),
        }
    except Exception as e:
        logging.warning(f"[VAL] validation failed: {e}")
        metrics = {"mAP50": -1.0, "mAP50_95": -1.0, "P": -1.0, "R": -1.0}
    finally:
        if osp.exists(tmp_pt):
            os.remove(tmp_pt)
        if osp.exists(tmp_yaml):
            os.remove(tmp_yaml)
    return metrics


# ═══════════════════════════════════════════════════════════════════
#  II.  Training loop
# ═══════════════════════════════════════════════════════════════════

def train(args, snapshot_path):
    cur_time = time_str()
    writer = SummaryWriter(osp.join(snapshot_path, "log"))
    csv_train = osp.join(snapshot_path, "log", f"train_{cur_time}.csv")
    csv_test  = osp.join(snapshot_path, "log", f"val_{cur_time}.csv")

    base_lr        = args["base_lr"]
    max_iterations = args["max_iterations"]
    img_size       = args.get("img_size", 1024)

    # ── 1. Create student + two EMA teachers ──────────────────────
    model             = setup_yolo_model(args["model"])
    ema_model         = setup_yolo_model(args["model"])
    ema_model_another = setup_yolo_model(args["model"])

    model.cuda();  ema_model.cuda();  ema_model_another.cuda()

    for p in model.parameters():
        p.requires_grad_(True)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    for p in ema_model_another.parameters():
        p.requires_grad_(False)

    # Lazy-init criterion on GPU
    for m in (model, ema_model, ema_model_another):
        if hasattr(m, "criterion"):
            m.criterion = None

    model.train();  ema_model.train();  ema_model_another.train()

    # AMP scaler
    use_amp = bool(args.get("use_amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 2. Data loaders (iMAS-style) ─────────────────────────────
    loader_lb, loader_ulb, loader_val = build_yolo_semi_loaders(args)

    logging.info(f"Labeled iters/ep: {len(loader_lb)}, "
                 f"Unlabeled iters/ep: {len(loader_ulb)}")

    # ── 3. Optimizer ──────────────────────────────────────────────
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # ── 4. AD-MT RPA (Random Periodic Alternate) scheduler ────────
    rpa_period = args["alt_param_updating_period_iters"]
    rpa_random = args["alt_flag_updating_period_random"]
    rpa = AlternateUpdate(rpa_period, initial_flag=True, flag_random=rpa_random)

    # ── 4b. AD-MT CCM config flags ──────────────────────────────
    conflict_weight = args["alt_param_conflict_weight"]
    conflict_mode = args.get("alt_flag_conflict_mode", "pixel_confidence")
    stu_use_more = args.get("alt_flag_conflict_stu_use_more", False)
    ensemble_temp = args.get("alt_param_ensemble_temp", 1.0)
    entropy_weighted = args.get("alt_flag_entropy_weighted_fusion", True)

    # ── 4c. Self-learning config (origin infrastructure) ────
    flag_pseudo_from_student = args.get("flag_pseudo_from_student", False)
    self_learning_start_iter = args.get("self_learning_start_iter", -1)
    threshold_self_training = args.get("alt_param_threshold_self_training", 0.5)

    logging.info(f"[AD-MT Config] CCM: weight={conflict_weight}, mode={conflict_mode}, "
                 f"stu_use_more={stu_use_more}, ensemble_temp={ensemble_temp}, "
                 f"entropy_weighted={entropy_weighted}")
    logging.info(f"[AD-MT Config] RPA: period={rpa_period}, random={rpa_random}")
    logging.info(f"[AD-MT Config] SelfLearn: pseudo_from_stu={flag_pseudo_from_student}, "
                 f"start_iter={self_learning_start_iter}, threshold={threshold_self_training}")

    # ── 5. Training ───────────────────────────────────────────────
    iter_num = 0
    iters_per_epoch = min(len(loader_lb), len(loader_ulb))
    max_epoch = max_iterations // iters_per_epoch + 1

    best_map50_tea = -1.0
    best_map50_tea2 = -1.0
    best_map50_stu = -1.0
    start_epoch = 0
    dataset_yaml = args.get("dataset_yaml",
                            osp.join(args["root_path"], "meta", "dataset_supervised.yaml"))
    log_interval = max(iters_per_epoch // 8, 1)  # print ~8 times per epoch

    # ── 5.0 Auto-resume from last_ckpt.pt if present ──────────────
    # Saves/restores STATE only — does NOT alter any AD-MT logic.
    last_ckpt_path = osp.join(snapshot_path, "last_ckpt.pt")
    if osp.exists(last_ckpt_path):
        try:
            ck = torch.load(last_ckpt_path, map_location="cuda", weights_only=False)
            model.load_state_dict(ck["model"])
            ema_model.load_state_dict(ck["ema"])
            ema_model_another.load_state_dict(ck["ema_another"])
            optimizer.load_state_dict(ck["optimizer"])
            if "scaler" in ck and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            iter_num         = int(ck.get("iter_num", 0))
            start_epoch      = int(ck.get("epoch_num", -1)) + 1
            best_map50_stu   = float(ck.get("best_map50_stu",  -1.0))
            best_map50_tea   = float(ck.get("best_map50_tea",  -1.0))
            best_map50_tea2  = float(ck.get("best_map50_tea2", -1.0))
            rpa_state = ck.get("rpa_state", {})
            if rpa_state:
                rpa._counter                 = int(rpa_state.get("counter", 0))
                rpa.flag_alternate           = bool(rpa_state.get("flag", True))
                rpa.random_alternate_period  = int(rpa_state.get("period",
                                                                 rpa.alternate_period))
            logging.info(f"[RESUME] loaded {last_ckpt_path} "
                         f"→ iter={iter_num}, start_epoch={start_epoch}, "
                         f"best_tea={best_map50_tea:.4f}")
            print(f"[RESUME] iter={iter_num}, epoch={start_epoch}, "
                  f"best_tea_mAP50={best_map50_tea:.4f}")
        except Exception as e:
            logging.warning(f"[RESUME] failed to load {last_ckpt_path}: {e}. "
                            f"Starting from scratch.")

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70, initial=start_epoch,
                    total=max_epoch)

    for epoch_num in iterator:
        # RPA: optionally shuffle initial teacher flag each epoch
        if args.get("alt_flag_epoch_shuffle_teachers", False):
            rpa.reset(rpa_period, initial_flag=(epoch_num % 2 == 0),
                      flag_random=rpa_random)

        # Self-learning flag (origin: checked per-epoch, activated by iteration threshold)
        flag_start_self_learning = False
        if self_learning_start_iter > 0 and iter_num >= self_learning_start_iter:
            flag_start_self_learning = True
        if flag_pseudo_from_student:
            flag_start_self_learning = True

        # Metric meters
        m_loss_lb  = AverageMeter()
        m_loss_ulb = AverageMeter(20)
        m_loss_all = AverageMeter(20)
        m_lr       = AverageMeter()
        m_high_r   = AverageMeter()
        m_conflict = AverageMeter()
        m_pseudo_r = AverageMeter()   # avg pseudo-box count per image
        m_loss_consist  = AverageMeter(20)
        m_loss_conflict = AverageMeter(20)
        m_n_from_stu    = AverageMeter()       # student arbitration acceptance rate
        m_rpa_switches  = 0                     # count RPA switches this epoch

        print(f"Epoch/Start [{epoch_num + 1}/{max_epoch}], "
              f"steps={iters_per_epoch}, batch={args.get('batch_size_lb', 4)}, "
              f"imgsz={img_size}")

        iter_lb  = iter(loader_lb)
        iter_ulb = iter(loader_ulb)
        t_start = time.time()

        for step in range(iters_per_epoch):
            # ── 5a. Load data ─────────────────────────────────────
            try:
                _, img_lb, target_lb = next(iter_lb)
            except StopIteration:
                iter_lb = iter(loader_lb)
                _, img_lb, target_lb = next(iter_lb)

            try:
                _, img_ulb_w, img_ulb_s, _ = next(iter_ulb)
            except StopIteration:
                iter_ulb = iter(loader_ulb)
                _, img_ulb_w, img_ulb_s, _ = next(iter_ulb)

            img_lb    = img_lb.cuda()
            img_ulb_w = img_ulb_w.cuda()
            img_ulb_s = img_ulb_s.cuda()
            target_lb = [t.cuda() for t in target_lb]

            num_lb  = img_lb.size(0)
            num_ulb = img_ulb_w.size(0)

            # ── 5b. Teacher pseudo-label generation (AD-MT) ───────
            loss_ulb_consist  = torch.tensor(0.0).cuda()
            loss_ulb_conflict = torch.tensor(0.0).cuda()
            conflict_ratio    = torch.tensor(0.0).cuda()

            with torch.no_grad():
                cur_conf = get_current_conf_threshold(iter_num, args)
                low_thresh = max(cur_conf * 0.3, 0.01)

                if not flag_start_self_learning:
                    # ── Normal AD-MT: 2-teacher pseudo generation ──
                    # RPA: set active / frozen teacher
                    if rpa.get_alternate_state():
                        ema_model.train();  ema_model_another.eval()
                    else:
                        ema_model_another.train();  ema_model.eval()

                    # Both teachers infer on weak unlabeled
                    pseudo_b1, pseudo_c1 = yolo_get_pseudo_boxes(
                        ema_model, img_ulb_w, conf_thresh=low_thresh, img_size=img_size)
                    pseudo_b2, pseudo_c2 = yolo_get_pseudo_boxes(
                        ema_model_another, img_ulb_w, conf_thresh=low_thresh, img_size=img_size)
                    ema_model.train();  ema_model_another.train()

                    # CCM: split consistent vs conflicting (Eq.8: entropy-weighted fusion)
                    consist_pseudo, consist_conf_list = [], []
                    conflict_pseudo, conflict_conf_list, conflict_src_list = [], [], []
                    crs = []
                    for pb1, pc1, pb2, pc2 in zip(pseudo_b1, pseudo_c1, pseudo_b2, pseudo_c2):
                        cb, cc, xb, xc, cr, xsrc = ccm_split_teachers(
                            pb1, pc1, pb2, pc2, img_size=img_size,
                            ensemble_temp=ensemble_temp,
                            entropy_weighted=entropy_weighted)
                        # Consistent: filter by cur_conf
                        c_mask = (cc >= cur_conf) if len(cc) > 0 else torch.zeros(0, dtype=torch.bool)
                        consist_pseudo.append(cb[c_mask] if len(cb) > 0 and c_mask.any()
                                              else torch.zeros(0, 5, device=img_ulb_w.device))
                        consist_conf_list.append(cc[c_mask] if len(cc) > 0 and c_mask.any()
                                                 else torch.zeros(0, device=img_ulb_w.device))
                        # Conflict: filter by low_thresh (more permissive for student arbitration)
                        x_mask = (xc >= low_thresh) if len(xc) > 0 else torch.zeros(0, dtype=torch.bool)
                        conflict_pseudo.append(xb[x_mask] if len(xb) > 0 and x_mask.any()
                                               else torch.zeros(0, 5, device=img_ulb_w.device))
                        conflict_conf_list.append(xc[x_mask] if len(xc) > 0 and x_mask.any()
                                                  else torch.zeros(0, device=img_ulb_w.device))
                        conflict_src_list.append(xsrc[x_mask] if len(xsrc) > 0 and x_mask.any()
                                                 else torch.zeros(0, dtype=torch.long, device=img_ulb_w.device))
                        crs.append(cr)
                    conflict_ratio = torch.stack(crs).mean()

                    # ── CCM Eq.9: Student arbitration for conflict boxes ──
                    # Origin semantics (mtx_teacher_conflict=None ⇔ stu_use_more):
                    #   - stu_use_more=False → student only overrides TEACHER-CONFLICT region.
                    #   - stu_use_more=True  → student can override CONSIST region too.
                    n_from_stu_total = 0
                    need_arbitration = (conflict_weight != 0 and
                                        sum(len(b) for b in conflict_pseudo) > 0) \
                                       or (stu_use_more and
                                           sum(len(b) for b in consist_pseudo) > 0)
                    if need_arbitration:
                        # Get student predictions on weak view (no grad, for arbitration only)
                        stu_b_list, stu_c_list = yolo_get_pseudo_boxes(
                            model, img_ulb_w, conf_thresh=low_thresh, img_size=img_size)

                        for idx_img in range(num_ulb):
                            xb = conflict_pseudo[idx_img]
                            xc = conflict_conf_list[idx_img]
                            xsrc = conflict_src_list[idx_img]
                            sb = stu_b_list[idx_img] if idx_img < len(stu_b_list) else torch.zeros(0, 5, device=img_ulb_w.device)
                            sc = stu_c_list[idx_img] if idx_img < len(stu_c_list) else torch.zeros(0, device=img_ulb_w.device)

                            # Arbitrate over conflict boxes
                            if len(xb) > 0:
                                resolved_b, resolved_c, n_stu = ccm_student_arbitration(
                                    xb, xc, xsrc, sb, sc,
                                    mode=conflict_mode,
                                    stu_use_more=stu_use_more,
                                    img_size=img_size)
                                # Filter resolved by cur_conf
                                r_mask = resolved_c >= cur_conf
                                conflict_pseudo[idx_img] = resolved_b[r_mask] if r_mask.any() else torch.zeros(0, 5, device=img_ulb_w.device)
                                conflict_conf_list[idx_img] = resolved_c[r_mask] if r_mask.any() else torch.zeros(0, device=img_ulb_w.device)
                                n_from_stu_total += n_stu

                            # stu_use_more: also arbitrate over CONSIST boxes
                            # (origin: when mtx_teacher_conflict=None, student can
                            #  override anywhere target_tea != target_stu).
                            if stu_use_more and len(consist_pseudo[idx_img]) > 0:
                                cb = consist_pseudo[idx_img]
                                cc = consist_conf_list[idx_img]
                                csrc = torch.zeros(len(cb), dtype=torch.long,
                                                   device=img_ulb_w.device)  # dummy src
                                resolved_b, resolved_c, n_stu2 = ccm_student_arbitration(
                                    cb, cc, csrc, sb, sc,
                                    mode=conflict_mode,
                                    stu_use_more=False,
                                    img_size=img_size)
                                r_mask = resolved_c >= cur_conf
                                consist_pseudo[idx_img] = resolved_b[r_mask] if r_mask.any() else torch.zeros(0, 5, device=img_ulb_w.device)
                                consist_conf_list[idx_img] = resolved_c[r_mask] if r_mask.any() else torch.zeros(0, device=img_ulb_w.device)
                                n_from_stu_total += n_stu2
                    else:
                        n_from_stu_total = 0

                else:
                    # ── Self-learning mode: student generates own pseudo labels ──
                    # Use low_thresh for NMS (same as teacher path), then filter
                    # by threshold_self_training as post-NMS confidence mask
                    # (mirrors origin: alt_param_threshold_self_training is ignore mask, not NMS filter)
                    model.eval()
                    stu_pseudo_b, stu_pseudo_c = yolo_get_pseudo_boxes(
                        model, img_ulb_w, conf_thresh=low_thresh, img_size=img_size)
                    model.train()
                    consist_pseudo, consist_conf_list = [], []
                    for sb, sc in zip(stu_pseudo_b, stu_pseudo_c):
                        if len(sc) > 0:
                            mask = sc >= threshold_self_training
                            consist_pseudo.append(sb[mask] if mask.any()
                                                  else torch.zeros(0, 5, device=img_ulb_w.device))
                            consist_conf_list.append(sc[mask] if mask.any()
                                                     else torch.zeros(0, device=img_ulb_w.device))
                        else:
                            consist_pseudo.append(torch.zeros(0, 5, device=img_ulb_w.device))
                            consist_conf_list.append(torch.zeros(0, device=img_ulb_w.device))
                    conflict_pseudo = [torch.zeros(0, 5, device=img_ulb_w.device) for _ in range(num_ulb)]
                    conflict_conf_list = [torch.zeros(0, device=img_ulb_w.device) for _ in range(num_ulb)]
                    conflict_src_list = [torch.zeros(0, dtype=torch.long, device=img_ulb_w.device) for _ in range(num_ulb)]
                    conflict_ratio = torch.tensor(0.0, device=img_ulb_w.device)
                    n_from_stu_total = 0

            # ── 5c. CutMix (AD-MT style) ─────────────────────────
            # Self-learning: CutMix on img_ulb_s directly (origin: "strongest augs")
            if flag_start_self_learning:
                if np.random.random() < args.get("cutmix_prob", 1.0):
                    all_pseudo = [
                        torch.cat([c, x], 0) if len(c) + len(x) > 0
                        else torch.zeros(0, 5, device=img_ulb_w.device)
                        for c, x in zip(consist_pseudo, conflict_pseudo)
                    ]
                    img_ulb_s, consist_pseudo = detection_cutmix(
                        img_ulb_s, all_pseudo, img_ulb_s, all_pseudo)
                    conflict_pseudo = [torch.zeros(0, 5, device=img_ulb_w.device)
                                       for _ in consist_pseudo]
            # Normal AD-MT: CutMix on the non-active teacher's turn
            elif not rpa.get_alternate_state():
                del img_ulb_s
                if np.random.random() < args.get("cutmix_prob", 1.0):
                    all_pseudo = [
                        torch.cat([c, x], 0) if len(c) + len(x) > 0
                        else torch.zeros(0, 5, device=img_ulb_w.device)
                        for c, x in zip(consist_pseudo, conflict_pseudo)
                    ]
                    img_ulb_s, consist_pseudo = detection_cutmix(
                        img_ulb_w, all_pseudo, img_ulb_w, all_pseudo)
                    conflict_pseudo = [torch.zeros(0, 5, device=img_ulb_w.device)
                                       for _ in consist_pseudo]
                else:
                    img_ulb_s = img_ulb_w.clone()

            # Recompute counts AFTER CutMix (CutMix merges conflict into consist)
            n_consist  = sum(len(b) for b in consist_pseudo)
            n_conflict = sum(len(b) for b in conflict_pseudo)
            # Update conflict_ratio to reflect post-CutMix state
            if n_consist + n_conflict > 0:
                conflict_ratio = torch.tensor(
                    n_conflict / (n_consist + n_conflict),
                    device=img_ulb_w.device, dtype=torch.float32)
            high_ratio = torch.tensor(
                sum(1 for b in consist_pseudo if len(b) > 0) / max(num_ulb, 1),
                dtype=torch.float32)

            # ── 5d. Student forward ──────────────────────────────
            model.train()
            with torch.cuda.amp.autocast(enabled=use_amp):
                img_all = torch.cat([img_lb, img_ulb_s], dim=0)
                preds_all = model(img_all)
                preds_lb    = [p[:num_lb]  for p in preds_all]
                preds_ulb_s = [p[num_lb:]  for p in preds_all]

                # Supervised loss
                batch_lb = boxes_list_to_batch(target_lb, img_lb)
                loss_lb, _ = yolo_detection_loss_with_grad(model, preds_lb, batch_lb)

                # Decide UNIFIED vs SPLIT path (mirrors origin Sec 3.3)
                unify_path = (abs(conflict_weight - 1.0) < 1e-8) and (conflict_weight != 0)

                # Unsupervised: consistent loss (split path only)
                if not unify_path and n_consist > 0:
                    batch_c = boxes_list_to_batch(consist_pseudo, img_ulb_s)
                    loss_ulb_consist, _ = yolo_detection_loss_with_grad(
                        model, preds_ulb_s, batch_c)

                # Unsupervised: conflict loss (split path only)
                if not unify_path and n_conflict > 0 and conflict_weight != 0:
                    batch_x = boxes_list_to_batch(conflict_pseudo, img_ulb_s)
                    loss_ulb_conflict, _ = yolo_detection_loss_with_grad(
                        model, preds_ulb_s, batch_x)

                # ── Unsupervised loss ─────────────────────────
                # Mirror origin (train_post_2d_aut.py:323):
                #   if conflict_weight == 1.0  → UNIFIED loss on (consist + conflict)
                #   else                       → SPLIT into L_consist + w * L_conflict
                # Unified path avoids running TaskAlignedAssigner twice with
                # disjoint GT sets (which could double-count anchor gradients).
                if unify_path:
                    # Merge consist + conflict into one GT set per image
                    merged = [
                        torch.cat([c, x], 0) if len(c) + len(x) > 0
                        else torch.zeros(0, 5, device=img_ulb_w.device)
                        for c, x in zip(consist_pseudo, conflict_pseudo)
                    ]
                    n_merged = sum(len(b) for b in merged)
                    if n_merged > 0:
                        batch_u = boxes_list_to_batch(merged, img_ulb_s)
                        loss_ulb_unified, _ = yolo_detection_loss_with_grad(
                            model, preds_ulb_s, batch_u)
                    else:
                        loss_ulb_unified = torch.tensor(0.0, device=img_ulb_s.device)
                    # Track split parts for logging (re-use unified value split
                    # by ratio of GT counts as a rough estimate)
                    if n_consist + n_conflict > 0:
                        ratio_c = n_consist / (n_consist + n_conflict)
                        loss_ulb_consist  = loss_ulb_unified.detach() * ratio_c
                        loss_ulb_conflict = loss_ulb_unified.detach() * (1.0 - ratio_c)

                    # Cap unified loss with same 1.5x policy
                    loss_ulb_cap = loss_lb.detach() * 1.5
                    if loss_ulb_unified.item() > loss_ulb_cap.item() > 0:
                        loss_ulb_unified = loss_ulb_unified * (loss_ulb_cap / loss_ulb_unified.detach())
                    loss_ulb = loss_ulb_unified
                else:
                    # SPLIT path (weight != 1.0): match origin's split branch
                    # Cap consist
                    loss_ulb_cap = loss_lb.detach() * 1.5
                    if loss_ulb_consist.item() > loss_ulb_cap.item() > 0:
                        loss_ulb_consist = loss_ulb_consist * (loss_ulb_cap / loss_ulb_consist.detach())

                    if conflict_weight != 0 and n_conflict > 0 and loss_ulb_conflict.item() > 0:
                        # Cap conflict part separately with a more lenient limit
                        conflict_cap = loss_lb.detach() * 2.0
                        if loss_ulb_conflict.item() > conflict_cap.item() > 0:
                            loss_ulb_conflict = loss_ulb_conflict * (conflict_cap / loss_ulb_conflict.detach())
                        loss_ulb = loss_ulb_consist + conflict_weight * loss_ulb_conflict
                    else:
                        loss_ulb = loss_ulb_consist * 1.0

                # Total loss with consistency ramp-up
                cons_w = get_current_consistency_weight(iter_num // 150, args)
                loss = loss_lb + cons_w * loss_ulb

            # Ensure grad graph exists
            if not loss.requires_grad:
                loss = loss + 0.0 * preds_all[0].sum()

            # ── 5e. Optimize student ─────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ── 5f. EMA update (RPA: alternate teacher) ──────────
            if not flag_start_self_learning:
                prev_state = rpa.get_alternate_state()
                if rpa.get_alternate_state():
                    update_ema_variables(model, ema_model,
                                         args["ema_decay"], iter_num // 2, args)
                else:
                    update_ema_variables(model, ema_model_another,
                                         args["ema_decay"], iter_num // 2, args)
                rpa.update()
                if rpa.get_alternate_state() != prev_state:
                    m_rpa_switches += 1
            else:
                # Self-learning: only update one EMA teacher (no alternation)
                prev_state = True
                update_ema_variables(model, ema_model,
                                     args["ema_decay"], iter_num, args)

            # ── 5g. Poly LR schedule ────────────────────────────
            if args.get("poly", True):
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_
            else:
                lr_ = base_lr

            # ── 5h. Logging ──────────────────────────────────────
            iter_num += 1
            t_elapsed = time.time() - t_start
            m_loss_lb.update(loss_lb.item())
            m_loss_ulb.update(loss_ulb.item())
            m_loss_all.update(loss.item())
            m_lr.update(lr_)
            m_high_r.update(high_ratio.item())
            m_conflict.update(conflict_ratio.item())
            # Track average pseudo-box count per image (different from high_ratio
            # which is the fraction of images with ≥1 box)
            m_pseudo_r.update((n_consist + n_conflict) / max(num_ulb, 1))
            m_loss_consist.update(loss_ulb_consist.item())
            m_loss_conflict.update(loss_ulb_conflict.item())
            m_n_from_stu.update(n_from_stu_total)

            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/loss_lb", loss_lb.item(), iter_num)
            writer.add_scalar("info/loss_ulb", loss_ulb.item(), iter_num)
            writer.add_scalar("info/loss_consist", loss_ulb_consist.item(), iter_num)
            writer.add_scalar("info/loss_conflict", loss_ulb_conflict.item(), iter_num)
            writer.add_scalar("info/conflict_ratio", conflict_ratio.item(), iter_num)
            writer.add_scalar("info/n_consist", n_consist, iter_num)
            writer.add_scalar("info/n_conflict", n_conflict, iter_num)
            writer.add_scalar("info/n_from_student", n_from_stu_total, iter_num)
            writer.add_scalar("info/active_teacher", 1 if prev_state else 2, iter_num)

            # Periodic iMAS-style log (~8 times per epoch)
            if (step + 1) % log_interval == 0 or step == iters_per_epoch - 1:
                print(
                    f"Epoch/Iter [{max_epoch}:  {step + 1}/{iters_per_epoch}]  "
                    f"AvgHard:{m_high_r.avg:.3f}  ConfW:{cons_w:.3f}  "
                    f"Sup:{loss_lb.item():.3f}({m_loss_lb.avg:.3f})  "
                    f"Uns:{loss_ulb.item():.3f}({m_loss_ulb.avg:.3f})  "
                    f"Pseudo:{high_ratio.item():.3f}({m_pseudo_r.avg:.3f})  "
                    f"Time:{t_elapsed / (step + 1):.2f}  LR:{lr_:.5f}")

            # Full detail to log file only
            logging.info(
                f"iter:{iter_num}  loss:{loss.item():.4f} "
                f"lb:{loss_lb.item():.4f} ulb:{loss_ulb.item():.4f} "
                f"consist:{loss_ulb_consist.item():.4f} "
                f"conflict:{loss_ulb_conflict.item():.4f} "
                f"w:{cons_w:.3f} high_r:{high_ratio.item():.2f} "
                f"cr:{conflict_ratio.item():.2f} lr:{lr_:.5f} "
                f"alt:{prev_state} conf:{cur_conf:.3f} "
                f"n_c/n_x:{n_consist}/{n_conflict} "
                f"stu_arb:{n_from_stu_total}")

            # CSV
            row = {"loss": loss.item(), "loss_lb": loss_lb.item(),
                   "loss_ulb": loss_ulb.item(),
                   "loss_consist": loss_ulb_consist.item(),
                   "loss_conflict": loss_ulb_conflict.item(),
                   "cons_w": cons_w, "high_r": high_ratio.item(),
                   "conflict_r": conflict_ratio.item(), "lr": lr_,
                   "n_consist": n_consist, "n_conflict": n_conflict,
                   "n_from_student": n_from_stu_total}
            pd.DataFrame(row, index=[iter_num]).to_csv(
                csv_train, mode="a",
                header=not (iter_num > 1 and osp.exists(csv_train)),
                index_label="iter")

            t_start = time.time()  # reset for next step timing

            if iter_num >= max_iterations:
                break

        # ── 6. Validation ────────────────────────────────────────
        if epoch_num % args.get("test_interval_ep", 1) == 0 or iter_num >= max_iterations:
            val_bs = args.get("batch_size_lb", 4)
            _rp = args.get("root_path")
            met_s  = compute_val_map(model, args["model"], dataset_yaml,
                                     img_size, val_bs, snapshot_path, _rp)
            met_t  = compute_val_map(ema_model, args["model"], dataset_yaml,
                                     img_size, val_bs, snapshot_path, _rp)
            if not flag_start_self_learning:
                met_t2 = compute_val_map(ema_model_another, args["model"], dataset_yaml,
                                         img_size, val_bs, snapshot_path, _rp)
            else:
                met_t2 = {"mAP50": -1.0, "mAP50_95": -1.0, "P": -1.0, "R": -1.0}

            if met_s["mAP50"] > best_map50_stu:
                best_map50_stu = met_s["mAP50"]
                os.makedirs(osp.join(snapshot_path, "student"), exist_ok=True)
                save_yolo_pt(model,
                             osp.join(snapshot_path, "best_stu_model.pt"),
                             args["model"])
            if met_t["mAP50"] > best_map50_tea:
                best_map50_tea = met_t["mAP50"]
                os.makedirs(osp.join(snapshot_path, "teacher"), exist_ok=True)
                save_yolo_pt(ema_model,
                             osp.join(snapshot_path, "best_tea_model.pt"),
                             args["model"])
            if met_t2["mAP50"] > best_map50_tea2:
                best_map50_tea2 = met_t2["mAP50"]

            logging.info(
                f"Epoch/Done [{epoch_num + 1}/{max_epoch}]  "
                f"Sup:{m_loss_lb.avg:.4f}  Uns:{m_loss_ulb.avg:.4f}  "
                f"Total:{m_loss_all.avg:.4f}  "
                f"Pseudo:{m_pseudo_r.avg:.4f}  "
                f"Val_mAP50:{met_t['mAP50']:.4f}  Best_mAP50:{best_map50_tea:.4f}")
            print(
                f"Epoch/Done [{epoch_num + 1}/{max_epoch}]  "
                f"Sup:{m_loss_lb.avg:.4f}  Uns:{m_loss_ulb.avg:.4f}  "
                f"Total:{m_loss_all.avg:.4f}  "
                f"Pseudo:{m_pseudo_r.avg:.4f}  "
                f"Val_mAP50:{met_t['mAP50']:.4f}  Best_mAP50:{best_map50_tea:.4f}")
            logging.info(
                f" <<Detail>> S: mAP50={met_s['mAP50']:.4f} "
                f"P={met_s['P']:.4f} R={met_s['R']:.4f} mAP50-95={met_s['mAP50_95']:.4f} | "
                f"T: mAP50={met_t['mAP50']:.4f} | T2: mAP50={met_t2['mAP50']:.4f}")

            row_val = {"mAP50_stu": met_s["mAP50"], "mAP50_stu_best": best_map50_stu,
                       "mAP50_95_stu": met_s["mAP50_95"],
                       "P_stu": met_s["P"], "R_stu": met_s["R"],
                       "mAP50_tea": met_t["mAP50"], "mAP50_tea_best": best_map50_tea,
                       "mAP50_tea2": met_t2["mAP50"], "mAP50_tea2_best": best_map50_tea2,
                       "loss_avg": m_loss_all.avg}
            pd.DataFrame(row_val, index=[epoch_num]).to_csv(
                csv_test, mode="a",
                header=not (epoch_num > 0 and osp.exists(csv_test)),
                index_label="epoch")

            writer.add_scalar("val/mAP50_stu", met_s["mAP50"], iter_num)
            writer.add_scalar("val/mAP50_tea", met_t["mAP50"], iter_num)
            writer.add_scalar("val/mAP50_tea2", met_t2["mAP50"], iter_num)
            writer.add_scalar("val/mAP50_95_stu", met_s["mAP50_95"], iter_num)

            # ── AD-MT epoch summary: RPA + CCM health ──
            writer.add_scalar("admt/rpa_switches", m_rpa_switches, iter_num)
            writer.add_scalar("admt/avg_conflict_ratio", m_conflict.avg, iter_num)
            writer.add_scalar("admt/avg_loss_consist", m_loss_consist.avg, iter_num)
            writer.add_scalar("admt/avg_loss_conflict", m_loss_conflict.avg, iter_num)
            writer.add_scalar("admt/avg_n_from_student", m_n_from_stu.avg, iter_num)
            logging.info(
                f" <<AD-MT>> RPA_switches:{m_rpa_switches}  "
                f"AvgConflictRatio:{m_conflict.avg:.4f}  "
                f"AvgConsistLoss:{m_loss_consist.avg:.4f}  "
                f"AvgConflictLoss:{m_loss_conflict.avg:.4f}  "
                f"AvgStuArbitration:{m_n_from_stu.avg:.1f}")

            # CCM health warning
            if conflict_weight > 0 and m_conflict.avg < 0.01 and epoch_num > 5:
                logging.warning(
                    f"[CCM WARNING] conflict_weight={conflict_weight} but avg conflict_ratio "
                    f"is near 0 ({m_conflict.avg:.4f}). CCM may not be effective. "
                    f"Consider checking teacher diversity or lowering match_iou.")

            # ── Save last_ckpt.pt for auto-resume (overwrites each val) ──
            try:
                torch.save({
                    "iter_num":        iter_num,
                    "epoch_num":       epoch_num,
                    "model":           model.state_dict(),
                    "ema":             ema_model.state_dict(),
                    "ema_another":     ema_model_another.state_dict(),
                    "optimizer":       optimizer.state_dict(),
                    "scaler":          scaler.state_dict() if scaler is not None else None,
                    "best_map50_stu":  best_map50_stu,
                    "best_map50_tea":  best_map50_tea,
                    "best_map50_tea2": best_map50_tea2,
                    "rpa_state": {
                        "counter": rpa._counter,
                        "flag":    rpa.flag_alternate,
                        "period":  rpa.random_alternate_period,
                    },
                }, osp.join(snapshot_path, "last_ckpt.pt"))
            except Exception as e:
                logging.warning(f"[RESUME] failed to save last_ckpt.pt: {e}")

            model.train();  ema_model.train();  ema_model_another.train()

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# ═══════════════════════════════════════════════════════════════════
#  III.  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AD-MT semi-supervised YOLO detection (iMAS-style data loading)")
    parser.add_argument("--cfg", type=str, required=True, help="YAML config file")
    parser.add_argument("--gpu_id", type=int, default=0)
    cli_args = parser.parse_args()

    # Load config — resolve path robustly regardless of cwd
    cfg_raw = cli_args.cfg
    candidates = [
        cfg_raw,                                      # absolute or already correct (preferred)
        osp.join("./exps", cfg_raw),                  # exps-relative
        osp.join("../exps", cfg_raw),                 # exps-relative from code/
        osp.join("./cfgs", cfg_raw),                  # legacy cfgs-relative
        osp.join("../cfgs", cfg_raw),                 # legacy from code/
    ]
    # Also strip leading "cfgs/" if user accidentally included it (legacy)
    if cfg_raw.startswith("cfgs/") or cfg_raw.startswith("cfgs\\"):
        stripped = cfg_raw[len("cfgs/"):]
        candidates += [osp.join("./cfgs", stripped), osp.join("../cfgs", stripped)]
    cfg_path = None
    for c in candidates:
        if osp.exists(c):
            cfg_path = c
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"Config not found. Tried: {candidates}")
    with open(cfg_path, "r") as f:
        args = yaml.safe_load(f)

    # GPU
    gid = cli_args.gpu_id if cli_args.gpu_id in range(10) else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

    # Determinism
    if args.get("deterministic", False):
        cudnn.benchmark = False;  cudnn.deterministic = True
    else:
        cudnn.benchmark = True;   cudnn.deterministic = False

    seed = args.get("seed", 2023)
    if seed > 0:
        random.seed(seed);  np.random.seed(seed)
        torch.manual_seed(seed);  torch.cuda.manual_seed(seed)

    # Output directory
    # AugSeg-style layout: config and results live together.
    # If the config file sits inside an "exps/" tree (e.g.
    #   exps/conf_001/bce/yolov11/config.yml), use its parent dir
    #   directly as snapshot_path. Otherwise fall back to legacy
    #   res_path/exp/model_tag.
    cfg_dir = osp.dirname(osp.abspath(cfg_path))
    if osp.sep + "exps" + osp.sep in cfg_dir + osp.sep or cfg_dir.endswith(osp.sep + "exps"):
        snapshot_path = cfg_dir
    else:
        # Legacy fallback (e.g. running cfgs/config_yolo_admt.yml directly)
        model_path = args["model"]
        model_tag = args.get("model_tag") \
            or osp.basename(osp.dirname(model_path)) \
            or osp.splitext(osp.basename(model_path))[0]
        snapshot_path = osp.join(
            args.get("res_path", "./results"),
            args.get("exp", "yolo_admt"),
            model_tag)
    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(osp.join(snapshot_path, "log"), exist_ok=True)

    # Logger
    logging.basicConfig(
        filename=osp.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(pprint.pformat(args))

    train(args, snapshot_path)
