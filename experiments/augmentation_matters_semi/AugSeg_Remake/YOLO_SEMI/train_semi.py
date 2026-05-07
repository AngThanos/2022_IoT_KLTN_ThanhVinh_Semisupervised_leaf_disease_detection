import argparse
import atexit
import yaml
import os
import os.path as osp
import pprint
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from augseg.dataset.builder import get_loader
from augseg.utils.dist_helper import setup_distributed
from augseg.utils.lr_helper import get_optimizer, get_scheduler
from augseg.utils.utils import AverageMeter, load_state, set_random_seed, setup_default_logging

from copy import deepcopy
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import torchvision.transforms.functional as TF_aug


# ============ Tensor-based TIBA (Paper Section 3.2) ============
# Applied AFTER ALIA in training loop so pasted patches get same intensity aug.

def _tiba_identity(img, scale):
    return img

def _tiba_autocontrast(img, scale):
    """Per-channel stretch to [0, 1]."""
    for c in range(img.shape[0]):
        ch = img[c]
        lo, hi = ch.min(), ch.max()
        if hi - lo > 1e-5:
            img[c] = (ch - lo) / (hi - lo)
    return img

def _tiba_blur(img, scale):
    sigma = np.random.uniform(scale[0], scale[1])
    ks = int(np.ceil(sigma * 3)) * 2 + 1
    ks = max(3, ks)
    return TF_aug.gaussian_blur(img.unsqueeze(0), kernel_size=ks, sigma=sigma).squeeze(0)

def _tiba_contrast(img, scale):
    v = np.random.uniform(scale[0], scale[1])
    return TF_aug.adjust_contrast(img, v)

def _tiba_brightness(img, scale):
    v = np.random.uniform(scale[0], scale[1])
    return TF_aug.adjust_brightness(img, v)

def _tiba_color(img, scale):
    """Saturation adjustment."""
    v = np.random.uniform(scale[0], scale[1])
    return TF_aug.adjust_saturation(img, v)

def _tiba_sharpness(img, scale):
    v = np.random.uniform(scale[0], scale[1])
    return TF_aug.adjust_sharpness(img, v)

def _tiba_posterize(img, scale):
    v = int(np.random.uniform(scale[0], scale[1]))
    v = max(1, min(8, v))
    # posterize needs uint8
    img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)
    img_uint8 = TF_aug.posterize(img_uint8, v)
    return img_uint8.float() / 255.0

def _tiba_solarize(img, scale):
    v = np.random.uniform(scale[0], scale[1])
    threshold = v / 256.0  # normalize to [0, 1] range
    return TF_aug.solarize(img, threshold)

def _tiba_equalize(img, scale):
    img_uint8 = (img.clamp(0, 1) * 255).to(torch.uint8)
    img_uint8 = TF_aug.equalize(img_uint8)
    return img_uint8.float() / 255.0

def _tiba_hue(img, scale):
    v = np.random.uniform(scale[0], scale[1])
    if np.random.random() < 0.5:
        v = -v
    return TF_aug.adjust_hue(img, v)


# Pool of 11 TIBA operations (same as PIL version in augmentations.py)
_TIBA_OPS = [
    (_tiba_identity, None),
    (_tiba_autocontrast, None),
    (_tiba_equalize, None),
    (_tiba_blur, [0.1, 2.0]),
    (_tiba_contrast, [0.05, 0.95]),
    (_tiba_brightness, [0.05, 0.95]),
    (_tiba_color, [0.05, 0.95]),
    (_tiba_sharpness, [0.05, 0.95]),
    (_tiba_posterize, [4, 8]),
    (_tiba_solarize, [1, 256]),
    (_tiba_hue, [0, 0.5]),
]

import random as _random_module

def apply_tiba_tensor(images, num_augs=3, flag_random_num=True):
    """Apply TIBA (Paper §3.2) on a batch of [0,1] tensors [B, C, H, W].
    
    Applied per-image with random ops + uniform continuous strength.
    Uses random.choices (WITH replacement) as per paper.
    """
    B = images.shape[0]
    result = images.clone()
    
    for i in range(B):
        img = result[i]  # [C, H, W], [0, 1]
        
        if flag_random_num:
            max_num = np.random.randint(1, num_augs + 1)
        else:
            max_num = num_augs
        
        ops = _random_module.choices(_TIBA_OPS, k=max_num)
        for op_fn, scales in ops:
            img = op_fn(img, scales)
        
        result[i] = img.clamp(0, 1)
    
    return result


def _clone_preds(preds):
    """Clone YOLO prediction tensors to prevent inplace mutations inside loss()
    from corrupting autograd graphs that share the same storage."""
    if isinstance(preds, torch.Tensor):
        return preds.clone()
    if isinstance(preds, (list, tuple)):
        cloned = [_clone_preds(p) for p in preds]
        return type(preds)(cloned)
    if isinstance(preds, dict):
        return {k: _clone_preds(v) for k, v in preds.items()}
    return preds


def _safe_yolo_loss(yolo_model, batch, preds):
    """Compute YOLO loss with an isolated criterion instance.

    Ultralytics criterion keeps mutable internal tensors that can be overwritten
    across consecutive loss() calls before a single backward(). Re-initializing
    criterion for each call avoids inplace-version conflicts while preserving a
    single total loss backward.
    """
    if hasattr(yolo_model, "init_criterion"):
        yolo_model.criterion = yolo_model.init_criterion()
    return yolo_model.loss(batch, _clone_preds(preds))


def sigmoid_rampup(current, rampup_length):
    """Exponential ramp-up from Mean Teacher to stabilize early unsupervised training."""
    # Ramp-up giúp nhánh unsupervised tăng dần ảnh hưởng,
    # tránh làm model nhiễu ở giai đoạn đầu khi pseudo-label còn kém chất lượng.
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


def apply_nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Apply Non-Maximum Suppression to YOLO predictions.
    
    Args:
        prediction: Raw model output tensor, typically [batch, 4+nc, num_boxes]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of detections per image, each [N, 6] (x1, y1, x2, y2, conf, cls)
    """
    try:
        # Ultralytics >= 8.3 moved NMS to ultralytics.utils.nms.
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        # Backward compatibility for older Ultralytics releases.
        from ultralytics.utils.ops import non_max_suppression

    # Ultralytics models can return tuple/list during inference.
    # Chuẩn hóa output để các bước sau luôn xử lý 1 tensor dự đoán chính.
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    output = non_max_suppression(
        prediction,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    # Keep a stable [N, 6] tensor format for downstream code.
    # Mỗi detection có dạng: [x1, y1, x2, y2, conf, cls].
    normalized = []
    for det in output:
        if det is None or det.shape[0] == 0:
            normalized.append(torch.zeros((0, 6), dtype=torch.float32, device=prediction.device))
        else:
            normalized.append(det[:, :6])

    return normalized


def compute_ap(recall, precision):
    """Compute Average Precision using all-point interpolation (VOC 2010+)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # AP = sum of rectangular areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute Average Precision for each class.

    Args:
        tp: np.ndarray [N, T] - True positive flags at T IoU thresholds
        conf: np.ndarray [N] - Confidence scores
        pred_cls: np.ndarray [N] - Predicted class indices
        target_cls: np.ndarray [M] - Ground truth class indices

    Returns:
        p: np.ndarray [C] - Precision per class (at max-F1 point, IoU=0.5)
        r: np.ndarray [C] - Recall per class (at max-F1 point, IoU=0.5)
        ap: np.ndarray [C, T] - AP per class at each IoU threshold
        unique_classes: np.ndarray [C] - Unique class indices
    """
    # Sort by confidence descending
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls).astype(int)
    nc = len(unique_classes)
    n_thres = tp.shape[1] if tp.ndim == 2 else 1

    ap = np.zeros((nc, n_thres))
    p_class = np.zeros(nc)
    r_class = np.zeros(nc)

    for ci, c in enumerate(unique_classes):
        pred_mask = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_pred = pred_mask.sum()

        if n_pred == 0 or n_gt == 0:
            continue

        tp_c = tp[pred_mask]

        for ti in range(n_thres):
            tpc = tp_c[:, ti].astype(float)
            fpc = 1.0 - tpc

            tp_cum = np.cumsum(tpc)
            fp_cum = np.cumsum(fpc)

            recall = tp_cum / (n_gt + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)

            ap[ci, ti] = compute_ap(recall, precision)

            # At IoU=0.5 (first threshold), store P/R at max-F1 operating point
            if ti == 0:
                f1 = 2 * precision * recall / (precision + recall + 1e-16)
                if len(f1) > 0:
                    idx = np.argmax(f1)
                    p_class[ci] = precision[idx]
                    r_class[ci] = recall[idx]

    return p_class, r_class, ap, unique_classes


def setup_yolo_model(pretrain_path):
    """Initialize YOLO model with proper hyperparameters."""
    # Lấy kiến trúc + trọng số nền từ Ultralytics.
    yolo_wrapper = YOLO(pretrain_path)
    model = yolo_wrapper.model
    
    # Set args as SimpleNamespace BEFORE init_criterion
    # Các trọng số loss cơ bản (box/cls/dfl) cho criterion của YOLO.
    default_args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.args = default_args
    
    # Initialize loss criterion
    if hasattr(model, 'init_criterion'):
        model.init_criterion()
    
    return model


def save_yolo_pt(model, save_path, train_args=None):
    """Save model in Ultralytics-compatible .pt format.

    The saved file can be loaded directly with ``YOLO('path.pt')``.
    """
    from ultralytics import __version__ as ul_version

    yolo_model = model.module if hasattr(model, 'module') else model
    ckpt = {
        'model': deepcopy(yolo_model).half(),
        'date': datetime.now().isoformat(),
        'version': ul_version,
        'license': 'AGPL-3.0 (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com',
    }
    if train_args is not None:
        ckpt['train_args'] = train_args
    torch.save(ckpt, save_path)


def build_pseudo_batch(
    teacher_preds,
    images,
    conf_thres=0.7,
    iou_thres=0.5,
    max_det=300,
    class_conf_thres=None,
    score_weighting=False,
    score_power=1.0,
    topk_per_image=0,
):
    """Build YOLO-format pseudo labels from teacher detections."""
    # Teacher dự đoán trên ảnh weak -> NMS -> lọc threshold -> convert về YOLO xywh normalized.
    detections = apply_nms(
        teacher_preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    batch_idx, cls_list, box_list = [], [], []
    score_list = []
    _, _, img_h, img_w = images.shape
    eps = 1e-6
    valid_images = 0

    for bi, det in enumerate(detections):
        if det is None or det.shape[0] == 0:
            continue

        if class_conf_thres:
            # Cho phép threshold theo từng lớp để giảm bias lớp khó/dễ.
            cls_ids = det[:, 5].long()
            thrs = torch.full_like(det[:, 4], float(conf_thres))
            for cls_id, cls_thr in class_conf_thres.items():
                thrs[cls_ids == int(cls_id)] = float(cls_thr)
            det = det[det[:, 4] >= thrs]
        else:
            det = det[det[:, 4] >= float(conf_thres)]

        if det.shape[0] == 0:
            continue

        # Keep only top-k pseudo boxes per image to reduce noisy supervision density.
        # Giới hạn mật độ pseudo để tránh unsupervised loss bị chi phối bởi ảnh quá nhiều box.
        if int(topk_per_image) > 0 and det.shape[0] > int(topk_per_image):
            topk_idx = torch.topk(det[:, 4], k=int(topk_per_image), largest=True).indices
            det = det[topk_idx]

        x1 = det[:, 0].clamp(0, img_w)
        y1 = det[:, 1].clamp(0, img_h)
        x2 = det[:, 2].clamp(0, img_w)
        y2 = det[:, 3].clamp(0, img_h)
        w = (x2 - x1).clamp(min=eps)
        h = (y2 - y1).clamp(min=eps)
        cx = x1 + w / 2
        cy = y1 + h / 2

        # Normalize to YOLO xywh format expected by loss()
        # loss của YOLO nhận box chuẩn hóa theo kích thước ảnh hiện tại.
        boxes_xywh = torch.stack([
            cx / float(img_w),
            cy / float(img_h),
            w / float(img_w),
            h / float(img_h),
        ], dim=1).clamp(0, 1)

        n = boxes_xywh.shape[0]
        if n == 0:
            continue

        valid_images += 1
        batch_idx.append(torch.full((n,), bi, dtype=torch.long, device=images.device))
        cls_list.append(det[:, 5:6].to(images.device).float())
        box_list.append(boxes_xywh.to(images.device).float())
        score_list.append(det[:, 4].to(images.device).float())

    if len(batch_idx) == 0:
        empty = {
            'img': images,
            'batch_idx': torch.zeros((0,), dtype=torch.long, device=images.device),
            'cls': torch.zeros((0, 1), dtype=torch.float32, device=images.device),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32, device=images.device),
        }
        return empty, 0.0, 0.0, 0.0

    pseudo_batch = {
        'img': images,
        'batch_idx': torch.cat(batch_idx, dim=0),
        'cls': torch.cat(cls_list, dim=0),
        'bboxes': torch.cat(box_list, dim=0),
    }
    pseudo_ratio = valid_images / float(images.shape[0])
    # Số pseudo box trung bình mỗi ảnh, dùng để theo dõi độ "dày" của pseudo supervision.
    pseudo_boxes_per_img = float(pseudo_batch['batch_idx'].numel()) / float(images.shape[0])
    if score_weighting:
        all_scores = torch.cat(score_list, dim=0)
        pseudo_weight = float((all_scores.clamp(0, 1) ** float(score_power)).mean().item())
    else:
        pseudo_weight = 1.0
    return pseudo_batch, pseudo_ratio, pseudo_weight, pseudo_boxes_per_img


def _prediction_tensor_list(prediction):
    """Normalize YOLO raw prediction structure into a list of tensors."""
    # Một số phiên bản/head YOLO trả tuple/list nhiều mức đặc trưng.
    # Hàm này gom về list tensor để tính consistency thống nhất.
    if isinstance(prediction, torch.Tensor):
        return [prediction]
    if isinstance(prediction, (list, tuple)):
        tensors = [x for x in prediction if isinstance(x, torch.Tensor)]
        if len(tensors) == 0 and len(prediction) > 0 and isinstance(prediction[0], (list, tuple)):
            tensors = [x for x in prediction[0] if isinstance(x, torch.Tensor)]
        return tensors
    return []


def compute_consistency_loss(student_preds, teacher_preds, mode="mse", temperature=1.0):
    """
    Soft consistency between student(strong) and teacher(weak) raw predictions.
    This assumes geometry-preserving strong augmentations (photometric transforms).
    """
    # Lưu ý: consistency hiện tính trên raw tensor dự đoán.
    # Cách này thực dụng nhưng chưa semantic-aware bằng decode+matching theo box.
    stu_list = _prediction_tensor_list(student_preds)
    tea_list = _prediction_tensor_list(teacher_preds)

    if len(stu_list) == 0 or len(tea_list) == 0:
        device = None
        if len(stu_list) > 0:
            device = stu_list[0].device
        elif len(tea_list) > 0:
            device = tea_list[0].device
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    total = 0.0
    count = 0
    for s, t in zip(stu_list, tea_list):
        if s.shape != t.shape:
            continue

        if mode == "kl":
            # KL với temperature để làm mềm phân phối teacher.
            s_log_prob = torch.log_softmax(s / temperature, dim=1)
            t_prob = torch.softmax(t.detach() / temperature, dim=1)
            loss = torch.nn.functional.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)
        else:
            # MSE trên sigmoid output là lựa chọn ổn định mặc định.
            loss = torch.nn.functional.mse_loss(torch.sigmoid(s), torch.sigmoid(t.detach()))

        total = total + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=stu_list[0].device)
    return total / float(count)


# ============ AugSeg ALIA helpers for detection ============

def _det_to_xywhn(det, img_w, img_h):
    """Convert NMS detections [x1,y1,x2,y2,conf,cls] to [cls, cx, cy, w, h] normalized."""
    if det is None or det.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    det_np = det.detach().cpu().numpy() if isinstance(det, torch.Tensor) else det
    x1, y1, x2, y2 = det_np[:, 0], det_np[:, 1], det_np[:, 2], det_np[:, 3]
    cls = det_np[:, 5]
    w = np.clip(x2 - x1, 1e-6, img_w)
    h = np.clip(y2 - y1, 1e-6, img_h)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    out = np.zeros((len(cls), 5), dtype=np.float32)
    out[:, 0] = cls
    out[:, 1] = np.clip(cx / float(img_w), 0, 1)
    out[:, 2] = np.clip(cy / float(img_h), 0, 1)
    out[:, 3] = np.clip(w / float(img_w), 0, 1)
    out[:, 4] = np.clip(h / float(img_h), 0, 1)
    return out


def _extract_per_image_targets(label_tensor, batch_size):
    """Extract per-image targets from collated label tensor [N, 6] → list of [M, 5]."""
    targets = []
    if len(label_tensor) == 0:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(batch_size)]
    for bi in range(batch_size):
        mask = label_tensor[:, 0].long() == bi
        if mask.any():
            t = label_tensor[mask, 1:].detach().cpu().numpy()  # [cls, cx, cy, w, h]
            targets.append(t.astype(np.float32))
        else:
            targets.append(np.zeros((0, 5), dtype=np.float32))
    return targets


def _reassign_boxes_det(boxes_keep, boxes_paste, nx1, ny1, nx2, ny2):
    """Keep boxes_keep with center OUTSIDE cut region.
    Add boxes_paste with center INSIDE cut region."""
    parts = []
    if len(boxes_keep) > 0:
        cx, cy = boxes_keep[:, 1], boxes_keep[:, 2]
        outside = ~((cx >= nx1) & (cx <= nx2) & (cy >= ny1) & (cy <= ny2))
        if outside.any():
            parts.append(boxes_keep[outside])
    if len(boxes_paste) > 0:
        cx, cy = boxes_paste[:, 1], boxes_paste[:, 2]
        inside = (cx >= nx1) & (cx <= nx2) & (cy >= ny1) & (cy <= ny2)
        if inside.any():
            parts.append(boxes_paste[inside])
    if parts:
        return np.concatenate(parts, axis=0)
    return np.zeros((0, 5), dtype=np.float32)


def cut_mix_label_adaptive_det(
    unlabeled_images,    # [B, C, H, W] strong-augmented tensor (cuda)
    unlabeled_pseudo,    # list[np.ndarray] per-image pseudo boxes [N, 5] (cls,cx,cy,w,h)
    labeled_images,      # [B, C, H, W] tensor (cuda)
    labeled_targets,     # list[np.ndarray] per-image GT boxes [N, 5]
    lst_confidences,     # list[float] per-image confidence scores ρ_i
):
    """AugSeg ALIA (Section 3.3) adapted for detection.
    
    Paper Eq.7-9:
    Step 1: if random() > ρ_i → inject labeled patch into unlabeled image i
    Step 2: copy-paste between unlabeled images (permuted)
    
    The core idea: less confident unlabeled samples are more likely to be 
    aided (mixed) by confident labeled samples.
    """
    import random as _random
    
    B, C, H, W = unlabeled_images.shape
    mix_images = unlabeled_images.clone()
    mix_pseudo = [p.copy() if len(p) > 0 else np.zeros((0, 5), dtype=np.float32)
                  for p in unlabeled_pseudo]

    # Random permutation for step 2
    u_rand_index = torch.randperm(B).tolist()

    # Paper: beta(8,2) for labeled patch (tends smaller), beta(4,4) for unlabeled
    lam_l = np.random.beta(8, 2)
    lam_u = np.random.beta(4, 4)

    cut_l = np.sqrt(1.0 - lam_l)
    cut_u = np.sqrt(1.0 - lam_u)
    cut_w_l, cut_h_l = int(W * cut_l), int(H * cut_l)
    cut_w_u, cut_h_u = int(W * cut_u), int(H * cut_u)

    # --- Step 1: Labeled injection (adaptive, Paper Eq.8) ---
    # "less confident → more likely to be aided by labeled samples"
    for i in range(B):
        if _random.random() > lst_confidences[i]:
            # Trigger: confidence low → inject labeled patch
            j_l = u_rand_index[i] % len(labeled_images)

            cx = _random.randint(max(1, W // 8), max(2, W - 1))
            cy = _random.randint(max(1, H // 8), max(2, H - 1))
            x1 = max(0, cx - cut_w_l // 2)
            y1 = max(0, cy - cut_h_l // 2)
            x2 = min(W, cx + cut_w_l // 2)
            y2 = min(H, cy + cut_h_l // 2)

            # Paste labeled patch onto unlabeled
            mix_images[i, :, y1:y2, x1:x2] = labeled_images[j_l, :, y1:y2, x1:x2]

            # Reassign boxes based on center location
            mix_pseudo[i] = _reassign_boxes_det(
                mix_pseudo[i], labeled_targets[j_l],
                x1 / W, y1 / H, x2 / W, y2 / H
            )

    # --- Step 2: Unlabeled-Unlabeled copy-paste (Paper Eq.9) ---
    # Eq.9: A_a(u_m) = M_m ⊙ u_m + (1-M_m) ⊙ u_n'
    # Base = ORIGINAL unlabeled (u_m), paste FROM mix candidates (u_n')
    final_images = unlabeled_images.clone()
    final_pseudo = [p.copy() if len(p) > 0 else np.zeros((0, 5), dtype=np.float32)
                    for p in unlabeled_pseudo]

    for i in range(B):
        j = u_rand_index[i]
        cx = _random.randint(max(1, W // 8), max(2, W - 1))
        cy = _random.randint(max(1, H // 8), max(2, H - 1))
        x1 = max(0, cx - cut_w_u // 2)
        y1 = max(0, cy - cut_h_u // 2)
        x2 = min(W, cx + cut_w_u // 2)
        y2 = min(H, cy + cut_h_u // 2)

        # Paste from mix_images[permuted] (step 1 result) into ORIGINAL
        final_images[i, :, y1:y2, x1:x2] = mix_images[j, :, y1:y2, x1:x2]
        final_pseudo[i] = _reassign_boxes_det(
            final_pseudo[i], mix_pseudo[j],
            x1 / W, y1 / H, x2 / W, y2 / H
        )

    return final_images, final_pseudo


def _build_pseudo_batch_from_list(pseudo_per_img, images):
    """Build YOLO-format batch dict from list of per-image pseudo boxes."""
    batch_idx_list, cls_list, box_list = [], [], []
    B = images.shape[0]
    valid_images = 0

    for bi, boxes in enumerate(pseudo_per_img):
        if len(boxes) == 0:
            continue
        valid_images += 1
        n = len(boxes)
        batch_idx_list.append(torch.full((n,), bi, dtype=torch.long, device=images.device))
        cls_list.append(torch.from_numpy(boxes[:, 0:1]).float().to(images.device))
        box_list.append(torch.from_numpy(boxes[:, 1:5]).float().to(images.device))

    if len(batch_idx_list) == 0:
        empty = {
            'img': images,
            'batch_idx': torch.zeros((0,), dtype=torch.long, device=images.device),
            'cls': torch.zeros((0, 1), dtype=torch.float32, device=images.device),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32, device=images.device),
        }
        return empty, 0.0, 0.0

    pseudo_batch = {
        'img': images,
        'batch_idx': torch.cat(batch_idx_list, dim=0),
        'cls': torch.cat(cls_list, dim=0),
        'bboxes': torch.cat(box_list, dim=0),
    }
    pseudo_ratio = float(valid_images) / float(B)
    pseudo_boxes_per_img = float(pseudo_batch['batch_idx'].numel()) / float(B)
    return pseudo_batch, pseudo_ratio, pseudo_boxes_per_img


def main(args):
    if args.seed is not None:
        # Cố định seed để dễ tái lập kết quả.
        set_random_seed(args.seed, deterministic=True)
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, world_size = setup_distributed(port=args.port)

    def _cleanup_dist():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    atexit.register(_cleanup_dist)

    # 1. Output settings
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]
    
    if rank == 0:
        os.makedirs(cfg["log_path"], exist_ok=True)
        os.makedirs(cfg["save_path"], exist_ok=True)
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
        csv_path = os.path.join(cfg["log_path"], f"seg_{curr_timestr}_stat.csv")
        logger.info("{}".format(pprint.pformat(cfg)))
        tb_logger = SummaryWriter(osp.join(cfg["log_path"], "events_seg", curr_timestr)) if flag_use_tb else None
    else:
        logger, csv_path, tb_logger = None, None, None
    
    dist.barrier()

    # 2. Prepare student model
    model = setup_yolo_model(cfg["net"]["encoder"]["pretrain"])
    
    for param in model.parameters():
        param.requires_grad = True
    
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # 3. Data loaders (Step Augmentation handled inside dataset)
    # train_loader_sup: dữ liệu có nhãn
    # train_loader_unsup: dữ liệu không nhãn (weak/strong)
    # val_loader: dữ liệu validation
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

    # 4. Optimizer
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    optimizer = get_optimizer(model.parameters(), cfg_optim)

    # 5. DDP wrapper
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # 6. Teacher model (EMA)
    # Teacher là bản làm mượt tham số từ student bằng EMA.
    model_teacher = setup_yolo_model(cfg["net"]["encoder"]["pretrain"])
    model_teacher.cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # Initialize teacher with student weights
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data.copy_(s_params.data)

    # 7. Resume checkpoint
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1
    
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pt")
        # Backward compat: fallback to old .pth if .pt not found
        if not os.path.exists(lastest_model):
            lastest_model_legacy = os.path.join(cfg["save_path"], "ckpt.pth")
            if os.path.exists(lastest_model_legacy):
                lastest_model = lastest_model_legacy
        if os.path.exists(lastest_model):
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(lastest_model, model, optimizer=optimizer, key="model_state")
            load_state(lastest_model, model_teacher, optimizer=optimizer, key="teacher_state")

    lr_scheduler = get_scheduler(cfg_trainer, len(train_loader_sup), optimizer, start_epoch=last_epoch)
    print(f"====================== {len(train_loader_sup)} ==============")
    # 8. Training loop
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        res_loss_sup, res_loss_unsup = train(
            model, model_teacher, optimizer, lr_scheduler,
            train_loader_sup, train_loader_unsup,
            epoch, tb_logger, logger, cfg
        )

        # Validation
        if cfg_trainer.get("evaluate_student", True):
            metrics_stu = validate_yolo(model, val_loader, epoch, logger, cfg, prefix="STU")
        else:
            metrics_stu = {'Precision': 0.0, 'Recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        metrics_tea = validate_yolo(model_teacher, val_loader, epoch, logger, cfg, prefix="EMA")
        prec_stu = metrics_stu['mAP50']
        prec_tea = metrics_tea['mAP50']
        prec = prec_tea

        # Save checkpoint
        if rank == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
            }
            
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pt"))
                # YOLO-compatible best.pt (EMA teacher) — loadable via YOLO('best.pt')
                save_yolo_pt(model_teacher, osp.join(cfg["save_path"], "best.pt"))
                # YOLO-compatible best student
                save_yolo_pt(model, osp.join(cfg["save_path"], "best_student.pt"))

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pt"))
            # YOLO-compatible last.pt (EMA teacher)
            save_yolo_pt(model_teacher, osp.join(cfg["save_path"], "last.pt"))
            
            # Save statistics
            tmp_results = {
                'loss_lb': res_loss_sup,
                'loss_ub': res_loss_unsup,
                'Precision_stu': metrics_stu['Precision'],
                'Recall_stu': metrics_stu['Recall'],
                'mAP50_stu': metrics_stu['mAP50'],
                'mAP50-95_stu': metrics_stu['mAP50-95'],
                'Precision_tea': metrics_tea['Precision'],
                'Recall_tea': metrics_tea['Recall'],
                'mAP50_tea': metrics_tea['mAP50'],
                'mAP50-95_tea': metrics_tea['mAP50-95'],
                'best_mAP50': best_prec,
                'best_mAP50_stu': best_prec_stu,
            }
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch + 1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')
            
            logger.info(
                f" <<Test>> - Epoch: {epoch}  "
                f"STU[Precision:{metrics_stu['Precision']:.4f} Recall:{metrics_stu['Recall']:.4f} mAP50:{metrics_stu['mAP50']:.4f} mAP50-95:{metrics_stu['mAP50-95']:.4f}]  "
                f"EMA[Precision:{metrics_tea['Precision']:.4f} Recall:{metrics_tea['Recall']:.4f} mAP50:{metrics_tea['mAP50']:.4f} mAP50-95:{metrics_tea['mAP50-95']:.4f}]  "
                f"\033[34mBest-STU mAP50:{best_prec_stu:.4f}/{best_epoch_stu}  "
                f"\033[31mBest-EMA mAP50:{best_prec:.4f}/{best_epoch}\033[0m"
            )
            
            if tb_logger is not None:
                tb_logger.add_scalar("mAP50_tea", metrics_tea['mAP50'], epoch)
                tb_logger.add_scalar("mAP50-95_tea", metrics_tea['mAP50-95'], epoch)
                tb_logger.add_scalar("mAP50_stu", metrics_stu['mAP50'], epoch)
                tb_logger.add_scalar("mAP50-95_stu", metrics_stu['mAP50-95'], epoch)

    if rank == 0 and tb_logger is not None:
        tb_logger.close()
    _cleanup_dist()


def train(model, model_teacher, optimizer, lr_scheduler, loader_l, loader_u, epoch, tb_logger, logger, cfg):
    """AugSeg training loop for YOLO semi-supervised detection.
    
    Paper pipeline:
    1. Teacher(weak_view) → NMS → pseudo boxes + confidence ρ_i
    2. ALIA: adaptive label-injecting CutMix (confidence-gated)
    3. Student(augmented_view) → YOLO loss on pseudo labels
    4. Total: L = L_sup + λ_u × L_unsup  (NO consistency, NO rampup)
    """

    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    
    # Semi-supervised settings (AugSeg: simple)
    unsup_cfg = cfg["trainer"].get("unsupervised", {})
    loss_weight = unsup_cfg.get("loss_weight", 1.0)   # Paper: λ_u = 1.0
    # Threshold: static or dynamic (start→end over epochs)
    pseudo_conf_thres_start = unsup_cfg.get("threshold", 0.7)
    pseudo_conf_thres_end = unsup_cfg.get("threshold_end", pseudo_conf_thres_start)
    total_epochs = cfg["trainer"]["epochs"]
    # Linearly interpolate threshold for this epoch
    if total_epochs > 1 and pseudo_conf_thres_start != pseudo_conf_thres_end:
        t = epoch / float(total_epochs - 1)
        pseudo_conf_thres = pseudo_conf_thres_start + t * (pseudo_conf_thres_end - pseudo_conf_thres_start)
    else:
        pseudo_conf_thres = pseudo_conf_thres_start
    pseudo_iou_thres = unsup_cfg.get("nms_iou", 0.5)
    pseudo_max_det = unsup_cfg.get("max_det", 300)
    use_cutmix = unsup_cfg.get("use_cutmix", True)
    sup_only_epoch = cfg["trainer"].get("sup_only_epoch", 0)

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    
    assert len(loader_l) == len(loader_u), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, mismatch!"

    # Metrics
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    pseudo_ratios = AverageMeter(20)
    pseudo_boxes_per_img_meter = AverageMeter(20)
    
    # Print frequency
    print_freq = max(len(loader_u) // 8, 1)
    print_freq_lst = [i * print_freq for i in range(1, 8)] + [len(loader_u) - 1]

    model.train()
    model_teacher.eval()

    for step in range(len(loader_l)):
        batch_start = time.time()
        i_iter = epoch * len(loader_l) + step

        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        # Load data
        _, image_l, label_l = next(loader_l_iter)
        image_l, label_l = image_l.cuda(), label_l.cuda()
        # image_u_aug already has TIBA applied in dataset (same as AugSeg Origin)
        _, image_u_weak, image_u_aug, _ = next(loader_u_iter)
        image_u_weak, image_u_aug = image_u_weak.cuda(), image_u_aug.cuda()

        # Get underlying YOLO model
        yolo_model = model.module if hasattr(model, 'module') else model
        yolo_model.train()
        # Freeze BN: small dataset → BN stats would be biased (iMAS pattern)
        for m in yolo_model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                m.eval()

        optimizer.zero_grad(set_to_none=True)

        # ============ Supervised Loss ============
        pred_l = yolo_model(image_l)
        
        batch_l = {
            'img': image_l,
            'batch_idx': label_l[:, 0].long() if len(label_l) > 0 else torch.tensor([], device=image_l.device, dtype=torch.long),
            'cls': label_l[:, 1:2] if len(label_l) > 0 else torch.zeros((0, 1), device=image_l.device),
            'bboxes': label_l[:, 2:] if len(label_l) > 0 else torch.zeros((0, 4), device=image_l.device),
        }
        
        loss_items, _ = _safe_yolo_loss(yolo_model, batch_l, pred_l)
        sup_loss = loss_items.sum()

        # ============ Unsupervised Loss (AugSeg: pseudo-label only, NO consistency) ============
        unsup_loss = torch.tensor(0.0, device=image_l.device)
        pseudo_ratio = 0.0
        pseudo_boxes_per_img = 0.0
        
        if epoch >= sup_only_epoch:
            # 1. Teacher predict on weak view → pseudo boxes + confidence
            with torch.no_grad():
                model_teacher.eval()
                teacher_preds = model_teacher(image_u_weak)

            # NMS all detections at low threshold, then filter
            detections = apply_nms(teacher_preds, conf_thres=0.001,
                                   iou_thres=pseudo_iou_thres, max_det=pseudo_max_det)
            
            _, _, img_h, img_w = image_u_weak.shape
            
            # Compute per-image confidence ρ_i (Paper Eq.7 adapted for detection)
            # and pseudo boxes for each image
            lst_confidences = []
            pseudo_per_img = []
            for bi, det in enumerate(detections):
                if det is not None and det.shape[0] > 0:
                    # Filter by threshold
                    good = det[det[:, 4] >= pseudo_conf_thres]
                    if good.shape[0] > 0:
                        conf = float(good[:, 4].mean().item())
                        # Convert to [cls, cx, cy, w, h] normalized
                        boxes = _det_to_xywhn(good, img_w, img_h)
                    else:
                        conf = 0.0
                        boxes = np.zeros((0, 5), dtype=np.float32)
                else:
                    conf = 0.0
                    boxes = np.zeros((0, 5), dtype=np.float32)
                lst_confidences.append(conf)
                pseudo_per_img.append(boxes)

            # 2. ALIA: Adaptive Label-Injecting CutMix (Paper Section 3.3)
            if use_cutmix:
                # Extract per-image labeled targets from collated label tensor
                labeled_targets = _extract_per_image_targets(label_l, image_l.shape[0])
                
                image_u_aug, pseudo_per_img = cut_mix_label_adaptive_det(
                    image_u_aug,
                    pseudo_per_img,
                    image_l,
                    labeled_targets,
                    lst_confidences,
                )

            # 3. Build pseudo batch from per-image pseudo boxes
            pseudo_batch, pseudo_ratio, pseudo_boxes_per_img = _build_pseudo_batch_from_list(
                pseudo_per_img, image_u_aug
            )

            # 4. Unsupervised loss: YOLO loss on pseudo labels (Paper: λ_u × L_u)
            if pseudo_batch['batch_idx'].numel() > 0:
                yolo_model.train()
                for m in yolo_model.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                        m.eval()
                student_preds_u = yolo_model(image_u_aug)
                pseudo_loss_items, _ = _safe_yolo_loss(yolo_model, pseudo_batch, student_preds_u)
                unsup_loss = pseudo_loss_items.sum() * float(loss_weight)
                if not torch.isfinite(unsup_loss):
                    unsup_loss = torch.tensor(0.0, device=image_l.device)
            
        pseudo_ratios.update(pseudo_ratio)
        pseudo_boxes_per_img_meter.update(pseudo_boxes_per_img)

        # ============ Total Loss (Paper Eq.2: L = L_x + λ_u × L_u) ============
        loss = sup_loss + unsup_loss
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()

        # Update teacher model with EMA (Paper Eq.1)
        with torch.no_grad():
            if epoch >= sup_only_epoch:
                ema_decay = min(
                    1 - 1 / (i_iter - len(loader_l) * sup_only_epoch + 1),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                # Only EMA-update float buffers (BN stats); copy non-float directly
                if buffer_eval.dtype in (torch.float32, torch.float64, torch.float16):
                    buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                else:
                    buffer_eval.data.copy_(buffer_train.data)

        # Gather losses from all GPUs
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        # Logging
        batch_times.update(time.time() - batch_start)
        
        if step in print_freq_lst and rank == 0:
            logger.info(
                f"Epoch/Iter [{cfg['trainer']['epochs']}:{epoch:3}/{step:3}].  "
                f"Sup:{sup_losses.val:.3f}({sup_losses.avg:.3f})  "
                f"Uns:{uns_losses.val:.3f}({uns_losses.avg:.3f})  "
                f"Pseudo:{pseudo_ratios.avg:.2f}  "
                f"PBox/Img:{pseudo_boxes_per_img_meter.avg:.2f}  "
                f"Time:{batch_times.avg:.2f}  "
                f"LR:{learning_rates.val:.5f}"
            )
            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("Pseudo Ratio", pseudo_ratios.avg, i_iter)
                tb_logger.add_scalar("Pseudo Boxes per Img", pseudo_boxes_per_img_meter.avg, i_iter)
    
    return sup_losses.avg, uns_losses.avg


def validate_yolo(model, data_loader, epoch, logger, cfg, prefix=""):
    """
    Validation function for YOLO detection computing P, R, mAP50, mAP50-95.

    Returns:
        dict with keys: 'P', 'R', 'mAP50', 'mAP50-95'
    """
    try:
        from ultralytics.utils.metrics import box_iou
    except ImportError:
        def box_iou(box1, box2):
            area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
            inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
            inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
            inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
            inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
            inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            union_area = area1[:, None] + area2 - inter_area
            return inter_area / (union_area + 1e-10)

    yolo_model = model.module if hasattr(model, 'module') else model
    yolo_model.eval()

    data_loader.sampler.set_epoch(epoch)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # IoU thresholds for mAP computation
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.50, 0.55, ..., 0.95]
    conf_thres = 0.001   # Low threshold for proper PR curve
    nms_iou_thres = 0.65
    zero_metrics = {'Precision': 0.0, 'Recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}

    stats = []  # list of (correct[N,10], conf[N], pred_cls[N], target_cls[M])

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        batch_size, _, img_h, img_w = images.shape

        with torch.no_grad():
            preds = yolo_model(images)
            nms_preds = apply_nms(preds, conf_thres=conf_thres, iou_thres=nms_iou_thres, max_det=300)

        for bi, pred in enumerate(nms_preds):
            # Get ground truth for this image
            if len(labels) > 0:
                target_boxes = labels[labels[:, 0] == bi]
                if len(target_boxes) > 0:
                    target_cls = target_boxes[:, 1].cpu()
                    tboxes_raw = target_boxes[:, 2:6].clone()
                    cx, cy, w, h = tboxes_raw[:, 0], tboxes_raw[:, 1], tboxes_raw[:, 2], tboxes_raw[:, 3]
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    tboxes = torch.stack([x1, y1, x2, y2], dim=1).cpu()
                else:
                    target_cls = torch.zeros(0)
                    tboxes = torch.zeros((0, 4))
            else:
                target_cls = torch.zeros(0)
                tboxes = torch.zeros((0, 4))

            nl = len(target_cls)

            if len(pred) == 0:
                if nl:
                    stats.append((
                        np.zeros((0, len(iou_thresholds)), dtype=bool),
                        np.zeros(0),
                        np.zeros(0),
                        target_cls.numpy()
                    ))
                continue

            pred_boxes = pred[:, :4].cpu()
            pred_conf = pred[:, 4].cpu()
            pred_cls_t = pred[:, 5].cpu()

            # Match predictions to GT at multiple IoU thresholds
            correct = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)

            if nl:
                iou = box_iou(pred_boxes, tboxes)  # [N_pred, N_gt]
                correct_class = pred_cls_t[:, None] == target_cls[None, :]  # [N_pred, N_gt]

                for ti, thr in enumerate(iou_thresholds):
                    matches = torch.nonzero(
                        (iou >= thr) & correct_class, as_tuple=False
                    )  # [K, 2] -> (pred_idx, gt_idx)

                    if matches.shape[0]:
                        # Sort by IoU descending to prioritize best matches
                        match_ious = iou[matches[:, 0], matches[:, 1]]
                        sorted_idx = match_ious.argsort(descending=True)
                        matches = matches[sorted_idx]

                        # Each prediction matched at most once (keep highest IoU)
                        _, unique_pred_idx = np.unique(
                            matches[:, 0].cpu().numpy(), return_index=True
                        )
                        matches = matches[unique_pred_idx]

                        # Each ground truth matched at most once
                        _, unique_gt_idx = np.unique(
                            matches[:, 1].cpu().numpy(), return_index=True
                        )
                        matches = matches[unique_gt_idx]

                        correct[matches[:, 0].cpu().numpy(), ti] = True

            stats.append((
                correct,
                pred_conf.numpy(),
                pred_cls_t.numpy(),
                target_cls.numpy()
            ))

    # Gather validation stats across all ranks so AP/mAP are computed globally.
    # Nếu chạy DDP, cần gộp stats từ mọi GPU rồi mới tính metric để tránh lệch kết quả.
    if world_size > 1:
        gathered_stats = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_stats, stats)
        merged_stats = []
        for rank_stats in gathered_stats:
            if rank_stats:
                merged_stats.extend(rank_stats)
        stats = merged_stats

    # Return zeros if no stats
    if len(stats) == 0:
        if rank == 0 and logger:
            logger.info(f" [{prefix}Val] No predictions/targets for evaluation")
        return zero_metrics

    # Concatenate all stats
    tp_all = np.concatenate([s[0] for s in stats], axis=0)      # [total_pred, 10]
    conf_all = np.concatenate([s[1] for s in stats], axis=0)    # [total_pred]
    pred_cls_all = np.concatenate([s[2] for s in stats], axis=0)  # [total_pred]
    target_cls_all = np.concatenate([s[3] for s in stats], axis=0)  # [total_gt]

    if len(tp_all) == 0 or len(target_cls_all) == 0:
        if rank == 0 and logger:
            logger.info(f" [{prefix}Val] Empty stats")
        return zero_metrics

    # Compute per-class AP
    p, r, ap, unique_classes = ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all)

    # Mean across classes
    mp = p.mean() if len(p) else 0.0        # Mean Precision (at max-F1, IoU=0.5)
    mr = r.mean() if len(r) else 0.0        # Mean Recall    (at max-F1, IoU=0.5)
    map50 = ap[:, 0].mean() if len(ap) else 0.0      # mAP@0.5
    map50_95 = ap.mean() if len(ap) else 0.0          # mAP@0.5:0.95

    if rank == 0 and logger:
        logger.info(
            f" [{prefix}Val] Precision:{mp:.4f}  Recall:{mr:.4f}  mAP50:{map50:.4f}  mAP50-95:{map50_95:.4f}  "
            f"(classes:{len(unique_classes)}, preds:{len(tp_all)}, targets:{len(target_cls_all)})"
        )

    return {'Precision': float(mp), 'Recall': float(mr), 'mAP50': float(map50), 'mAP50-95': float(map50_95)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised YOLO Detection Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
