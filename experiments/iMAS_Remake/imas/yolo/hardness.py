from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DetectionHardnessConfig:
    iou_match_thres: float = 0.5
    alpha_unmatched: float = 0.5
    weight_map: str = "linear"


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x, y, w, h = boxes.T
    half_w = w / 2.0
    half_h = h / 2.0
    return np.stack((x - half_w, y - half_h, x + half_w, y + half_h), axis=1)


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def map_hardness(values: np.ndarray, mode: str = "linear") -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if mode == "linear":
        return np.clip(values, 0.0, 1.0)
    if mode == "sqrt":
        return np.sqrt(np.clip(values, 0.0, 1.0))
    if mode == "quadratic":
        return np.square(np.clip(values, 0.0, 1.0))
    return np.clip(values, 0.0, 1.0)


def compute_detection_hardness(
    student_boxes: np.ndarray,
    teacher_boxes: np.ndarray,
    cfg: DetectionHardnessConfig | dict | None = None,
) -> tuple[float, dict]:
    if cfg is None:
        cfg = DetectionHardnessConfig()
    elif isinstance(cfg, dict):
        cfg = DetectionHardnessConfig(**cfg)

    student_boxes = np.asarray(student_boxes, dtype=np.float32)
    teacher_boxes = np.asarray(teacher_boxes, dtype=np.float32)

    if student_boxes.size == 0 and teacher_boxes.size == 0:
        return 0.5, {"matched": 0, "unmatched_student": 0, "unmatched_teacher": 0, "mean_iou": 0.0}

    student_xyxy = xywh_to_xyxy(student_boxes[:, 1:5]) if student_boxes.size else np.zeros((0, 4), dtype=np.float32)
    teacher_xyxy = xywh_to_xyxy(teacher_boxes[:, 1:5]) if teacher_boxes.size else np.zeros((0, 4), dtype=np.float32)

    student_used = set()
    teacher_used = set()
    matched_scores: list[float] = []

    for t_idx, t_box in enumerate(teacher_boxes):
        t_cls = int(t_box[0])
        best_iou = 0.0
        best_s_idx = -1
        for s_idx, s_box in enumerate(student_boxes):
            if s_idx in student_used:
                continue
            if int(s_box[0]) != t_cls:
                continue
            iou = box_iou_xyxy(teacher_xyxy[t_idx], student_xyxy[s_idx])
            if iou > best_iou:
                best_iou = iou
                best_s_idx = s_idx
        if best_s_idx >= 0 and best_iou >= cfg.iou_match_thres:
            student_used.add(best_s_idx)
            teacher_used.add(t_idx)
            conf_t = float(np.clip(t_box[5] if t_box.shape[0] > 5 else 1.0, 0.0, 1.0))
            conf_s = float(np.clip(student_boxes[best_s_idx][5] if student_boxes.shape[1] > 5 else 1.0, 0.0, 1.0))
            matched_scores.append(best_iou * np.sqrt(conf_t * conf_s))

    unmatched_teacher = max(0, len(teacher_boxes) - len(teacher_used))
    unmatched_student = max(0, len(student_boxes) - len(student_used))
    unmatched_ratio = (unmatched_teacher + unmatched_student) / max(1, len(teacher_boxes) + len(student_boxes))
    mean_score = float(np.mean(matched_scores)) if matched_scores else 0.0

    hardness_raw = 1.0 - mean_score + cfg.alpha_unmatched * unmatched_ratio
    hardness = float(np.clip(hardness_raw, 0.0, 1.0))
    hardness = float(map_hardness(np.array([hardness], dtype=np.float32), cfg.weight_map)[0])

    details = {
        "matched": len(matched_scores),
        "unmatched_student": unmatched_student,
        "unmatched_teacher": unmatched_teacher,
        "mean_iou": mean_score,
        "unmatched_ratio": float(unmatched_ratio),
    }
    return hardness, details


def batch_hardness(student_batch: Iterable[np.ndarray], teacher_batch: Iterable[np.ndarray], cfg=None) -> tuple[np.ndarray, list[dict]]:
    hardness_values = []
    details_list = []
    for student_boxes, teacher_boxes in zip(student_batch, teacher_batch):
        hardness, details = compute_detection_hardness(student_boxes, teacher_boxes, cfg=cfg)
        hardness_values.append(hardness)
        details_list.append(details)
    return np.asarray(hardness_values, dtype=np.float32), details_list
