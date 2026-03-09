#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

yolo train data=configs/Banana_Disease_Dataset.yaml model=yolo11-SA-Custom.yaml    batch=8 imgsz=1024 patience=0  name=YOLOv11-SA-Custom-400 project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True

yolo train data=configs/Banana_Disease_Dataset.yaml model=yolo11-SA.yaml    batch=8 imgsz=1024 patience=0  name=YOLOv11-SA-Origin-400 project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True

yolo train data=configs/Banana_Disease_Dataset.yaml model=yolo11.yaml    batch=8 imgsz=1024 patience=100  name=YOLOv11-Base-400 project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



