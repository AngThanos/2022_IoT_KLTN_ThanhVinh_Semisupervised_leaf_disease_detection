#!/bin/bash
# UniMatch V2 for YOLOv11 Semi-Supervised Object Detection
# Usage: bash scripts/train_yolo.sh

set -e

CONFIG="configs/yolo_semi.yaml"
SAVE_PATH="./runs/unimatch_v2_yolo/$(date +%Y%m%d_%H%M%S)"

python unimatch_v2_yolo.py \
    --config ${CONFIG} \
    --save-path ${SAVE_PATH} \
    --seed 42

echo "Training complete. Results saved to: ${SAVE_PATH}"
