#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-2}
CONFIG=${CONFIG:-./exps/yolo_det/config_semi.yaml}
python train_yolo_semi.py --config "${CONFIG}" --seed "${SEED}"
