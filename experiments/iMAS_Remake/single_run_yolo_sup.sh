#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:-2}
CONFIG=${CONFIG:-./exps/yolo_det/config_sup.yaml}
python train_yolo_sup.py --config "${CONFIG}" --seed "${SEED}"
