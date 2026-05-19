#!/bin/bash
# ─────────────────────────────────────────────────────────
#  Run AD-MT semi-supervised YOLO detection training
#  with iMAS-style data loading
#
#  Usage:
#    ./run_yolo_admt.sh [GPU_ID] [CONFIG]
#
#  Examples:
#    ./run_yolo_admt.sh 0                                  # default
#    ./run_yolo_admt.sh 0 config_yolo_admt.yml             # full AD-MT
#    ./run_yolo_admt.sh 0 ablation/ablation_no_ccm.yml     # ablation: no CCM
#    ./run_yolo_admt.sh 0 ablation/ablation_1tea.yml       # ablation: 1 teacher
# ─────────────────────────────────────────────────────────
set -e

GPU_ID=${1:-0}
CFG=${2:-config_yolo_admt.yml}

cd /home/jupyter-iec2021iot13/Vinh/AD-MT

echo "=== AD-MT YOLO Semi-supervised Training ==="
echo "GPU: ${GPU_ID}"
echo "Config: cfgs/${CFG}"
echo "============================================"

python ./code/train_yolo_admt.py \
    --cfg "${CFG}" \
    --gpu_id "${GPU_ID}"
