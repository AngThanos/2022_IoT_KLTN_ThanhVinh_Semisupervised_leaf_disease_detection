#!/bin/bash
# Run 9 UniMatch-V2_Remake sweep cells for loss = bce
#   BCEWithLogitsLoss (Ultralytics default).
#
# 9 = 3 confs (conf_001 / conf_025 / dynamic) x 3 models (yolov11 / -sa / -sa-custom)
#
# IMPORTANT: BEFORE running this script you MUST patch the Ultralytics source
# (`ultralytics/utils/loss.py`) so that v8DetectionLoss uses:
#   BCEWithLogitsLoss (default v8DetectionLoss, no patch needed).
#
# Usage:
#   nohup bash run_bce.sh > logs/bce_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown

set -u
cd "$(dirname "$0")"

# Activate conda
source /opt/tljh/user/etc/profile.d/conda.sh
conda activate Cinh_env

mkdir -p logs

GPU=0

CONFIGS=(
    "exps/conf_001/bce/yolov11/config_semi.yaml"
    "exps/conf_001/bce/yolov11-sa/config_semi.yaml"
    "exps/conf_001/bce/yolov11-sa-custom/config_semi.yaml"
    "exps/conf_025/bce/yolov11/config_semi.yaml"
    "exps/conf_025/bce/yolov11-sa/config_semi.yaml"
    "exps/conf_025/bce/yolov11-sa-custom/config_semi.yaml"
    "exps/dynamic/bce/yolov11/config_semi.yaml"
    "exps/dynamic/bce/yolov11-sa/config_semi.yaml"
    "exps/dynamic/bce/yolov11-sa-custom/config_semi.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "UniMatch-V2_Remake sweep -- loss = bce"
echo "Total cells: $TOTAL"
echo "GPU        : $GPU"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    tag=$(echo "$cfg" | sed -E 's#exps/##; s#/config_semi.yaml##; s#/#_#g')
    log_file="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "[$idx/$TOTAL] >>> $cfg"
    echo "             log -> $log_file"

    if [[ ! -f "$cfg" ]]; then
        echo "             SKIP (config not found)"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python unimatch_v2_yolo.py \
        --config "$cfg" \
        2>&1 | tee "$log_file"

    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "             FAILED (exit $rc) -- continuing to next cell"
    else
        echo "             OK"
    fi
done

echo ""
echo "=========================================="
echo "Sweep (bce) finished."
echo "=========================================="
