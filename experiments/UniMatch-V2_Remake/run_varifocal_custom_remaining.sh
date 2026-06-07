#!/usr/bin/env bash
# Run 8 remaining varifocal_custom configs (skipping conf_001/yolov11 already done with SGD).
set -u
cd "$(dirname "$0")"

source /opt/tljh/user/etc/profile.d/conda.sh
conda activate Cinh_env

GPU=${GPU:-0}
mkdir -p logs

CONFIGS=(
    "exps/conf_001/varifocal_custom/yolov11-sa/config_semi.yaml"
    "exps/conf_001/varifocal_custom/yolov11-sa-custom/config_semi.yaml"
    "exps/conf_025/varifocal_custom/yolov11/config_semi.yaml"
    "exps/conf_025/varifocal_custom/yolov11-sa/config_semi.yaml"
    "exps/conf_025/varifocal_custom/yolov11-sa-custom/config_semi.yaml"
    "exps/dynamic/varifocal_custom/yolov11/config_semi.yaml"
    "exps/dynamic/varifocal_custom/yolov11-sa/config_semi.yaml"
    "exps/dynamic/varifocal_custom/yolov11-sa-custom/config_semi.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    tag=$(echo "$cfg" | sed -E 's#exps/##; s#/config_semi.yaml##; s#/#_#g')
    log_file="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "=========================================="
    echo "[$(date '+%F %T')] START $cfg"
    echo "  log: $log_file"
    echo "=========================================="

    if [[ ! -f "$cfg" ]]; then
        echo "  SKIP (config not found)"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python unimatch_v2_yolo.py \
        --config "$cfg" \
        > "$log_file" 2>&1

    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED $cfg (exit $rc)"
    else
        echo "[$(date '+%F %T')] OK $cfg"
    fi
done

echo "=========================================="
echo "[$(date '+%F %T')] ALL DONE"
echo "=========================================="
