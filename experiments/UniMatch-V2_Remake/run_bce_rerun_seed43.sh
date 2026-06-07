#!/bin/bash
# Rerun 3 collapsed BCE cells with seed=43 (was 42).
# Deletes existing checkpoints first so training starts fresh.
#
# Usage:
#   setsid nohup bash run_bce_rerun_seed43.sh \
#     > logs/bce_rerun_seed43_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & disown

set -u
cd "$(dirname "$0")"

source /opt/tljh/user/etc/profile.d/conda.sh
conda activate Cinh_env

mkdir -p logs

GPU=0
SEED=43

CONFIGS=(
    "exps/conf_025/bce/yolov11/config_semi.yaml"
    "exps/conf_025/bce/yolov11-sa-custom/config_semi.yaml"
    "exps/dynamic/bce/yolov11-sa/config_semi.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "BCE rerun with seed=$SEED  (collapsed cells)"
echo "Total cells: $TOTAL"
echo "GPU        : $GPU"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    tag=$(echo "$cfg" | sed -E 's#exps/##; s#/config_semi.yaml##; s#/#_#g')
    log_file="logs/${tag}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

    # Resolve output_root from config to wipe before rerun
    out_root=$(python -c "import yaml,sys; c=yaml.safe_load(open('$cfg')); print(c['project']['output_root'])")

    echo ""
    echo "[$idx/$TOTAL] >>> $cfg"
    echo "             out -> $out_root"
    echo "             log -> $log_file"

    if [[ -d "$out_root" ]]; then
        echo "             wiping previous checkpoints (keeping config)..."
        rm -f "$out_root"/latest.pth "$out_root"/best.pth \
              "$out_root"/best.pt "$out_root"/last.pt \
              "$out_root"/summary.json "$out_root"/_val_data.yaml \
              "$out_root"/_tmp_val.pt
    fi

    if [[ ! -f "$cfg" ]]; then
        echo "             SKIP (config not found)"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU python unimatch_v2_yolo.py \
        --config "$cfg" \
        --seed "$SEED" \
        2>&1 | tee "$log_file"

    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "             FAILED (exit $rc) -- continuing"
    else
        echo "             OK"
    fi
done

echo ""
echo "=========================================="
echo "Rerun (seed=$SEED) finished."
echo "=========================================="
