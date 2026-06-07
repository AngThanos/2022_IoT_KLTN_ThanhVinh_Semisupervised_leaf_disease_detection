#!/bin/bash
# Backup the 6 BCE cells that ran successfully WITHOUT the cls=0.2 + pos_weight fix,
# then rerun them with the fix for fair comparison across all 9 BCE cells.
#
# Backup goes to: backups/bce_seed42_no_fix/<cell_name>/
#
# Usage:
#   setsid nohup bash run_bce_rerun_remaining6.sh \
#     > logs/bce_rerun_remaining6_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & disown

set -u
cd "$(dirname "$0")"

source /opt/tljh/user/etc/profile.d/conda.sh
conda activate Cinh_env

mkdir -p logs backups/bce_seed42_no_fix

GPU=0
SEED=43

CONFIGS=(
    "exps/conf_001/bce/yolov11/config_semi.yaml"
    "exps/conf_001/bce/yolov11-sa/config_semi.yaml"
    "exps/conf_001/bce/yolov11-sa-custom/config_semi.yaml"
    "exps/conf_025/bce/yolov11-sa/config_semi.yaml"
    "exps/dynamic/bce/yolov11/config_semi.yaml"
    "exps/dynamic/bce/yolov11-sa-custom/config_semi.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "BCE rerun remaining 6 cells (with fix, seed=$SEED)"
echo "Backup target: backups/bce_seed42_no_fix/"
echo "GPU          : $GPU"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    tag=$(echo "$cfg" | sed -E 's#exps/##; s#/config_semi.yaml##; s#/#_#g')
    log_file="logs/${tag}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

    out_root=$(python -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['project']['output_root'])")
    backup_dir="backups/bce_seed42_no_fix/${tag}"

    echo ""
    echo "[$idx/$TOTAL] >>> $cfg"
    echo "             out    -> $out_root"
    echo "             backup -> $backup_dir"
    echo "             log    -> $log_file"

    if [[ ! -f "$cfg" ]]; then
        echo "             SKIP (config not found)"
        continue
    fi

    # ---- BACKUP existing checkpoints/summary (keep config) ----
    if [[ -d "$out_root" ]]; then
        mkdir -p "$backup_dir"
        for f in latest.pth best.pth best.pt last.pt summary.json _val_data.yaml; do
            if [[ -f "$out_root/$f" ]]; then
                cp -p "$out_root/$f" "$backup_dir/$f"
            fi
        done
        echo "             backed up files in $backup_dir/"

        # Wipe checkpoints (keep config_semi.yaml)
        rm -f "$out_root"/latest.pth "$out_root"/best.pth \
              "$out_root"/best.pt "$out_root"/last.pt \
              "$out_root"/summary.json "$out_root"/_val_data.yaml \
              "$out_root"/_tmp_val.pt
        echo "             wiped previous checkpoints"
    fi

    # ---- RERUN with fix + seed=43 ----
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
echo "Rerun remaining 6 cells (seed=$SEED) finished."
echo "Backups stored at backups/bce_seed42_no_fix/"
echo "=========================================="
