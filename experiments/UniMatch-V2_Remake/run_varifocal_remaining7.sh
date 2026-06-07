#!/bin/bash
# Run 7 varifocal cells (new loss patch applied to ultralytics source).
# Cells: conf_001/yolov11-sa-custom + conf_025/* + dynamic/*
#
# Hardening to avoid mid-run interruption:
#   - setsid + nohup + disown : detached from terminal/session, immune to SIGHUP
#   - stdin closed (< /dev/null) : no read-from-tty stalls
#   - set -u (no -e) : a single cell failure does NOT abort the sweep
#   - per-cell log via tee : output captured even if controlling terminal dies
#
# Usage:
#   setsid nohup bash run_varifocal_remaining7.sh \
#     > logs/varifocal_remaining7_master_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & disown

set -u
cd "$(dirname "$0")"

source /opt/tljh/user/etc/profile.d/conda.sh
conda activate Cinh_env

mkdir -p logs

GPU=0

CONFIGS=(
    "exps/conf_001/varifocal/yolov11-sa-custom/config_semi.yaml"
    "exps/conf_025/varifocal/yolov11/config_semi.yaml"
    "exps/conf_025/varifocal/yolov11-sa/config_semi.yaml"
    "exps/conf_025/varifocal/yolov11-sa-custom/config_semi.yaml"
    "exps/dynamic/varifocal/yolov11/config_semi.yaml"
    "exps/dynamic/varifocal/yolov11-sa/config_semi.yaml"
    "exps/dynamic/varifocal/yolov11-sa-custom/config_semi.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "=========================================="
echo "Varifocal rerun (new loss patch)  cells=$TOTAL  GPU=$GPU"
echo "PID=$$  date=$(date '+%F %T')"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    tag=$(echo "$cfg" | sed -E 's#exps/##; s#/config_semi.yaml##; s#/#_#g')
    log_file="logs/${tag}_$(date +%Y%m%d_%H%M%S).log"

    if [[ ! -f "$cfg" ]]; then
        echo "[$idx/$TOTAL] SKIP (config not found): $cfg"
        continue
    fi

    out_root=$(python -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['project']['output_root'])")

    echo ""
    echo "[$idx/$TOTAL] >>> $cfg"
    echo "             out -> $out_root"
    echo "             log -> $log_file"
    echo "             start: $(date '+%F %T')"

    # Wipe previous checkpoints (keep config_semi.yaml) so training restarts fresh
    if [[ -d "$out_root" ]]; then
        rm -f "$out_root"/latest.pth "$out_root"/best.pth \
              "$out_root"/best.pt    "$out_root"/last.pt \
              "$out_root"/summary.json "$out_root"/_val_data.yaml \
              "$out_root"/_tmp_val.pt
        echo "             wiped previous checkpoints (config kept)"
    fi

    CUDA_VISIBLE_DEVICES=$GPU python -u unimatch_v2_yolo.py \
        --config "$cfg" \
        2>&1 | tee "$log_file"

    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[$idx/$TOTAL] FAILED (exit $rc) -- continuing to next cell"
    else
        echo "[$idx/$TOTAL] OK   end: $(date '+%F %T')"
    fi
done

echo ""
echo "=========================================="
echo "Varifocal rerun finished. $(date '+%F %T')"
echo "=========================================="
