#!/usr/bin/env bash
# Evaluate all 27 UniMatch-V2_Remake checkpoints on Banana test set.
# Run from /home/jupyter-iec2021iot13/Vinh/ (or anywhere — uses absolute paths).
#
# Usage:
#   bash /home/jupyter-iec2021iot13/Vinh/UniMatch-V2_Remake/eval_all_27.sh
#   # or specific bucket:
#   bash eval_all_27.sh bce            # only bce
#   bash eval_all_27.sh varifocal      # only varifocal
#   bash eval_all_27.sh varifocal_custom

set -u

REPO=/home/jupyter-iec2021iot13/Vinh/UniMatch-V2_Remake
DATA_YAML=/home/jupyter-iec2021iot13/Vinh/Banana_Disease_Dataset_Test.yaml
OUT_ROOT=$REPO/eval_results
SUMMARY=$OUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).csv

mkdir -p "$OUT_ROOT"

LOSSES=(bce varifocal varifocal_custom)
CONFS=(conf_001 conf_025 dynamic)
MODELS=(yolov11 yolov11-sa yolov11-sa-custom)

# Optional filter (first positional arg = loss bucket)
if [[ $# -ge 1 ]]; then
    LOSSES=("$1")
fi

echo "config,best_pt,P,R,mAP50,mAP50-95" > "$SUMMARY"
echo "Summary CSV -> $SUMMARY"
echo "=========================================="

TOTAL=0; OK=0; SKIP=0; FAIL=0

for loss in "${LOSSES[@]}"; do
    for conf in "${CONFS[@]}"; do
        for model in "${MODELS[@]}"; do
            TOTAL=$((TOTAL+1))
            tag="${conf}_${loss}_${model}"
            best_pt="$REPO/exps/$conf/$loss/$model/best.pt"
            run_dir="$OUT_ROOT/$tag"

            echo ""
            echo "[$TOTAL] $tag"

            if [[ ! -f "$best_pt" ]]; then
                echo "    SKIP (no best.pt: $best_pt)"
                echo "$tag,MISSING,,,," >> "$SUMMARY"
                SKIP=$((SKIP+1))
                continue
            fi

            mkdir -p "$run_dir"
            log_file="$run_dir/val.log"

            yolo val \
                data="$DATA_YAML" \
                model="$best_pt" \
                imgsz=1024 \
                conf=0.1 \
                iou=0.1 \
                agnostic_nms=True \
                project="$run_dir" \
                name=val \
                exist_ok=True \
                > "$log_file" 2>&1

            rc=$?
            if [[ $rc -ne 0 ]]; then
                echo "    FAILED (exit $rc) -- see $log_file"
                echo "$tag,FAILED,,,," >> "$SUMMARY"
                FAIL=$((FAIL+1))
                continue
            fi

            # Parse final "all" line: "all  N  M  P  R  mAP50  mAP50-95"
            metrics=$(grep -E "^\s+all\s" "$log_file" | tail -1 \
                | awk '{print $4","$5","$6","$7}')
            if [[ -z "$metrics" ]]; then
                metrics=",,,"
                echo "    OK but parse failed"
            else
                echo "    P=$(echo $metrics | cut -d, -f1)  R=$(echo $metrics | cut -d, -f2)  mAP50=$(echo $metrics | cut -d, -f3)  mAP50-95=$(echo $metrics | cut -d, -f4)"
            fi
            echo "$tag,$best_pt,$metrics" >> "$SUMMARY"
            OK=$((OK+1))
        done
    done
done

echo ""
echo "=========================================="
echo "DONE  total=$TOTAL  ok=$OK  skip=$SKIP  fail=$FAIL"
echo "Summary: $SUMMARY"
echo "=========================================="
# Pretty table sorted by mAP50 desc
echo ""
echo "TOP results (sorted by mAP50 desc):"
{
  head -1 "$SUMMARY"
  tail -n +2 "$SUMMARY" | sort -t, -k5 -gr
} | column -s, -t | head -30
