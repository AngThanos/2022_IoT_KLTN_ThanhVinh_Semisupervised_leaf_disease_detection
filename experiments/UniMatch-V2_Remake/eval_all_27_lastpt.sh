#!/usr/bin/env bash
# Evaluate all 27 checkpoints using last.pt on Banana_Disease_Dataset_Test.
# Command template follows user request, only model .pt path changes.

set -u

REPO=/home/jupyter-iec2021iot13/Vinh/UniMatch-V2_Remake
DATA_YAML=/home/jupyter-iec2021iot13/Vinh/Banana_Disease_Dataset_Test.yaml

LOSSES=(bce varifocal varifocal_custom)
CONFS=(conf_001 conf_025 dynamic)
MODELS=(yolov11 yolov11-sa yolov11-sa-custom)

TOTAL=0
OK=0
SKIP=0
FAIL=0

for conf in "${CONFS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        for arch in "${MODELS[@]}"; do
            TOTAL=$((TOTAL + 1))
            model_pt="$REPO/exps/$conf/$loss/$arch/last.pt"

            echo ""
            echo "[$TOTAL/27] $conf/$loss/$arch"

            if [[ ! -f "$model_pt" ]]; then
                echo "  SKIP: missing $model_pt"
                SKIP=$((SKIP + 1))
                continue
            fi

            yolo val \
                data="$DATA_YAML" \
                model="$model_pt" \
                imgsz=1024 \
                exist_ok=True \
                conf=0.1 \
                iou=0.1 \
                agnostic_nms=True

            rc=$?
            if [[ $rc -eq 0 ]]; then
                OK=$((OK + 1))
            else
                FAIL=$((FAIL + 1))
                echo "  FAILED (exit $rc): $model_pt"
            fi
        done
    done
done

echo ""
echo "DONE total=$TOTAL ok=$OK skip=$SKIP fail=$FAIL"
