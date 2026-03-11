#!/bin/bash
# Semi-Supervised YOLO (AugSeg-style) with EMA Teacher
#
# ITER i:
#   Teacher (EMA)
#      weak predict unlabeled → pseudo labels (lọc conf)
#        
#
#   Student training:
#         với augmentations mạnh hơn cái trên
#
#   Student best → EMA → Teacher(i+1)

set -e
set -x

# CONFIG (workspace-relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${SCHEME_DIR}/../../.." && pwd)"
cd "$SCHEME_DIR"

WORKDIR="$ROOT_DIR"

# MODEL_BASE="$ROOT_DIR/experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt"
MODEL_BASE="${MODEL_BASE:-$ROOT_DIR/experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt}"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/banana_dataset/Banana_Dataset_2024_2025}"

# LABELED_FROM_VAL: Ground-truth labeled seed data for semi-supervised training
#   - Small set of manually labeled images
#   - Used to initialize the labeled pool before pseudo-labeling
LABELED_IMG="${LABELED_IMG:-$ROOT_DIR/banana_dataset/Labeled_From_Val/images}"

# UNLABELED_IMAGES: Large pool of unlabeled images for pseudo-labeling
UNLABELED_IMG="$DATA_ROOT/Unlabeled_Images_2025/images"

PROJECT="${PROJECT:-$SCHEME_DIR/YOLOv11-AugSeg-Scheme-Fixed/YOLOv11-Base}"

# VAL_CLEAN: Validation set for model evaluation (NOT used in training)
#   - Contains labeled validation images for objective metrics (mAP, precision, recall)
#   - Independent from Labeled_From_Val and Unlabeled_Images
VAL_PATH="${VAL_PATH:-$ROOT_DIR/banana_dataset/Val_Clean}"

MAX_ITER=35
EPOCHS=50
BATCH=8
IMGSZ=1024

CONF_THRES=0.01
IOU_THRES=0.6
EMA_DECAY=0.995

#PATHS 
mkdir -p "$PROJECT"
mkdir -p "$SCHEME_DIR/runs/predict"

LABELED_LIST="$DATA_ROOT/labeled.txt"
PSEUDO_LIST="$DATA_ROOT/pseudo.txt"
TRAIN_LIST="$DATA_ROOT/train.txt"

DATA_YAML="$SCHEME_DIR/configs/data_train.yaml"

TEACHER="$PROJECT/teacher_ema.pt"

# PREP LABELED 
if [ ! -f "$LABELED_LIST" ]; then
  find "$LABELED_IMG" -type f | sort > "$LABELED_LIST"
fi

cat > "$DATA_YAML" <<EOF
train: $TRAIN_LIST
val: $VAL_PATH
nc: 2
names: ['early_sigatoka','late_sigatoka']
EOF

# INIT TEACHER 
if [ ! -f "$TEACHER" ]; then
  python3 "$SCHEME_DIR/ema_update.py" \
    --student "$MODEL_BASE" \
    --teacher None \
    --out "$TEACHER" \
    --decay $EMA_DECAY
fi

#MAIN LOOP
for ((ITER=1; ITER<=MAX_ITER; ITER++)); do
  echo "=========== ITER $ITER ==========="

  #EMA decay 
  if [ $ITER -le 2 ]; then
    EMA_DECAY=0.999
  else
    EMA_DECAY=0.995
  fi
  
  # CONF WARMUP (QUAN TRỌNG) 

  if [ $ITER -le 2 ]; then
    CONF_THRES=0.3
  elif [ $ITER -le 5 ]; then
    CONF_THRES=0.15
  else
    CONF_THRES=0.05
  fi

  if [ $ITER -le 2 ]; then
    PSEUDO_CONF=0.001
  elif [ $ITER -le 5 ]; then
    PSEUDO_CONF=0.005
  else
    PSEUDO_CONF=0.01
  fi
  echo "[INFO] ITER $ITER - CONF_THRES=$CONF_THRES"
  
  #1. PSEUDO LABEL (WEAK)
  mkdir -p "$DATA_ROOT/Unlabeled_Images_2025/labels"
  : > "$PSEUDO_LIST"

  yolo predict \
    model="$TEACHER" \
    source="$UNLABELED_IMG" \
    conf=$CONF_THRES \
    iou=$IOU_THRES \
    imgsz=$IMGSZ \
    augment=True \
    agnostic_nms=True \
    max_det=300 \
    save_txt=True \
    save_conf=True \
    project="$SCHEME_DIR/runs/predict" \
    name="iter${ITER}" \
    exist_ok=True

  shopt -s nullglob
  for f in "$SCHEME_DIR"/runs/predict/iter${ITER}/labels/*.txt; do
    awk -v conf="$PSEUDO_CONF" '$6>=conf && ($4*$5)>=0.005 {print $1,$2,$3,$4,$5}' "$f" \
      > "$DATA_ROOT/Unlabeled_Images_2025/labels/$(basename "$f")"

    base=$(basename "$f" .txt)
    for ext in jpg JPG png PNG jpeg JPEG; do
      img="$UNLABELED_IMG/$base.$ext"
      [ -f "$img" ] && echo "$img" >> "$PSEUDO_LIST" && break
    done
  done
  shopt -u nullglob

  # 2. MERGE LABELED + PSEUDO 
  # cat "$LABELED_LIST" "$PSEUDO_LIST" | sort -u > "$TRAIN_LIST"
  LABELED_COUNT=$(wc -l < "$LABELED_LIST")

  if [ $ITER -le 10 ]; then
    MAX_PSEUDO=$((LABELED_COUNT * 2))
  elif [ $ITER -le 20 ]; then
    MAX_PSEUDO=$((LABELED_COUNT * 3))
  else
    MAX_PSEUDO=$((LABELED_COUNT * 4))
  fi

  shuf "$PSEUDO_LIST" | head -n $MAX_PSEUDO > "$PSEUDO_LIST.tmp"
  mv "$PSEUDO_LIST.tmp" "$PSEUDO_LIST"

    
  cat "$LABELED_LIST" "$PSEUDO_LIST" | sort -u > "$TRAIN_LIST"

  echo "[INFO] Train images count:"
  wc -l "$TRAIN_LIST"

  if [ $ITER -le 2 ]; then
    FREEZE=5
  else
    FREEZE=0
  fi
  # 3. TRAIN STUDENT (LABELED + PSEUDO) 
  yolo train \
    model="$TEACHER" \
    freeze=$FREEZE \
    data="$DATA_YAML" \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMGSZ \
    hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 \
    degrees=5 scale=0.1 translate=0.05 \
    mosaic=0 mixup=0 \
    project="$PROJECT" \
    name="iter${ITER}_student" \
    exist_ok=True

  STUDENT="$PROJECT/iter${ITER}_student/weights/best.pt"

  # 4. EMA UPDATE
  python3 "$SCHEME_DIR/ema_update.py" \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --out "$PROJECT/teacher_iter${ITER}.pt" \
    --decay $EMA_DECAY

  TEACHER="$PROJECT/teacher_iter${ITER}.pt"
done

echo "DONE. Final teacher: $TEACHER"
