#!/bin/bash
# Semi-Supervised YOLO Training Script

cd "$(dirname "$0")"

# Activate virtual environment
# source ../../AugSeg/AugSeg/venv/bin/activate

CONFIG=${1:-"exps/yolov11-base-400/config_semi.yaml"}
GPUS=${2:-1}
PORT=${3:-29500}
SEED=${4:-42}

echo "=========================================="
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Port: $PORT"
echo "Seed: $SEED"
echo "=========================================="

if [ $GPUS -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0 torchrun \
        --nproc_per_node=1 \
        --master_port=$PORT \
        train_semi.py \
        --config $CONFIG \
        --seed $SEED
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        train_semi.py \
        --config $CONFIG \
        --seed $SEED
fi
