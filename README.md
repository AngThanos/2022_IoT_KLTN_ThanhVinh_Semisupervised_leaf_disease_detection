<div align="center">    
 
# Semi-Supervised Leaf Disease Detection using YOLOv11

</div>

## Overview
This repository implements multiple semi-supervised learning approaches for banana leaf disease detection using YOLOv11 with various attention mechanisms. We propose multiple YOLOv11 variants (Base, SA-Origin, SA-Custom) with Self-Attention mechanisms to enhance feature extraction and detection accuracy.

## Schemes

### Scheme 1: Iterative Pseudo-Labeling (Baseline)
Original semi-supervised approach with iterative pseudo-labeling to improve detection with limited labeled data. Generates pseudo-labels on unlabeled images and refines the model by incorporating high-confidence predictions.

**Location**: `experiments/quoccuong_original/`  
**Training script**: `bash scripts/original_semi_supervised.sh` (run from quoccuong_original directory)

### Scheme 2: EMA-based Semi-Supervised with Augmentation (general)
Advanced semi-supervised learning with Exponential Moving Average (EMA) teacher-student framework and strong augmentations. Gradually increases pseudo-labeled examples ratio (2x-> 3x-> 4x labeled set size) with confidence warmup strategy.

**Location**: `experiments/augmentation_matters_semi/scheme1_augmat_general/`  
**Training script**: `bash scripts/augmat_semi_general.sh` (run from scheme1_augmat_general directory)

### Scheme 3: [Coming Soon]
The third semi-supervised learning scheme 

## Prerequisites
Step by Step installation,
```bash
conda create -n leaf_disease python=3.12
conda activate leaf_disease

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 12.4:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# For other CUDA versions, visit: https://pytorch.org

# Install Ultralytics YOLOv11
cd ultralytics-8.3.225
pip install -e .
```

**Tested Environment:**
- Python 3.12.6
- PyTorch 2.6.0 + CUDA 12.4
- Ultralytics 8.3.225

## Dataset Preparation

Dataset is organized in `banana_dataset/` folder:

| Directory | Purpose |
|-----------|---------|
| `Banana_Dataset_2024_TrainValTest/` | Supervised baseline training (labeled) |
| `Banana_Dataset_2024_2025/Unlabeled_Images_2025/` | Large unlabeled pool for pseudo-labeling |
| `Val_Clean/` | Validation set for model evaluation |
| `Labeled_From_Val/` | Labeled seed data for semi-supervised schemes |

See [banana_dataset/README.md](banana_dataset/README.md) for detailed structure and setup instructions.

**Model output locations:**
- Supervised baseline: `experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/`
- Semi-supervised: `experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/` (iterations)
- Augmentation scheme: `experiments/augmentation_matters_semi/scheme1_augmat_general/runs/YOLOv11-AugSeg-Scheme-Fixed/`

## Training

### Supervised Baseline Training
Train YOLOv11 variants using labeled data only:
```bash
cd experiments/quoccuong_original
bash scripts/original_train.sh
```
Trains YOLOv11-Base, YOLOv11-SA-Origin, YOLOv11-SA-Custom for 400 epochs.

### Semi-Supervised Iterative Training (Scheme 1)
Run iterative pseudo-labeling with the original approach:
```bash
cd experiments/quoccuong_original
bash scripts/original_semi_supervised.sh
```

### Semi-Supervised with EMA Teacher (Scheme 2)
Advanced semi-supervised training with strong augmentation:
```bash
cd experiments/augmentation_matters_semi/scheme1_augmat_general
bash scripts/augmat_semi_general.sh
```
Runs 35 iterations with confidence warmup and dynamic pseudo-label sampling.

### Evaluation
Evaluate trained models:
```bash
yolo val model=experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt \
         data=experiments/quoccuong_original/configs/Banana_Disease_Dataset_Test.yaml \
         imgsz=1024
```

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| YOLOv11-Base | - | - | - | - |
| YOLOv11-SA-Origin | - | - | - | - |
| YOLOv11-SA-Custom | - | - | - | - |
| YOLOv11-SA-Custom-Iter5 | - | - | - | - |

## Acknowledgements
This project is based on the following open-source projects:
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Semi-supervised learning techniques for object detection

We thank their authors for making the source code publicly available.

## Contact
If you have any problem with our code, feel free to contact
- phamthanh050204@gmail.com

or describe your problem in Issues.
