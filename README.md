<div align="center">    
 
# Semi-Supervised Leaf Disease Detection using YOLOv11

</div>

## Overview
This repository implements multiple semi-supervised learning approaches for banana leaf disease detection using YOLOv11 with various attention mechanisms. We propose multiple YOLOv11 variants (Base, SA-Origin, SA-Custom) with Self-Attention mechanisms to enhance feature extraction and detection accuracy.

## Schemes

### Scheme 1: Iterative Pseudo-Labeling
This scheme leverages unlabeled data through iterative pseudo-labeling to improve detection performance in scenarios with limited labeled data. The framework generates pseudo-labels on unlabeled images and iteratively refines the model by incorporating high-confidence predictions.

![image](scheme1.png)

**Training script**: `Sample_Semi_Train_Iterative_Simple.sh`

### Scheme 2: [Coming Soon]
The second semi-supervised learning scheme

![image](scheme2.png)

**Training script**: 

### Scheme 3: [Coming Soon]
The third semi-supervised learning scheme

![image](scheme3.png)

**Training script**: 

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
Organize your dataset in the following structure:
```
your_data_path/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── Unlabeled_Images_2025/
    └── images/
```

Update dataset configuration files (`Sample_Banana_Disease_Dataset.yaml`, `Sample_Banana_Disease_Dataset_Semi.yaml`) with your dataset paths:
```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test
```

## Training

### Supervised Baseline Training
Train the initial models using only labeled data:
```bash
bash Sample_Supervised_Train_Baseline_Simple.sh
```
This will train YOLOv11-Base, YOLOv11-SA-Origin, and YOLOv11-SA-Custom for 400 epochs.

### Semi-Supervised Iterative Training
Run the semi-supervised learning pipeline with iterative pseudo-labeling:
```bash
bash Sample_Semi_Train_Iterative_Simple.sh
```

**Note**: Update the paths in shell scripts to match your dataset location before running.

### Evaluation
Evaluate trained models:
```bash
yolo val model=runs/detect/YOLOv11-All-Scheme-Flinta/model_name/weights/best.pt \
         data=Sample_Banana_Disease_Dataset_Test.yaml \
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
