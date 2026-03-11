<div align="center">    
 
# Xây dựng hệ thống học bán giám sát kết hợp tăng cường dữ liệu dựa trên mô hình YOLOv11 để phát hiện bệnh trên lá chuối. 

</div>

## Tổng quan
Repository này triển khai nhiều hướng tiếp cận semi-supervised learning cho bài toán phát hiện bệnh lá chuối bằng YOLOv11 với các attention mechanism khác nhau. Sử dụng nhiều biến thể YOLOv11 (Base, SA-Origin, SA-Custom) có tích hợp Self-Attention để tăng khả năng trích xuất đặc trưng và cải thiện độ chính xác phát hiện.

## Các scheme

### Scheme 1: Iterative Pseudo-Labeling (Original)
Hướng semi-supervised gốc với iterative pseudo-labeling để cải thiện kết quả khi dữ liệu labeled còn hạn chế. Mô hình sinh pseudo-label trên unlabeled images và tiếp tục tinh chỉnh bằng các dự đoán có confidence cao.

**Vị trí**: `experiments/quoccuong_original/`  
**Chi tiết**: [xem tại đây](experiments/quoccuong_original/README.md)

### Scheme 2: EMA-based Semi-Supervised with Augmentation (general)
Hướng semi-supervised nâng cao dùng khung teacher-student với Exponential Moving Average (EMA) và strong augmentation. Tỉ lệ mẫu pseudo-labeled được tăng dần (2x -> 3x -> 4x theo kích thước labeled set) cùng chiến lược confidence warmup.

**Vị trí**: `experiments/augmentation_matters_semi/`  
**Chi tiết**: [xem tại đây](experiments/augmentation_matters_semi/README.md)

### Scheme 3: [Coming Soon]
Scheme semi-supervised thứ ba...

## Môi trường và cài đặt
Cài đặt:
```bash
conda create -n leaf_disease python=3.12
conda activate leaf_disease

# Cài đặt PyTorch
# Ví dụ cho CUDA 12.4:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Nếu dùng CUDA khác, xem tại: https://pytorch.org

# Cài đặt Ultralytics YOLOv11
cd ultralytics-8.3.225
pip install -e .
```

**Môi trường đã kiểm thử:**
- Python 3.12.6
- PyTorch 2.6.0 + CUDA 12.4
- Ultralytics 8.3.225

## Chuẩn bị Dataset

Xem [banana_dataset/README.md](banana_dataset/README.md) để biết chi tiết cấu trúc dataset và thống kê số lượng images/bounding box cho từng tập 2024, 2025.

**Model output locations:**
- Supervised baseline: `experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/`
- Semi-supervised: `experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/` (theo từng iteration)
- Augmentation scheme: `experiments/augmentation_matters_semi/scheme_augmat_general/runs/YOLOv11-AugSeg-Scheme-Fixed/`

## Training

### Supervised Baseline Training
Train các biến thể YOLOv11 chỉ với labeled data:
```bash
cd experiments/quoccuong_original
bash scripts/original_train.sh
```
Script này train YOLOv11-Base, YOLOv11-SA-Origin, YOLOv11-SA-Custom trong 400 epochs.

### Semi-Supervised Iterative Training (Scheme 1)
Chạy iterative pseudo-labeling theo hướng gốc:
```bash
cd experiments/quoccuong_original
bash scripts/original_semi_supervised.sh
```

### Semi-Supervised with EMA Teacher (Scheme 2)
Chạy huấn luyện semi-supervised nâng cao với strong augmentation và ema:
```bash
cd experiments/augmentation_matters_semi
bash scheme_augmat_general/scripts/augmat_semi_general.sh
```
Script chạy 35 iterations với confidence warmup và dynamic pseudo-label sampling.

### Evaluation
Đánh giá mô hình đã train:
```bash
yolo val model=experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt \
         data=experiments/quoccuong_original/configs/Banana_Disease_Dataset_Test.yaml \
         imgsz=1024
```

## Kết quả

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|---------|--------------|-----------|--------|
| YOLOv11-Base | - | - | - | -- |
| YOLOv11-SA-Origin | - | - | - | -|
| YOLOv11-SA-Custom | - | - | - | - 


## Acknowledgements
Đồ án này tham khảo và phát triển dựa trên các mã nguồn mở sau:
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Các kỹ thuật semi-supervised learning cho object detection

Xin cảm ơn tác giả các dự án đã công khai source code.

## Liên hệ
Nếu bạn gặp vấn đề khi chạy code, có thể liên hệ:
- phamthanh050204@gmail.com
