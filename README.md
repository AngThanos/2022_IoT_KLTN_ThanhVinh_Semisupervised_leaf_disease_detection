<div align="center">    
 
# Xây dựng hệ thống học bán giám sát kết hợp tăng cường dữ liệu dựa trên mô hình YOLOv11 để phát hiện bệnh trên lá chuối. 

</div>

## Tổng quan
Repository này tập trung vào bài toán phát hiện bệnh trên lá chuối bằng YOLOv11 với nhiều hướng semi-supervised learning. Trang README này chỉ giữ thông tin ngắn gọn để bạn nắm nhanh cấu trúc repo và cách chạy chính.

## Repo này gồm gì

- [experiments/quoccuong_original/README.md](experiments/quoccuong_original/README.md): Scheme 1, iterative pseudo-labeling theo hướng gốc.
- [experiments/augmentation_matters_semi/README.md](experiments/augmentation_matters_semi/README.md): Scheme 2, teacher-student với EMA và augmentation.
- [experiments/iMAS_Remake/README.md](experiments/iMAS_Remake/README.md): iMAS remake cho YOLO detection.
- [experiments/UniMatch-V2_Remake/README.md](experiments/UniMatch-V2_Remake/README.md): UniMatch V2 remake cho YOLO detection.
- [experiments/AD-MT_Remake/README.md](experiments/AD-MT_Remake/README.md): AD-MT remake cho YOLO detection.
- [banana_dataset/README.md](banana_dataset/README.md): mô tả dataset và thống kê ảnh / bounding box.

## Cài đặt nhanh

```bash
conda create -n leaf_disease python=3.12
conda activate leaf_disease

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

cd ultralytics-8.3.225
pip install -e .
```

Môi trường đã kiểm thử:
- Python 3.12.6
- PyTorch 2.6.0 + CUDA 12.4
- Ultralytics 8.3.225

## Chạy nhanh

### 1. Train baseline

```bash
cd experiments/quoccuong_original
bash scripts/original_train.sh
```

### 2. Train semi-supervised theo Scheme 1

```bash
cd experiments/quoccuong_original
bash scripts/original_semi_supervised.sh
```

### 3. Train semi-supervised theo Scheme 2

```bash
cd experiments/augmentation_matters_semi
bash scheme_augmat_general/scripts/augmat_semi_general.sh
```

### 4. Đánh giá mô hình

```bash
yolo val model=experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt \
         data=experiments/quoccuong_original/configs/Banana_Disease_Dataset_Test.yaml \
         imgsz=1024
```

## Ghi chú quan trọng

- Với từng scheme, hãy mở README con tương ứng để xem chi tiết config, đường dẫn checkpoint và kết quả.
- Nếu đường dẫn dataset hoặc pretrained weights khác trên máy của bạn, hãy sửa trực tiếp trong file config của từng experiment.

## Acknowledgements

Đồ án này tham khảo và phát triển dựa trên các mã nguồn mở sau:
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Các kỹ thuật semi-supervised learning cho object detection

## Liên hệ

Nếu bạn gặp vấn đề khi chạy code, có thể liên hệ: phamthanh050204@gmail.com
