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

Một số experiment có thêm dependency riêng (ví dụ `UniMatch-V2_Remake/requirements.txt`) — xem README con tương ứng trước khi chạy.

## Dataset

Bộ dữ liệu **không public** và không được đính kèm trong repo này (thư mục `/data/` bị `.gitignore`). [banana_dataset/README.md](banana_dataset/README.md) chỉ mô tả cấu trúc thư mục và số liệu thống kê để người đọc hình dung cách dữ liệu được tổ chức, không phải hướng dẫn tải về.

Nếu bạn muốn chạy lại các script train/eval bên dưới:

1. Chuẩn bị dataset của riêng bạn theo đúng cấu trúc thư mục mô tả trong [banana_dataset/README.md](banana_dataset/README.md) (`Banana_Dataset_2024_TrainValTest/`, `Banana_Dataset_2024_2025/Unlabeled_Images_2025/`, `Banana_Dataset_ValTest_2025/`, `Val_Clean/`, `Labeled_From_Val/`).
2. Sửa đường dẫn dataset trong file config của từng experiment (`configs/*.yaml`) để trỏ tới vị trí data trên máy bạn.

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
bash AugSeg_General/scripts/augmat_semi_general.sh
```

### 4. Train semi-supervised với iMAS

```bash
cd experiments/iMAS_Remake
bash single_run_yolo_sup.sh      # baseline có giám sát
bash single_run_yolo_semi.sh     # bán giám sát theo iMAS
```

Chi tiết cấu hình (`config_sup.yaml` / `config_semi.yaml`, đường dẫn dataset, pretrained weights) xem [experiments/iMAS_Remake/README.md](experiments/iMAS_Remake/README.md).

### 5. Train semi-supervised với UniMatch V2

```bash
cd experiments/UniMatch-V2_Remake
pip install -r requirements.txt
```

Xem [experiments/UniMatch-V2_Remake/README.md](experiments/UniMatch-V2_Remake/README.md) để biết lệnh chạy chính xác (`unimatch_v2_yolo.py`, `fixmatch.py`, `supervised.py`) và cách cấu hình.

### 6. Train semi-supervised với AD-MT

```bash
cd experiments/AD-MT_Remake
bash run_yolo_admt.sh
```

Chi tiết config/log xem [experiments/AD-MT_Remake/README.md](experiments/AD-MT_Remake/README.md).

### 7. Đánh giá mô hình

```bash
yolo val model=experiments/quoccuong_original/YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt \
         data=experiments/quoccuong_original/configs/Banana_Disease_Dataset_Test.yaml \
         imgsz=1024
```

> Đường dẫn `weights/best.pt` ở trên chỉ xuất hiện **sau khi** bạn đã train xong bằng một trong các lệnh phía trên — đây là ví dụ minh hoạ, hãy thay bằng checkpoint thật của bạn.

## Ghi chú quan trọng

- Với từng scheme, hãy mở README con tương ứng để xem chi tiết config, đường dẫn checkpoint và kết quả.
- Nếu đường dẫn dataset hoặc pretrained weights khác trên máy của bạn, hãy sửa trực tiếp trong file config của từng experiment.

## Acknowledgements

Đồ án này tham khảo và phát triển dựa trên các mã nguồn mở sau:

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Các kỹ thuật semi-supervised learning cho object detection

## Liên hệ

Nếu bạn gặp vấn đề khi chạy code, có thể liên hệ: phamthanh050204@gmail.com
