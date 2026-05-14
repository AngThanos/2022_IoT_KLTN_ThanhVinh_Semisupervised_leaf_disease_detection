# YOLO_SEMI — AugSeg cho YOLO Detection

**Tham khảo:**
- AugSeg (CVPR 2023) — code gốc: <https://github.com/zhenzhao/AugSeg>
- Paper: *Augmentation Matters: A Simple-Yet-Effective Approach to Semi-Supervised Semantic Segmentation* — <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Augmentation_Matters_A_Simple-Yet-Effective_Approach_to_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.pdf>

## 1. Huấn luyện

```bash
cd AugSeg_Remake/YOLO_SEMI
```

### 1.1. Train có giám sát (baseline)

```bash
python train_yolo_simple.py
```

### 1.2. Train bán giám sát (Teacher–Student với weak/strong augmentation)

Script `run.sh` nhận tham số `CONFIG GPUS PORT SEED`:

```bash
bash run.sh ../exps/release_25/vfl_custom/yolov11-sa-custom/config_semi.yaml 1 29500 42
```

**Trước khi chạy — cấu hình cần sửa:**

Mở file config tương ứng (ví dụ `../exps/release_25/vfl_custom/yolov11-sa-custom/config_semi.yaml`) và sửa:
1. `dataset.train.data_root` và `dataset.train.data_list` → trỏ tới folder ảnh có nhãn / unlabeled.
2. `dataset.val.data_root` và `dataset.val.data_list` → trỏ tới tập đánh giá.
3. `trainer.unsupervised.threshold` (mặc định `0.25`) → điều chỉnh ngưỡng pseudo-label nếu cần.
4. `net.encoder.pretrain` → trỏ tới file `.pt` pretrained tương ứng trong `models/`.

## 2. Đánh giá

Sau khi train xong, trong thư mục snapshot của exp tương ứng (`saver.snapshot_dir` trong config, ví dụ `checkpoint/` hoặc `checkpoints/`) sẽ có:
- `best.pt` — teacher EMA tại epoch tốt nhất, lưu theo định dạng Ultralytics, dùng trực tiếp với `yolo val` / `YOLO('best.pt')`.
- `best_student.pt` — student tại epoch tốt nhất, cùng định dạng Ultralytics.
- `last.pt` — teacher EMA tại epoch cuối, định dạng Ultralytics.
- `ckpt_best.pt`, `ckpt.pt` — checkpoint nội bộ của vòng train (chứa `model_state`, `optimizer_state`, `teacher_state`, `epoch`, ...), dùng để resume training, không nạp trực tiếp bằng `YOLO(...)` được.

### 2.1. Đánh giá bằng `yolo val`

```bash
yolo val \
    data=Banana_Disease_Dataset_Test.yaml \
    model=../exps/dynamic/varifocal_custom/yolov11/checkpoint/best.pt \
    imgsz=1024 \
    conf=0.1 \
    iou=0.1 \
    agnostic_nms=True \
    exist_ok=True
```

**Sửa đường dẫn theo model:**
- `data=...`: dataset yaml (chứa đường dẫn tập test)
- `model=...`: đường dẫn file `.pt` checkpoint cần đánh giá
- `imgsz`, `conf`, `iou`, ... → điều chỉnh tham số đánh giá nếu cần

## 3. Kết quả

Validation trên tập test (181 ảnh, 15052 instances, 2 classes). Giá trị **in đậm** = tăng so với baseline YOLOv11 cùng nhóm.

### 3.1. Ngưỡng tin cậy 1%

| Trọng số mất mát | Mô hình | Độ chính xác | Độ nhạy | mAP0.5 | mAP0.5:0.95 | F1-score |
|---|---|---|---|---|---|---|
| BCE | YOLOv11 | 52.1 | 50.6 | 45.5 | 20.2 | 51.3 |
| | YOLOv11-SA | 52.0 | 50.1 | 44.9 | 20.2 | 51.0 |
| | YOLOv11-SA custom | **52.4** | **51.0** | **46.1** | 20.2 | **51.7** |
| Varifocal | YOLOv11 | 52.5 | 47.8 | 47.3 | 21.5 | 50.0 |
| | YOLOv11-SA | 51.5 | **48.0** | 45.2 | 19.9 | 49.7 |
| | YOLOv11-SA custom | 49.7 | 44.6 | 43.1 | 19.0 | 47.0 |
| Varifocal Custom | YOLOv11 | 53.1 | 46.4 | 46.9 | 21.2 | 49.5 |
| | YOLOv11-SA | 51.4 | **48.0** | 45.1 | 19.9 | **49.6** |
| | YOLOv11-SA custom | 49.2 | 44.5 | 42.9 | 19.0 | 46.7 |

### 3.2. Ngưỡng tin cậy 25%

| Trọng số mất mát | Mô hình | Độ chính xác | Độ nhạy | mAP0.5 | mAP0.5:0.95 | F1-score |
|---|---|---|---|---|---|---|
| BCE | YOLOv11 | 56.9 | 54.8 | 54.7 | 24.4 | 55.8 |
| | YOLOv11-SA | 56.5 | 52.8 | 53.0 | 23.7 | 54.6 |
| | YOLOv11-SA custom | **58.3** | 51.7 | 53.1 | 23.1 | 54.8 |
| Varifocal | YOLOv11 | 51.9 | 47.1 | 46.3 | 21.0 | 49.4 |
| | YOLOv11-SA | 51.1 | **48.8** | **46.6** | **21.1** | **49.9** |
| | YOLOv11-SA custom | 47.9 | **48.0** | 46.1 | 20.4 | 47.9 |
| Varifocal Custom | YOLOv11 | 51.7 | 47.0 | 46.2 | 20.9 | 49.2 |
| | YOLOv11-SA | 51.6 | **48.3** | **46.7** | **21.1** | **49.9** |
| | YOLOv11-SA custom | 48.0 | **48.2** | 46.2 | 20.5 | 48.1 |

### 3.3. Ngưỡng tin cậy dynamic

| Trọng số mất mát | Mô hình | Độ chính xác | Độ nhạy | mAP0.5 | mAP0.5:0.95 | F1-score |
|---|---|---|---|---|---|---|
| BCE | YOLOv11 | 56.3 | 55.4 | 54.5 | 24.4 | 55.8 |
| | YOLOv11-SA | **57.5** | 51.8 | 52.6 | 23.4 | 54.5 |
| | YOLOv11-SA custom | **58.5** | 51.4 | 52.6 | 22.8 | 54.7 |
| Varifocal | YOLOv11 | 51.8 | 47.0 | 46.3 | 21.0 | 49.3 |
| | YOLOv11-SA | 51.2 | **48.7** | **46.8** | **21.1** | **49.9** |
| | YOLOv11-SA custom | 48.0 | **48.1** | 46.2 | 20.5 | 48.0 |
| Varifocal Custom | YOLOv11 | 51.8 | 46.9 | 46.1 | 21.0 | 49.2 |
| | YOLOv11-SA | 51.2 | **48.7** | **46.7** | **21.1** | **49.9** |
| | YOLOv11-SA custom | 48.0 | **48.1** | **46.2** | 20.5 | 48.0 |
