# Quoc Cuong Original

## Scheme 1: Iterative Pseudo-Labeling (Original)

![Scheme 1](../../resources/scheme1.png)

Trong nhánh này:
- `original_train.sh` là bước train gốc (khởi tạo model ban đầu).
- `original_semi_supervised.sh` mới là Scheme semi-supervised (iterative pseudo-labeling): sinh pseudo-label cho `Unlabeled_Images_2025` rồi train lặp theo iteration.

## Lệnh chạy

```bash
cd experiments/quoccuong_original
```

Train gốc:

```bash
bash scripts/original_train.sh
```

Chạy Scheme semi-supervised:

```bash
bash scripts/original_semi_supervised.sh
```

## Bảng kết quả

### Bộ dữ liệu 2025 (chưa gán nhãn) - Ngưỡng tin cậy 1%

| Trọng số mất mát | Mô hình | Độ chính xác | Độ nhạy | mAP0.5 | mAP0.5:0.95 | F1-score |
|---|---|---:|---:|---:|---:|---:|
| BCE | YOLOv11 | 45.2 | 38.6 | 39.9 | 17.2 | 41.6 |
| BCE | YOLOv11-SA | 46.5 | 37.5 | 40.2 | 17.4 | 41.5 |
| BCE | YOLOv11-SA custom | 47.9 | 38.9 | 41.5 | 18.3 | 42.9 |
| Varifocal | YOLOv11 | 48.1 | 39.7 | 35.8 | 14.1 | 43.5 |
| Varifocal | YOLOv11-SA | 48.1 | 41.7 | 36.9 | 14.5 | 44.7 |
| Varifocal | YOLOv11-SA custom | 45.7 | 42.6 | 38.2 | 15.1 | 44.1 |
| Varifocal Custom | YOLOv11 | 48.8 | 40.7 | 39.7 | 16.4 | 44.4 |
| Varifocal Custom | YOLOv11-SA | 47.3 | 41.4 | 39.4 | 16.3 | 44.2 |
| Varifocal Custom | YOLOv11-SA custom | 49.7 | 42.9 | 40.5 | 16.1 | 46.1 |

### Bộ dữ liệu 2025 (chưa gán nhãn) - Ngưỡng tin cậy 25%

| Trọng số mất mát | Mô hình | Độ chính xác | Độ nhạy | mAP0.5 | mAP0.5:0.95 | F1-score |
|---|---|---:|---:|---:|---:|---:|
| BCE | YOLOv11 | 72.7 | 11.0 | 41.0 | 21.7 | 19.1 |
| BCE | YOLOv11-SA | 73.1 | 16.4 | 44.1 | 22.3 | 26.8 |
| BCE | YOLOv11-SA custom | 71.9 | 14.4 | 42.3 | 21.2 | 24.0 |
| Varifocal | YOLOv11 | 49.9 | 41.9 | 41.9 | 19.0 | 45.6 |
| Varifocal | YOLOv11-SA | 46.2 | 42.1 | 40.4 | 18.2 | 44.1 |
| Varifocal | YOLOv11-SA custom | 51.2 | 44.3 | 44.2 | 19.5 | 47.5 |
| Varifocal Custom | YOLOv11 | 50.7 | 44.1 | 45.0 | 20.6 | 47.2 |
| Varifocal Custom | YOLOv11-SA | 48.9 | 45.5 | 45.3 | 20.3 | 47.1 |
| Varifocal Custom | YOLOv11-SA custom | 50.7 | 45.7 | 46.4 | 21.2 | 48.1 |


