# Augmentation Matters Semi
# YOLO_SEMI

## Scheme 1: Iterative Pseudo-Labeling (YOLO_SEMI)

![Scheme 1](../../resources/schem.png)

Trong nhánh này:
- `train_yolo_simple.py` là bước train gốc (baseline YOLO thường).
- `train_semi.py` (hoặc `run.sh`) là Scheme semi-supervised (iterative pseudo-labeling): teacher sinh pseudo-label cho ảnh unlabeled rồi train lặp theo epoch.

## Lệnh chạy

```bash
cd AugSeg_Remake/YOLO_SEMI
```

Train gốc:

```bash
python train_yolo_simple.py
```

Chạy Scheme semi-supervised:

```bash
# Cách 1: chạy bằng launcher script
bash run.sh exps/yolov11-base-400/config_semi.yaml 1 29500 42

# Cách 2: gọi torchrun trực tiếp
torchrun --nproc_per_node=1 --master_port=29500 train_semi.py \
    --config exps/yolov11-base-400/config_semi.yaml \
    --seed 42
```

## Lệnh đánh giá

Từ thư mục `AugSeg_Remake/YOLO_SEMI`, chạy:

### Đánh giá theo pipeline custom trong YOLO_SEMI (.pth)

```bash
# Đánh giá checkpoint tốt nhất (teacher_state) bằng validate custom
torchrun --nproc_per_node=1 validate_from_pth.py \
    --config exps/yolov11-base-400/config_semi.yaml \
    --weights exps/yolov11-base-400/checkpoints/ckpt_best.pth \
    --state-key teacher_state \
    --prefix VAL

# Nếu muốn đánh giá student_state:
torchrun --nproc_per_node=1 validate_from_pth.py \
    --config exps/yolov11-base-400/config_semi.yaml \
    --weights exps/yolov11-base-400/checkpoints/ckpt_best.pth \
    --state-key model_state \
    --prefix VAL-STUDENT
```

## Bảng kết quả

### Bộ dữ liệu 2024 (đã gán nhãn)

<table>
    <thead>
        <tr>
            <th>Trọng số mất mát</th>
            <th>Mô hình</th>
            <th>Độ chính xác</th>
            <th>Độ nhạy</th>
            <th>mAP0.5</th>
            <th>mAP0.5:0.95</th>
            <th>F1-score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">-</td>
            <td>YOLOv11</td>
            <td>79.4</td>
            <td>74.6</td>
            <td>81.1</td>
            <td>54.1</td>
            <td>76.9</td>
        </tr>
        <tr>
            <td>YOLOv11-SA</td>
            <td>79.3</td>
            <td>74.7</td>
            <td>81.6</td>
            <td>54.7</td>
            <td>76.9</td>
        </tr>
        <tr>
            <td>YOLOv11-SA custom</td>
            <td>78.8</td>
            <td>75.6</td>
            <td>81.5</td>
            <td>54.4</td>
            <td>77.2</td>
        </tr>
    </tbody>
</table>

### Đánh giá tập 2025 (chạy theo từng iter, chọn kết quả tốt nhất)

```bash
# 1) Sửa đường dẫn val trong exps/yolov11-base-400/config_semi.yaml về tập 2025 cần test.
# 2) Chạy val cho nhiều checkpoint .pth và tự chọn kết quả mAP50-95 tốt nhất.

for ckpt in exps/*/checkpoints/ckpt_best.pth exps/*/checkpoints/ckpt.pth
do
    [ -f "$ckpt" ] || continue
    run_name=$(basename "$(dirname "$(dirname "$ckpt")")")
    torchrun --nproc_per_node=1 validate_from_pth.py \
        --config exps/yolov11-base-400/config_semi.yaml \
        --weights "$ckpt" \
        --state-key teacher_state \
        --prefix "$run_name"
done

# Tuỳ chọn: convert .pth -> .pt để dùng yolo val chuẩn Ultralytics
python convert_pth_to_pt.py \
    --weights exps/yolov11-base-400/checkpoints/ckpt_best.pth \
    --pretrain ../../yolov8n.pt \
    --state-key teacher_state

```

Sau khi val ra kết quả, tự chọn run có mAP50-95 cao nhất

### Bộ dữ liệu 2025 (chưa gán nhãn) - Ngưỡng tin cậy 25% 

Ghi chú: trong code hiện tại, ngưỡng pseudo-label được đọc từ `trainer.unsupervised.threshold` và đang đặt `0.25`.

<table>
    <thead>
        <tr>
            <th>Trọng số mất mát</th>
            <th>Mô hình</th>
            <th>Độ chính xác</th>
            <th>Độ nhạy</th>
            <th>mAP0.5</th>
            <th>mAP0.5:0.95</th>
            <th>F1-score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3">BCE</td>
            <td>YOLOv11</td>
            <td>66.0</td>
            <td>36.8</td>
            <td>50.1</td>
            <td>24.2</td>
            <td>47.3</td>
        </tr>
        <tr>
            <td>YOLOv11-SA</td>
            <td>58.0</td>
            <td>50.3</td>
            <td>51.4</td>
            <td>21.9</td>
            <td>53.9</td>
        </tr>
        <tr>
            <td>YOLOv11-SA custom</td>
            <td>58.5</td>
            <td>50.9</td>
            <td>52.9</td>
            <td>24.0</td>
            <td>54.4</td>
        </tr>
        <tr>
            <td rowspan="3">Varifocal</td>
            <td>YOLOv11</td>
            <td>39.2</td>
            <td>42.8</td>
            <td>35.9</td>
            <td>15.9</td>
            <td>40.9</td>
        </tr>
        <tr>
            <td>YOLOv11-SA</td>
            <td>44.8</td>
            <td>40.7</td>
            <td>37.7</td>
            <td>16.9</td>
            <td>42.7</td>
        </tr>
        <tr>
            <td>YOLOv11-SA custom</td>
            <td>47.7</td>
            <td>45.1</td>
            <td>42.7</td>
            <td>19.3</td>
            <td>46.4</td>
        </tr>
        <tr>
            <td rowspan="3">Varifocal Custom</td>
            <td>YOLOv11</td>
            <td>41.3</td>
            <td>40.7</td>
            <td>36.1</td>
            <td>15.6</td>
            <td>41.0</td>
        </tr>
        <tr>
            <td>YOLOv11-SA</td>
            <td>41.5</td>
            <td>42.1</td>
            <td>35.9</td>
            <td>15.6</td>
            <td>41.8</td>
        </tr>
        <tr>
            <td>YOLOv11-SA custom</td>
            <td>46.8</td>
            <td>43.9</td>
            <td>40.1</td>
            <td>17.7</td>
            <td>45.3</td>
        </tr>
    </tbody>
</table>
