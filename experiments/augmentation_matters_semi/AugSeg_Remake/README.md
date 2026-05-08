# YOLO_SEMI — AugSeg cho YOLO Detection

![Scheme 3](../../../resources/ppAug_Remake.png)


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
bash run.sh ../exps/conf_025/varifocal_custom/yolov11-sa-custom/config_semi.yaml 1 29500 42
```

**Trước khi chạy — cấu hình cần sửa:**

Mở file config tương ứng (ví dụ `../exps/conf_025/varifocal_custom/yolov11-sa-custom/config_semi.yaml`) và sửa:
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
