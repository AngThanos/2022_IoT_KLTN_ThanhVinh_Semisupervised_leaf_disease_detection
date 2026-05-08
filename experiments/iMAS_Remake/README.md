# iMAS_Remake — iMAS cho YOLO Detection

Bản remake của iMAS (Instance-specific and Model-adaptive Supervision) áp dụng cho semi-supervised YOLO object detection trên dataset banana.

**Tham khảo:**
- iMAS (CVPR 2023, paper): *Instance-specific and Model-adaptive Supervision for Semi-supervised Semantic Segmentation* — <https://arxiv.org/abs/2211.11335>
- iMAS (code gốc): <https://github.com/zhenzhao/iMAS>

![iMAS diagram](docs/ppiMAS_Remake.png)

## 1. Tổng quan

Pipeline teacher–student với EMA, mở rộng theo 3 đóng góp chính của iMAS áp dụng cho detection:
1. **Hardness evaluation** — đánh giá độ khó của mỗi ảnh unlabeled từ dự đoán teacher/student.
2. **Adaptive augmentation** — pha trộn weak/strong augmentation theo độ khó từng ảnh.
3. **Adaptive CutMix** — ghép cặp hard–easy, kích hoạt CutMix theo hardness trung bình.

Mã nguồn chính:
- [train_yolo_sup.py](train_yolo_sup.py) — train có giám sát (baseline).
- [train_yolo_semi.py](train_yolo_semi.py) — train bán giám sát (iMAS).
- [eval_yolo.py](eval_yolo.py) — đánh giá checkpoint trên val set.
- [imas/](imas/) — bridge sang Ultralytics + helper.
- [exps/yolo_det/](exps/yolo_det/) — config cho phần YOLO.

Các thư mục `exps/voc_*`, `exps/citys_*`, `exps/city_sups`, `train_semi.py`, `train_sup.py` là phần segmentation gốc của iMAS, vẫn giữ lại để tham khảo.

## 2. Huấn luyện

### 2.1. Train có giám sát (baseline)

```bash
bash single_run_yolo_sup.sh
```

Tương đương:

```bash
python train_yolo_sup.py --config ./exps/yolo_det/config_sup.yaml --seed 2
```

Có thể đổi config qua biến môi trường: `CONFIG=./exps/yolo_det/config_sup.yaml SEED=42 bash single_run_yolo_sup.sh`.

### 2.2. Train bán giám sát (iMAS)

```bash
bash single_run_yolo_semi.sh
```

Tương đương:

```bash
python train_yolo_semi.py --config ./exps/yolo_det/config_semi.yaml --seed 2
```

### 2.3. Trước khi chạy — cấu hình cần sửa

Mở `exps/yolo_det/config_semi.yaml` (hoặc `config_sup.yaml`) và sửa:
1. `data.root` → folder chứa `banana_data`.
2. `data.labeled_images` / `labeled_labels` / `unlabeled_images` / `val_images` / `val_labels` → đường dẫn tương đối tới `data.root`.
3. `student.init_pt`, `teacher.pt` → đường dẫn `.pt` pretrained YOLO trong `models/`.
4. `train.imgsz`, `train.batch`, `train.epochs`, `train.device` → tham số train.
5. `loss.lambda_u`, `pseudo.conf_start` / `conf_end`, `teacher.ema_decay` → tham số semi-supervised.
6. `semi.adaptive_aug`, `semi.adaptive_cutmix`, `hardness.enabled` → bật/tắt các đóng góp của iMAS.
7. `project.output_root` → nơi lưu checkpoint và log.

## 3. Đánh giá

### 3.1. Vị trí checkpoint

Sau khi train xong, trong `${project.output_root}/online/` sẽ có:
- `best.pt` — checkpoint tại epoch có `mAP50` val cao nhất, định dạng Ultralytics, dùng trực tiếp với `yolo val` / `YOLO('best.pt')`.
- `last.pt` — checkpoint tại epoch cuối, định dạng Ultralytics, hỗ trợ resume (`train.resume: auto`).

### 3.2. Đánh giá bằng `eval_yolo.py`

```bash
python eval_yolo.py \
    --config ./exps/yolo_det/config_sup.yaml \
    --weights ./release/SA-Origin/varifocal_custom/dynamic_conf/online/best.pt
```

Kết quả lưu ở `${project.output_root}/eval/<timestamp>/eval_report.json` và `eval/latest/`.

### 3.3. Đánh giá bằng `yolo val`

```bash
yolo val \
    data=./data/banana_data/meta/dataset_supervised.yaml \
    model=./release/SA-Origin/varifocal_custom/dynamic_conf/online/best.pt \
    imgsz=1024 \
    conf=0.1 \
    iou=0.1 \
    agnostic_nms=True \
    exist_ok=True
```

**Sửa đường dẫn theo model:**
- `data=...`: dataset yaml chứa đường dẫn tập đánh giá.
- `model=...`: file `.pt` checkpoint cần đánh giá.
- `imgsz`, `conf`, `iou`, ... → điều chỉnh tham số đánh giá nếu cần.

## 4. Kết quả

Đánh giá trên bộ dữ liệu **2025 (chưa gán nhãn)** với 3 cấu hình ngưỡng tin cậy pseudo-label: `1%`, `25%`, và `dynamic`.

<table>
<thead>
<tr>
<th>Ngưỡng tin cậy</th>
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
<td rowspan="9">1%</td>
<td rowspan="3">BCE</td>
<td>YOLOv11</td>
<td>56.8</td><td>50.8</td><td>51.5</td><td>22.5</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>54.2</td><td>49.7</td><td>47.0</td><td>20.6</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>54.3</b></td><td><b>51.1</b></td><td><b>48.8</b></td><td><b>21.5</b></td><td><b>00</b></td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td><b>53.4</b></td><td>49.7</td><td>49.0</td><td>22.1</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td><b>54.7</b></td><td>49.3</td><td>48.8</td><td>22.1</td><td><b>00</b></td></tr>
<tr><td>YOLOv11-SA custom</td><td>48.6</td><td><b>47.9</b></td><td><b>44.9</b></td><td><b>19.8</b></td><td>00</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>53.5</td><td>49.9</td><td>49.2</td><td><b>22.1</b></td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>54.2</td><td>47.3</td><td>47.7</td><td>21.5</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>48.6</b></td><td><b>47.7</b></td><td><b>45.6</b></td><td>20.1</td><td><b>00</b></td></tr>
<tr>
<td rowspan="9">25%</td>
<td rowspan="3">BCE</td>
<td>YOLOv11</td>
<td>54.8</td><td>48.7</td><td>48.0</td><td>21.2</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>44.0</td><td>47.9</td><td>36.5</td><td>15.5</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>38.7</b></td><td><b>46.8</b></td><td><b>31.7</b></td><td><b>13.6</b></td><td><b>00</b></td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td><b>48.5</b></td><td>48.0</td><td>42.7</td><td>18.4</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td><b>46.2</b></td><td>46.9</td><td>41.1</td><td>18.1</td><td><b>00</b></td></tr>
<tr><td>YOLOv11-SA custom</td><td>44.6</td><td><b>47.8</b></td><td><b>40.3</b></td><td><b>17.4</b></td><td>00</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>48.3</td><td>47.5</td><td>42.9</td><td><b>18.7</b></td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>45.4</td><td>47.2</td><td>40.4</td><td>17.5</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>47.6</b></td><td><b>43.5</b></td><td><b>40.2</b></td><td>17.2</td><td><b>00</b></td></tr>
<tr>
<td rowspan="9">dynamic</td>
<td rowspan="3">BCE</td>
<td>YOLOv11</td>
<td>59.4</td><td>49.9</td><td>51.8</td><td>24.3</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>50.2</td><td>47.0</td><td>42.4</td><td>18.5</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>57.5</b></td><td><b>51.8</b></td><td><b>52.6</b></td><td><b>23.5</b></td><td><b>00</b></td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td><b>47.8</b></td><td>48.0</td><td>43.2</td><td>19.1</td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td><b>46.1</b></td><td>47.4</td><td>41.6</td><td>18.1</td><td><b>00</b></td></tr>
<tr><td>YOLOv11-SA custom</td><td>44.7</td><td><b>47.1</b></td><td><b>40.3</b></td><td><b>17.5</b></td><td>00</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>43.9</td><td>47.0</td><td>39.7</td><td><b>17.6</b></td><td>00</td>
</tr>
<tr><td>YOLOv11-SA</td><td>47.2</td><td>45.8</td><td>40.8</td><td>17.5</td><td>00</td></tr>
<tr><td>YOLOv11-SA custom</td><td><b>45.1</b></td><td><b>46.8</b></td><td><b>40.7</b></td><td>17.8</td><td><b>00</b></td></tr>
</tbody>
</table>