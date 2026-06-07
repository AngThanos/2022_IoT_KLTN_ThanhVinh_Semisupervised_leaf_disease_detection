# UniMatch-V2_Remake — UniMatch V2 cho YOLO Detection

Bản remake của **UniMatch V2** áp dụng cho semi-supervised YOLO object detection trên dataset banana.

**Tham khảo:**
- UniMatch V2 (TPAMI 2025, paper): *UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation* — <https://arxiv.org/abs/2410.10777>
- UniMatch V2 (code gốc): <https://github.com/LiheYoung/UniMatch-V2>
- UniMatch V1 (tiền thân): <https://github.com/LiheYoung/UniMatch>

![UniMatch-V2 diagram](docs/pipeline_unimatchv2.png)

<p align="left">
<img src="./docs/framework.png" width=80% height=80%>
</p>

## 1. Tổng quan

FixMatch-style semi-supervised detection với 1 student + 1 teacher EMA:
- **Dual-stream consistency** (s1, s2) + Complementary Channel-wise Dropout — buộc 2 stream học từ feature views khác nhau
- **CutMix** giữa các ảnh unlabeled, kích hoạt độc lập cho s1/s2
- **Online EMA teacher** (γ = min(1 − 1/(t+1), 0.996)) + Loss L = (L_x + L_u) / 2

## 2. Huấn luyện

### 2.1. Train bán giám sát (UniMatch V2)

```bash
python unimatch_v2_yolo.py --config exps/conf_001/varifocal/yolov11-sa/config_semi.yaml
```

### 2.2. Trước khi chạy — cấu hình cần sửa

Mở `exps/<conf>/<loss>/<model>/config_semi.yaml` và sửa các đường dẫn:
1. `data.*` paths (root, labeled_images, unlabeled_images, val_images, val_labels) → đối với `banana_data`.
2. `student.init_pt`, `teacher.pt` → file `.pt` pretrained YOLO.
3. `project.output_root` → nơi lưu checkpoint và log.

## 3. Đánh giá

```bash
yolo val \
    data=Banana_Disease_Dataset_Test.yaml \
    model=exps/conf_001/varifocal/yolov11-sa/online/best.pt \
    imgsz=1024 \
    conf=0.1 \
    iou=0.1 \
    agnostic_nms=True \
    exist_ok=True
```

Sửa `data` (dataset yaml) và `model` (checkpoint `.pt`); điều chỉnh `imgsz`, `conf`, `iou` nếu cần.

## 4. Kết quả

Validation trên tập test (181 ảnh, 15052 instances, 2 classes) với 3 cấu hình ngưỡng tin cậy pseudo-label: `1%`, `25%`, và `dynamic`.

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
<td>0.57</td><td>0.51</td><td><b>0.499</b></td><td>0.214</td><td>0.538</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.533</td><td>0.508</td><td>0.449</td><td>0.192</td><td>0.520</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.547</td><td>0.497</td><td>0.458</td><td>0.2</td><td>0.520</td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td>0.494</td><td>0.483</td><td>0.455</td><td>0.199</td><td>0.488</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.521</td><td>0.487</td><td>0.456</td><td>0.197</td><td>0.503</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.499</td><td>0.484</td><td>0.455</td><td>0.195</td><td>0.491</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>0.498</td><td>0.481</td><td>0.456</td><td>0.199</td><td>0.489</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.525</td><td>0.486</td><td>0.455</td><td>0.194</td><td>0.504</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.507</td><td>0.486</td><td>0.453</td><td>0.195</td><td>0.496</td></tr>
<tr>
<td rowspan="9">25%</td>
<td rowspan="3">BCE</td>
<td>YOLOv11</td>
<td>0.537</td><td>0.511</td><td>0.472</td><td>0.206</td><td>0.524</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.566</td><td>0.521</td><td><b>0.491</b></td><td>0.21</td><td>0.543</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.576</td><td>0.514</td><td><b>0.498</b></td><td>0.214</td><td>0.544</td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td>0.437</td><td>0.46</td><td>0.393</td><td>0.179</td><td>0.448</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.48</td><td>0.461</td><td>0.427</td><td>0.19</td><td>0.470</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.415</td><td>0.482</td><td>0.41</td><td>0.178</td><td>0.446</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>0.44</td><td>0.457</td><td>0.392</td><td>0.18</td><td>0.448</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.509</td><td>0.475</td><td>0.439</td><td>0.193</td><td>0.491</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.435</td><td>0.499</td><td>0.442</td><td>0.193</td><td>0.465</td></tr>
<tr>
<td rowspan="9">dynamic</td>
<td rowspan="3">BCE</td>
<td>YOLOv11</td>
<td>0.546</td><td>0.501</td><td>0.456</td><td>0.198</td><td>0.523</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.545</td><td>0.485</td><td>0.453</td><td>0.197</td><td>0.513</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.563</td><td>0.505</td><td><b>0.476</b></td><td>0.208</td><td>0.533</td></tr>
<tr>
<td rowspan="3">Varifocal</td>
<td>YOLOv11</td>
<td>0.495</td><td>0.488</td><td>0.449</td><td>0.195</td><td>0.491</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.517</td><td>0.482</td><td>0.45</td><td>0.195</td><td>0.499</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.452</td><td>0.493</td><td>0.443</td><td>0.195</td><td>0.472</td></tr>
<tr>
<td rowspan="3">Varifocal Custom</td>
<td>YOLOv11</td>
<td>0.501</td><td>0.482</td><td>0.45</td><td>0.195</td><td>0.491</td>
</tr>
<tr><td>YOLOv11-SA</td><td>0.518</td><td>0.477</td><td>0.441</td><td>0.193</td><td>0.496</td></tr>
<tr><td>YOLOv11-SA custom</td><td>0.449</td><td>0.499</td><td>0.447</td><td>0.197</td><td>0.473</td></tr>
</tbody>
</table>
