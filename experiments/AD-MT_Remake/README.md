# AD-MT_Remake — AD-MT cho YOLO Detection


Bản remake của AD-MT (Alternate Diverse Teaching) áp dụng cho semi-supervised YOLO object detection trên dataset banana, mượn data pipeline kiểu iMAS.

**Tham khảo:**
- AD-MT (paper): *Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation* — <https://arxiv.org/abs/2311.17325>
- AD-MT (code gốc): <https://github.com/zhenzhao/AD-MT>
- iMAS (paper): *Instance-specific and Model-adaptive Supervision for Semi-supervised Semantic Segmentation* — <https://arxiv.org/abs/2211.11335>
- iMAS (code gốc, dùng cho data pipeline): <https://github.com/zhenzhao/iMAS>

## 1. Tổng quan

Pipeline gồm hai thành phần chính từ AD-MT:
- **RPA (Random Periodic Alternate Updating):** một student, hai teacher non-trainable; teacher được EMA-update **luân phiên** theo chu kỳ ngẫu nhiên để giữ tính đa dạng.
- **CCM (Conflict-Combating Module):** kết hợp pseudo-label từ hai teacher bằng entropy-weighted ensembling, xử lý vùng hai teacher mâu thuẫn dựa trên độ tự tin của student.

Mã nguồn chính nằm trong [code/train_yolo_admt.py](code/train_yolo_admt.py); module RPA + CCM ở [code/train_utils.py](code/train_utils.py); data loader ở [code/dataloaders/yolo_loader.py](code/dataloaders/yolo_loader.py).

## 2. Huấn luyện

### 2.1. Lệnh chạy

Script `run_yolo_admt.sh` nhận `GPU_ID` và `CONFIG`:

```bash
bash run_yolo_admt.sh 0 config_yolo_admt.yml
```

Chạy toàn bộ ablation tuần tự:

```bash
bash run_ablation.sh 0
```

### 2.2. Trước khi chạy — cấu hình cần sửa

Mở `cfgs/config_yolo_admt.yml` (hoặc file ablation trong `cfgs/ablation/`) và sửa:
1. `root_path` → folder chứa `banana_data`.
2. `dataset_yaml` → file YAML mô tả dataset (theo format Ultralytics).
3. `model` → đường dẫn `.pt` pretrained YOLO dùng để init student/teacher.
4. `res_path`, `exp` → nơi lưu kết quả (`results/<exp>/<model_path_escaped>/`).
5. `conf_threshold`, `consistency`, `ema_decay` → tham số semi-supervised.
6. `alt_param_conflict_weight` → trọng số CCM (đặt `0` để tắt CCM, xem ablation).
7. `alt_flag_updating_period_random`, `alt_param_updating_period_iters` → cấu hình RPA.

Ngoài ra cần sửa dòng `cd /home/jupyter-iec2021iot13/Vinh/AD-MT` trong [run_yolo_admt.sh](run_yolo_admt.sh) cho khớp đường dẫn repo thực tế (`AD-MT_Remake`).

## 3. Đánh giá

Trong thư mục `${res_path}/${exp}/<model_path_escaped>/` sẽ có:
- `best_tea_model.pt` — teacher EMA tại epoch tốt nhất (theo `mAP50` của teacher), định dạng Ultralytics, dùng trực tiếp với `yolo val` / `YOLO('best_tea_model.pt')`.
- `best_stu_model.pt` — student tại epoch tốt nhất (theo `mAP50` của student), cùng định dạng Ultralytics.
- `log/` — TensorBoard logs và CSV (`train_*.csv`, `val_*.csv`).
- `log.txt` — log chính của quá trình train.

Đánh giá bằng `yolo val`:

```bash
yolo val \
    data=Banana_Disease_Dataset_Test.yaml \
    model=results/banana/admt_yolo/<model_path_escaped>/best_tea_model.pt \
    imgsz=640 \
    conf=0.1 \
    iou=0.1 \
    agnostic_nms=True \
    exist_ok=True
```

**Sửa đường dẫn theo model:**
- `data=...`: dataset yaml chứa đường dẫn tập test.
- `model=...`: file `.pt` checkpoint cần đánh giá (`best_tea_model.pt` hoặc `best_stu_model.pt`).
- `imgsz`, `conf`, `iou`, ... → điều chỉnh tham số đánh giá nếu cần.

## 4. Ablation

Các config trong `cfgs/ablation/`:
- `ablation_1teacher.yml` — chỉ 1 teacher (tắt RPA + CCM).
- `ablation_no_ccm.yml` — bật RPA, tắt CCM (`alt_param_conflict_weight: 0`).
- `ablation_ccm_tea_only.yml` — CCM chỉ giữa hai teacher, không dùng student arbitration.
- `ablation_fixed_period.yml` — RPA dùng chu kỳ cố định thay vì ngẫu nhiên.

## 5. Citation

```bibtex
@article{zhao2023alternate,
  title={Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation},
  author={Zhao, Zhen and Wang, Zicheng and Wang, Longyue and Yuan, Yixuan and Zhou, Luping},
  journal={arXiv preprint arXiv:2311.17325},
  year={2023}
}
```
