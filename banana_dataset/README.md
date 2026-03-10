# Banana Dataset

Cấu trúc dataset dùng cho bài toán semi-supervised YOLO nhằm phát hiện bệnh lá chuối.

---

## Thống kê Dataset

### Tổng quan (2024 vs 2025)

| Dataset     | Train | Validation | Test | Tổng số images | Images có bounding box | Số lượng bounding box |
| ----------- | ----- | ---------- | ---- | -------------- | ---------------------- | --------------------- |
| Banana 2024 | 2633  | 658        | 823  | 4114      | 4114            | 36474       |
| Banana 2025 (labeled) | 0     | 146        | 181  | 327       | 323      | 25058       |
| Banana 2025 (unlabeled) | 0 | 0 | 0 | 5871 | 0 | 0 |
| **Banana 2025 (tổng)** | **0** | **146** | **181** | **6198** | **323** | **25058** |

Ghi chú:
- Banana 2025 không có tập train được gán nhãn; phần `Unlabeled_Images_2025` được dùng cho pseudo-labeling trong semi-supervised learning.
- Trong tập Banana 2025, có 4 validation images không có bounding box

### Banana Dataset 2024

| Split      | Images   |
| ---------- | -------- |
| Train      | 2633     |
| Validation | 658      |
| Test       | 823      |
| **Total**  | **4114** |

Toàn bộ 4,114 images trong dataset 2024 đều có bounding box annotations.

| Dataset     | Labeled Images | Số lượng bounding box |
| ----------- | -------------- | --------------------- |
| Banana 2024 | 4114           | 36474          |

---

### Banana Dataset 2025

| Split      | Images  |
| ---------- | ------- |
| Validation | 146     |
| Test       | 181     |
| **Total**  | **327** |

| Dataset                  | Images có bounding box | Số lượng bounding box |
| ------------------------ | ---------------------- | --------------------- |
| Banana 2025 (Validation) | 142                        | 10187          |
| Banana 2025 (Test)       | 181                        | 14871          |
| **Total**                | **323**                    | **25058**      |

---

### Dữ liệu dùng cho Semi-Supervised Learning

| Dataset               | Images | Số lượng bounding box | Mô tả                                                       |
| --------------------- | ------ | --------------------- | ----------------------------------------------------------- |
| Unlabeled_Images_2025 | 5871   | 0              | Unlabeled images dùng để sinh pseudo-label                    |
| Val_Clean             | 72     | 5380           | Validation dataset dùng để đánh giá scheme1_augmat_general    |
| Labeled_From_Val      | 76     | 4907           | Labeled subset dùng làm seed data cho semi-supervised training |

---

## Cấu trúc thư mục

| Folder                                            | Mục đích                                                                            |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `Banana_Dataset_2024_TrainValTest/`               | Supervised training dataset (train/val/test splits từ images năm 2024)             |
| `Banana_Dataset_2024_2025/Unlabeled_Images_2025/` | Unlabeled image pool cho semi-supervised learning                                  |
| `Banana_Dataset_ValTest_2025/`                    | Validation và test images thu thập trong năm 2025                                  |
| `Val_Clean/`                                      | Validation dataset đã làm sạch để đánh giá scheme1 augmentation                    |
| `Labeled_From_Val/`                               | Labeled subset nhỏ tách từ validation data cho các thí nghiệm semi-supervised      |


