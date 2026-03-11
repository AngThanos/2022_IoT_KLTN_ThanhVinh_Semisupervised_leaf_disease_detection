# Quoc Cuong Original

## Scheme

![Scheme 1](../../resources/scheme.png)

Trong workflow này:
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


