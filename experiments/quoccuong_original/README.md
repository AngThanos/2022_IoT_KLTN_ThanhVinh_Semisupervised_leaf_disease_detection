# 01 - Cuong Original

This folder stores the original workflow (legacy code) reorganized for easier management.

## Includes
- `scripts/original_train.sh`: run original supervised baseline training.
- `scripts/original_semi_supervised.sh`: run original iterative semi-supervised training.
- `configs/`: dataset YAML files used by this scheme (`Banana_Disease_Dataset*.yaml`).

## Notes
- Scripts are self-contained for this folder and intended to run from `quoccuong_original`.
- You can gradually migrate logic from legacy scripts into this folder later.
