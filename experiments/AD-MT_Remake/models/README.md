# Model Storage

This folder stores original model checkpoint files used by this project.

## What to put here
- `.pt` files (YOLO checkpoints, teacher models, student initial weights)
- Optional model metadata files (for example: `.txt`, `.json`, `.yaml`) that describe each checkpoint

## Naming suggestion
Use clear names so training scripts are easy to configure later, for example:
- `teacher_best.pt`
- `student_init.pt`
- `yolo_sup_best.pt`
- `yolo_semi_best.pt`

## Notes
- Keep this folder at the workspace root, at the same level as `imas/`.
- Do not place datasets or logs here.
