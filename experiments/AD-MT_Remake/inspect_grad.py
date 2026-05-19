import torch
import sys
import os

from unittest.mock import MagicMock
# Mock everything that could cause issues
for m in ['torch.utils.tensorboard', 'tensorboard', 'tensorflow', 'keras', 'IPython', 'pygments', 'fsspec']:
    sys.modules[m] = MagicMock()

def setup_yolo_model_minimal(weights_path):
    from ultralytics import YOLO
    model = YOLO(weights_path).model
    return model

weights_path = "/home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/YOLO_SEMI/models/YOLOv11-Base-400/best.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    sys.path.append(os.path.join(os.getcwd(), 'code'))
    # Use a safer import or fallback
    from train_semi import setup_yolo_model
    model = setup_yolo_model(weights_path)
except Exception:
    model = setup_yolo_model_minimal(weights_path)

model.to(device)
model.train()
# Force requires_grad for inspection
for p in model.parameters():
    p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
first_params = list(model.parameters())[:3]
requires_grad_flags = [p.requires_grad for p in first_params]

print(f"Trainable params: {trainable_params}")
print(f"First few params requires_grad: {requires_grad_flags}")

dummy_input = torch.randn(2, 3, 640, 640, device=device)
# YOLO models often return a list/tuple in train mode or validation mode
preds = model(dummy_input)

print(f"Preds type: {type(preds)}")
if isinstance(preds, (list, tuple)):
    print(f"Preds len: {len(preds)}")
    if len(preds) > 0:
        p0 = preds[0]
        if torch.is_tensor(p0):
             print(f"First pred head requires_grad: {p0.requires_grad}")
        elif isinstance(p0, (list, tuple)) and len(p0) > 0 and torch.is_tensor(p0[0]):
             print(f"First pred head element requires_grad: {p0[0].requires_grad}")
elif torch.is_tensor(preds):
    print(f"Preds requires_grad: {preds.requires_grad}")
