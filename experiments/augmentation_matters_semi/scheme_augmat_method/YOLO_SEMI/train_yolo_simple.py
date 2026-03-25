"""Simple YOLO training script using Ultralytics directly."""
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train with your data
results = model.train(
    data='data/data_yolo/data_yolo.yaml',
    epochs=100,
    imgsz=640,
    batch=4,
    patience=20,  # Early stopping
    lr0=0.001,    # Lower learning rate for fine-tuning
    augment=True, # Enable augmentation
    mosaic=1.0,   # Mosaic augmentation
    mixup=0.1,    # Mixup augmentation
    project='runs/yolo_simple',
    name='train'
)

print("Training complete!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
