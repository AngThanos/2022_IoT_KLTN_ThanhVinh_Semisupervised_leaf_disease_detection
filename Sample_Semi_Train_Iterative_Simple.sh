#!/bin/bash

yolo train data=Banana_Disease_Dataset.yaml model=yolo11-SA-Custom.yaml    batch=8 imgsz=1024 patience=0  name=test-SA project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


yolo train data=Banana_Disease_Dataset.yaml model=yolo11-SA.yaml    batch=8 imgsz=1024 patience=0  name=YOLOv11-SA-Origin-400 project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True
yolo train data=Banana_Disease_Dataset.yaml model=yolo11.yaml    batch=8 imgsz=1024 patience=100  name=YOLOv11-Base-400 project=YOLOv11-All-Scheme-Flinta epochs=400 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# # Semi-supervised learning

# # YOLOv11 SA-Origin conf=0.25

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter1/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter4/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter4/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# YOLOv11 SA Customm conf=0.25

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# # YOLOv11 Base conf=0.25
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-Base-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter1/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-Base-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter2/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-Base-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter3/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-Base-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True










# # YOLOv11 Base conf=0.01


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter1/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter2/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter3/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-Base-400-Conf001-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True












# # # YOLOv11 SA-Origin conf=0.01


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter1/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter2/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# # rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# # yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# # yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# # rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# # yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter4/weights/best.pt iou=0.1 agnostic_nms=True

# # yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter4/weights/best.pt   batch=4 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True










# # YOLOv11 SA Customm conf=0.01

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter1/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter2/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True








# # YOLOv11 SA-Origin conf=0.25

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-FL-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter1/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-FL-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter2/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-FL-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter3/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-FL-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter4/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FL-Iter4/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-SA-Origin-400-FL-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True







# YOLOv11 Base conf=0.25
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter1/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter2/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter3/weights/best.pt   batch=2 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# YOLOv11 SA Customm conf=0.25

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter1-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1-test/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1-test/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter2-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2-test/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2-test/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter3-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3-test/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3-test/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter4-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter4-test/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter4-test/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter5-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter5-test/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter5-test/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-SA-Custom-400-Iter6-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True






# # YOLOv11 Base

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=3 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter1/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter1/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter2/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter2/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter3/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter3/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter4/weights/best.pt iou=0.1 agnostic_nms=True

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter4/weights/best.pt   batch=8 imgsz=1024 patience=10  name=YOLOv11-Base-400-FL-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FL-Iter4/weights/best.pt iou=0.1 agnostic_nms=True 



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemi-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemi-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemi-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemi-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter4/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemi-Iter4/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemi-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True













# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemi-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemi-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemi-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemi-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemi-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True







# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemi-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemi-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemi-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemi-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemi-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemi-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemi-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True











# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter1/weights/best.pt iou=0.1 agnostic_nms=True
# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter2/weights/best.pt iou=0.1 agnostic_nms=True
# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter3/weights/best.pt iou=0.1 agnostic_nms=True
# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



















# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True






# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter4/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Iter4/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Iter5 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True














# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True













# #YOLOv11 SA Origin FLSemiCustom Conf001
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-FLSemiCustom-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True








# #YOLOv11 SA Custom FLSemiCustom Conf001
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-FLSemiCustom-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# #YOLOv11 Base FLSemiCustom Conf001


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.01  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.01  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-FLSemiCustom-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-FLSemiCustom-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.01  agnostic_nms=True


























# # ###############################33

# #YOLOv11-SA 
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# # rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# # yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# # yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# #YOLOv11-SA-Custom
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# #YOLOv11-Base
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter3/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Iter3/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-Iter4 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



#YOLOv11 SA Origin Conf001

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True


# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



#YOLOv11 SA Custom Conf001

# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True



# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




#YOLOv11 Base Conf001
# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-Conf001-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True





# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter1/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-Conf001-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400-Conf001-Iter2/weights/best.pt   batch=4 imgsz=1024  patience=10  name=YOLOv11-Base-400-Conf001-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter1 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter1/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter2 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




# rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

# yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.25 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt iou=0.1 agnostic_nms=True 

# yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400-Iter2/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Iter3 project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt iou=0.1 agnostic_nms=True 

yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Custom-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Custom-400-Conf001-Iter1-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt iou=0.1 agnostic_nms=True 

yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-SA-Origin-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-SA-Origin-400-Conf001-Iter1-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True




rm -r ../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/labels

yolo predict source=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025/images conf=0.01 imgsz=1024 save_txt=True save=False project=../YOLO_Leaf_Disease/banana_dataset/Banana_Dataset_2024_2025/Unlabeled_Images_2025 name=./ exist_ok=True model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt iou=0.1 agnostic_nms=True 

yolo train data=Banana_Disease_Dataset_Semi.yaml model=YOLOv11-All-Scheme-Flinta/YOLOv11-Base-400/weights/best.pt   batch=8 imgsz=1024  patience=10  name=YOLOv11-Base-400-Conf001-Iter1-test project=YOLOv11-All-Scheme-Flinta epochs=50 close_mosaic=0 exist_ok=True iou=0.1 conf=0.25  agnostic_nms=True
