#!/bin/bash

# Activate the virtual environment
conda activate Cinh_env

# ============ Change loss function to VFL and run training ============
# /home/jupyter-iec2021iot13/Vinh/ChangeLoss/install_loss.sh vfl
# echo "training VFL ..." > status_auto_train.txt

# echo "training yolov11-base ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl/yolov11-base/config_semi.yaml

# echo "training yolov11-sa ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl/yolov11-sa/config_semi.yaml

# echo "training yolov11-sa-custom ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl/yolov11-sa-custom/config_semi.yaml



# ============ Change loss function to VFL Custom and run training ============
# /home/jupyter-iec2021iot13/Vinh/ChangeLoss/install_loss.sh vfl_custom
# echo "training VFL Custom ..." > status_auto_train.txt

# echo "training yolov11-base ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl_custom/yolov11-base/config_semi.yaml

# echo "training yolov11-sa ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl_custom/yolov11-sa/config_semi.yaml

# echo "training yolov11-sa-custom ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl_custom/yolov11-sa-custom/config_semi.yaml

./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/test/yolov11-base/vfl_custom/config_semi.yaml
./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_25/vfl_custom/yolov11-base/config_semi.yaml
./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/vfl_custom/yolov11-sa/config_semi.yaml

# ============ Change loss function to BCE and run training ============
# /home/jupyter-iec2021iot13/Vinh/ChangeLoss/install_loss.sh bce
# echo "training BCE ..." > status_auto_train.txt

# echo "training yolov11-base ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/bce/yolov11-base/config_semi.yaml

# echo "training yolov11-sa ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/bce/yolov11-sa/config_semi.yaml

# echo "training yolov11-sa-custom ..." >> status_auto_train.txt
# ./run.sh /home/jupyter-iec2021iot13/Vinh/AugSeg_Remake/exps/release_1/bce/yolov11-sa-custom/config_semi.yaml
