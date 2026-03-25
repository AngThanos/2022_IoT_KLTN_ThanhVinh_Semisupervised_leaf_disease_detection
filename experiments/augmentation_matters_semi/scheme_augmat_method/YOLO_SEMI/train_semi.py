import argparse
import atexit
import yaml
import os
import os.path as osp
import pprint
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from augseg.dataset.builder import get_loader
from augseg.utils.dist_helper import setup_distributed
from augseg.utils.lr_helper import get_optimizer, get_scheduler
from augseg.utils.utils import AverageMeter, load_state, set_random_seed, setup_default_logging

import warnings
warnings.filterwarnings('ignore')


def sigmoid_rampup(current, rampup_length):
    """Exponential ramp-up from Mean Teacher to stabilize early unsupervised training."""
    # Ramp-up giúp nhánh unsupervised tăng dần ảnh hưởng,
    # tránh làm model nhiễu ở giai đoạn đầu khi pseudo-label còn kém chất lượng.
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(np.exp(-5.0 * phase * phase))


def apply_nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Apply Non-Maximum Suppression to YOLO predictions.
    
    Args:
        prediction: Raw model output tensor, typically [batch, 4+nc, num_boxes]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of detections per image, each [N, 6] (x1, y1, x2, y2, conf, cls)
    """
    try:
        # Ultralytics >= 8.3 moved NMS to ultralytics.utils.nms.
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        # Backward compatibility for older Ultralytics releases.
        from ultralytics.utils.ops import non_max_suppression

    # Ultralytics models can return tuple/list during inference.
    # Chuẩn hóa output để các bước sau luôn xử lý 1 tensor dự đoán chính.
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    output = non_max_suppression(
        prediction,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    # Keep a stable [N, 6] tensor format for downstream code.
    # Mỗi detection có dạng: [x1, y1, x2, y2, conf, cls].
    normalized = []
    for det in output:
        if det is None or det.shape[0] == 0:
            normalized.append(torch.zeros((0, 6), dtype=torch.float32, device=prediction.device))
        else:
            normalized.append(det[:, :6])

    return normalized


def compute_ap(recall, precision):
    """Compute Average Precision using all-point interpolation (VOC 2010+)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # AP = sum of rectangular areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute Average Precision for each class.

    Args:
        tp: np.ndarray [N, T] - True positive flags at T IoU thresholds
        conf: np.ndarray [N] - Confidence scores
        pred_cls: np.ndarray [N] - Predicted class indices
        target_cls: np.ndarray [M] - Ground truth class indices

    Returns:
        p: np.ndarray [C] - Precision per class (at max-F1 point, IoU=0.5)
        r: np.ndarray [C] - Recall per class (at max-F1 point, IoU=0.5)
        ap: np.ndarray [C, T] - AP per class at each IoU threshold
        unique_classes: np.ndarray [C] - Unique class indices
    """
    # Sort by confidence descending
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls).astype(int)
    nc = len(unique_classes)
    n_thres = tp.shape[1] if tp.ndim == 2 else 1

    ap = np.zeros((nc, n_thres))
    p_class = np.zeros(nc)
    r_class = np.zeros(nc)

    for ci, c in enumerate(unique_classes):
        pred_mask = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_pred = pred_mask.sum()

        if n_pred == 0 or n_gt == 0:
            continue

        tp_c = tp[pred_mask]

        for ti in range(n_thres):
            tpc = tp_c[:, ti].astype(float)
            fpc = 1.0 - tpc

            tp_cum = np.cumsum(tpc)
            fp_cum = np.cumsum(fpc)

            recall = tp_cum / (n_gt + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)

            ap[ci, ti] = compute_ap(recall, precision)

            # At IoU=0.5 (first threshold), store P/R at max-F1 operating point
            if ti == 0:
                f1 = 2 * precision * recall / (precision + recall + 1e-16)
                if len(f1) > 0:
                    idx = np.argmax(f1)
                    p_class[ci] = precision[idx]
                    r_class[ci] = recall[idx]

    return p_class, r_class, ap, unique_classes


def setup_yolo_model(pretrain_path):
    """Initialize YOLO model with proper hyperparameters."""
    # Lấy kiến trúc + trọng số nền từ Ultralytics.
    yolo_wrapper = YOLO(pretrain_path)
    model = yolo_wrapper.model
    
    # Set args as SimpleNamespace BEFORE init_criterion
    # Các trọng số loss cơ bản (box/cls/dfl) cho criterion của YOLO.
    default_args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.args = default_args
    
    # Initialize loss criterion
    if hasattr(model, 'init_criterion'):
        model.init_criterion()
    
    return model


def build_pseudo_batch(
    teacher_preds,
    images,
    conf_thres=0.7,
    iou_thres=0.5,
    max_det=300,
    class_conf_thres=None,
    score_weighting=False,
    score_power=1.0,
    topk_per_image=0,
):
    """Build YOLO-format pseudo labels from teacher detections."""
    # Teacher dự đoán trên ảnh weak -> NMS -> lọc threshold -> convert về YOLO xywh normalized.
    detections = apply_nms(
        teacher_preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
    )

    batch_idx, cls_list, box_list = [], [], []
    score_list = []
    _, _, img_h, img_w = images.shape
    eps = 1e-6
    valid_images = 0

    for bi, det in enumerate(detections):
        if det is None or det.shape[0] == 0:
            continue

        if class_conf_thres:
            # Cho phép threshold theo từng lớp để giảm bias lớp khó/dễ.
            cls_ids = det[:, 5].long()
            thrs = torch.full_like(det[:, 4], float(conf_thres))
            for cls_id, cls_thr in class_conf_thres.items():
                thrs[cls_ids == int(cls_id)] = float(cls_thr)
            det = det[det[:, 4] >= thrs]
        else:
            det = det[det[:, 4] >= float(conf_thres)]

        if det.shape[0] == 0:
            continue

        # Keep only top-k pseudo boxes per image to reduce noisy supervision density.
        # Giới hạn mật độ pseudo để tránh unsupervised loss bị chi phối bởi ảnh quá nhiều box.
        if int(topk_per_image) > 0 and det.shape[0] > int(topk_per_image):
            topk_idx = torch.topk(det[:, 4], k=int(topk_per_image), largest=True).indices
            det = det[topk_idx]

        x1 = det[:, 0].clamp(0, img_w)
        y1 = det[:, 1].clamp(0, img_h)
        x2 = det[:, 2].clamp(0, img_w)
        y2 = det[:, 3].clamp(0, img_h)
        w = (x2 - x1).clamp(min=eps)
        h = (y2 - y1).clamp(min=eps)
        cx = x1 + w / 2
        cy = y1 + h / 2

        # Normalize to YOLO xywh format expected by loss()
        # loss của YOLO nhận box chuẩn hóa theo kích thước ảnh hiện tại.
        boxes_xywh = torch.stack([
            cx / float(img_w),
            cy / float(img_h),
            w / float(img_w),
            h / float(img_h),
        ], dim=1).clamp(0, 1)

        n = boxes_xywh.shape[0]
        if n == 0:
            continue

        valid_images += 1
        batch_idx.append(torch.full((n,), bi, dtype=torch.long, device=images.device))
        cls_list.append(det[:, 5:6].to(images.device).float())
        box_list.append(boxes_xywh.to(images.device).float())
        score_list.append(det[:, 4].to(images.device).float())

    if len(batch_idx) == 0:
        empty = {
            'img': images,
            'batch_idx': torch.zeros((0,), dtype=torch.long, device=images.device),
            'cls': torch.zeros((0, 1), dtype=torch.float32, device=images.device),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32, device=images.device),
        }
        return empty, 0.0, 0.0, 0.0

    pseudo_batch = {
        'img': images,
        'batch_idx': torch.cat(batch_idx, dim=0),
        'cls': torch.cat(cls_list, dim=0),
        'bboxes': torch.cat(box_list, dim=0),
    }
    pseudo_ratio = valid_images / float(images.shape[0])
    # Số pseudo box trung bình mỗi ảnh, dùng để theo dõi độ "dày" của pseudo supervision.
    pseudo_boxes_per_img = float(pseudo_batch['batch_idx'].numel()) / float(images.shape[0])
    if score_weighting:
        all_scores = torch.cat(score_list, dim=0)
        pseudo_weight = float((all_scores.clamp(0, 1) ** float(score_power)).mean().item())
    else:
        pseudo_weight = 1.0
    return pseudo_batch, pseudo_ratio, pseudo_weight, pseudo_boxes_per_img


def _prediction_tensor_list(prediction):
    """Normalize YOLO raw prediction structure into a list of tensors."""
    # Một số phiên bản/head YOLO trả tuple/list nhiều mức đặc trưng.
    # Hàm này gom về list tensor để tính consistency thống nhất.
    if isinstance(prediction, torch.Tensor):
        return [prediction]
    if isinstance(prediction, (list, tuple)):
        tensors = [x for x in prediction if isinstance(x, torch.Tensor)]
        if len(tensors) == 0 and len(prediction) > 0 and isinstance(prediction[0], (list, tuple)):
            tensors = [x for x in prediction[0] if isinstance(x, torch.Tensor)]
        return tensors
    return []


def compute_consistency_loss(student_preds, teacher_preds, mode="mse", temperature=1.0):
    """
    Soft consistency between student(strong) and teacher(weak) raw predictions.
    This assumes geometry-preserving strong augmentations (photometric transforms).
    """
    # Lưu ý: consistency hiện tính trên raw tensor dự đoán.
    # Cách này thực dụng nhưng chưa semantic-aware bằng decode+matching theo box.
    stu_list = _prediction_tensor_list(student_preds)
    tea_list = _prediction_tensor_list(teacher_preds)

    if len(stu_list) == 0 or len(tea_list) == 0:
        device = None
        if len(stu_list) > 0:
            device = stu_list[0].device
        elif len(tea_list) > 0:
            device = tea_list[0].device
        return torch.tensor(0.0, device=device if device is not None else "cpu")

    total = 0.0
    count = 0
    for s, t in zip(stu_list, tea_list):
        if s.shape != t.shape:
            continue

        if mode == "kl":
            # KL với temperature để làm mềm phân phối teacher.
            s_log_prob = torch.log_softmax(s / temperature, dim=1)
            t_prob = torch.softmax(t.detach() / temperature, dim=1)
            loss = torch.nn.functional.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature ** 2)
        else:
            # MSE trên sigmoid output là lựa chọn ổn định mặc định.
            loss = torch.nn.functional.mse_loss(torch.sigmoid(s), torch.sigmoid(t.detach()))

        total = total + loss
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=stu_list[0].device)
    return total / float(count)


def main(args):
    if args.seed is not None:
        # Cố định seed để dễ tái lập kết quả.
        set_random_seed(args.seed, deterministic=True)
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, world_size = setup_distributed(port=args.port)

    def _cleanup_dist():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    atexit.register(_cleanup_dist)

    # 1. Output settings
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]
    
    if rank == 0:
        os.makedirs(cfg["log_path"], exist_ok=True)
        os.makedirs(cfg["save_path"], exist_ok=True)
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
        csv_path = os.path.join(cfg["log_path"], f"seg_{curr_timestr}_stat.csv")
        logger.info("{}".format(pprint.pformat(cfg)))
        tb_logger = SummaryWriter(osp.join(cfg["log_path"], "events_seg", curr_timestr)) if flag_use_tb else None
    else:
        logger, csv_path, tb_logger = None, None, None
    
    dist.barrier()

    # 2. Prepare student model
    model = setup_yolo_model(cfg["net"]["encoder"]["pretrain"])
    
    for param in model.parameters():
        param.requires_grad = True
    
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # 3. Data loaders (Step Augmentation handled inside dataset)
    # train_loader_sup: dữ liệu có nhãn
    # train_loader_unsup: dữ liệu không nhãn (weak/strong)
    # val_loader: dữ liệu validation
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

    # 4. Optimizer
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    optimizer = get_optimizer(model.parameters(), cfg_optim)

    # 5. DDP wrapper
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # 6. Teacher model (EMA)
    # Teacher là bản làm mượt tham số từ student bằng EMA.
    model_teacher = setup_yolo_model(cfg["net"]["encoder"]["pretrain"])
    model_teacher.cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # Initialize teacher with student weights
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    # 7. Resume checkpoint
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1
    
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if os.path.exists(lastest_model):
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(lastest_model, model, optimizer=optimizer, key="model_state")
            load_state(lastest_model, model_teacher, optimizer=optimizer, key="teacher_state")

    lr_scheduler = get_scheduler(cfg_trainer, len(train_loader_sup), optimizer, start_epoch=last_epoch)
    print(f"====================== {len(train_loader_sup)} ==============")
    # 8. Training loop
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        res_loss_sup, res_loss_unsup = train(
            model, model_teacher, optimizer, lr_scheduler,
            train_loader_sup, train_loader_unsup,
            epoch, tb_logger, logger, cfg
        )

        # Validation
        if cfg_trainer.get("evaluate_student", True):
            metrics_stu = validate_yolo(model, val_loader, epoch, logger, cfg, prefix="STU")
        else:
            metrics_stu = {'Precision': 0.0, 'Recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        metrics_tea = validate_yolo(model_teacher, val_loader, epoch, logger, cfg, prefix="EMA")
        prec_stu = metrics_stu['mAP50']
        prec_tea = metrics_tea['mAP50']
        prec = prec_tea

        # Save checkpoint
        if rank == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
            }
            
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            
            # Save statistics
            tmp_results = {
                'loss_lb': res_loss_sup,
                'loss_ub': res_loss_unsup,
                'Precision_stu': metrics_stu['Precision'],
                'Recall_stu': metrics_stu['Recall'],
                'mAP50_stu': metrics_stu['mAP50'],
                'mAP50-95_stu': metrics_stu['mAP50-95'],
                'Precision_tea': metrics_tea['Precision'],
                'Recall_tea': metrics_tea['Recall'],
                'mAP50_tea': metrics_tea['mAP50'],
                'mAP50-95_tea': metrics_tea['mAP50-95'],
                'best_mAP50': best_prec,
                'best_mAP50_stu': best_prec_stu,
            }
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch + 1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')
            
            logger.info(
                f" <<Test>> - Epoch: {epoch}  "
                f"STU[Precision:{metrics_stu['Precision']:.4f} Recall:{metrics_stu['Recall']:.4f} mAP50:{metrics_stu['mAP50']:.4f} mAP50-95:{metrics_stu['mAP50-95']:.4f}]  "
                f"EMA[Precision:{metrics_tea['Precision']:.4f} Recall:{metrics_tea['Recall']:.4f} mAP50:{metrics_tea['mAP50']:.4f} mAP50-95:{metrics_tea['mAP50-95']:.4f}]  "
                f"\033[34mBest-STU mAP50:{best_prec_stu:.4f}/{best_epoch_stu}  "
                f"\033[31mBest-EMA mAP50:{best_prec:.4f}/{best_epoch}\033[0m"
            )
            
            if tb_logger is not None:
                tb_logger.add_scalar("mAP50_tea", metrics_tea['mAP50'], epoch)
                tb_logger.add_scalar("mAP50-95_tea", metrics_tea['mAP50-95'], epoch)
                tb_logger.add_scalar("mAP50_stu", metrics_stu['mAP50'], epoch)
                tb_logger.add_scalar("mAP50-95_stu", metrics_stu['mAP50-95'], epoch)

    if rank == 0 and tb_logger is not None:
        tb_logger.close()
    _cleanup_dist()


def train(model, model_teacher, optimizer, lr_scheduler, loader_l, loader_u, epoch, tb_logger, logger, cfg):
    """Training loop for YOLO semi-supervised detection with pseudo-labels."""

    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    
    # Semi-supervised settings
    unsup_cfg = cfg["trainer"].get("unsupervised", {})
    loss_weight = unsup_cfg.get("loss_weight", 1.0)
    consistency_weight = unsup_cfg.get("consistency_weight", 1.0)
    consistency_mode = unsup_cfg.get("consistency_mode", "mse")
    consistency_temp = unsup_cfg.get("consistency_temperature", 1.0)
    rampup_epochs = unsup_cfg.get("rampup_epochs", 10)
    pseudo_conf_thres = unsup_cfg.get("threshold", 0.7)
    pseudo_iou_thres = unsup_cfg.get("nms_iou", 0.5)
    pseudo_max_det = unsup_cfg.get("max_det", 300)
    class_conf_thres = unsup_cfg.get("class_thresholds", None)
    pseudo_score_weighting = unsup_cfg.get("score_weighting", False)
    pseudo_score_power = unsup_cfg.get("score_power", 1.0)
    pseudo_topk_per_image = unsup_cfg.get("topk_per_image", 0)
    sup_only_epoch = cfg["trainer"].get("sup_only_epoch", 0)

    # Công thức tổng quát trong vòng lặp:
    # loss = L_sup + lambda_u * (L_pseudo + w_cons * L_consistency)

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    
    assert len(loader_l) == len(loader_u), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, mismatch!"

    # Metrics
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    pseudo_losses = AverageMeter(20)
    cons_losses = AverageMeter(20)
    unsup_weights = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    pseudo_ratios = AverageMeter(20)
    pseudo_boxes_per_img_meter = AverageMeter(20)
    
    # Print frequency
    print_freq = max(len(loader_u) // 8, 1)
    print_freq_lst = [i * print_freq for i in range(1, 8)] + [len(loader_u) - 1]

    model.train()
    model_teacher.eval()

    for step in range(len(loader_l)):
        batch_start = time.time()
        i_iter = epoch * len(loader_l) + step

        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        # Load data
        _, image_l, label_l = next(loader_l_iter)
        image_l, label_l = image_l.cuda(), label_l.cuda()
        _, image_u_weak, image_u_aug, _ = next(loader_u_iter)
        image_u_weak, image_u_aug = image_u_weak.cuda(), image_u_aug.cuda()

        # Get underlying YOLO model
        yolo_model = model.module if hasattr(model, 'module') else model
        yolo_model.train()

        # ============ Supervised Loss ============
        # Nhánh supervised chuẩn trên dữ liệu có nhãn thật.
        pred_l = yolo_model(image_l)
        
        batch_l = {
            'img': image_l,
            'batch_idx': label_l[:, 0].long() if len(label_l) > 0 else torch.tensor([], device=image_l.device, dtype=torch.long),
            'cls': label_l[:, 1:2] if len(label_l) > 0 else torch.zeros((0, 1), device=image_l.device),
            'bboxes': label_l[:, 2:] if len(label_l) > 0 else torch.zeros((0, 4), device=image_l.device),
        }
        
        loss_items, _ = yolo_model.loss(batch_l, pred_l)
        sup_loss = loss_items.sum()

        # ============ Unsupervised Loss with Teacher Pseudo Labels ============
        unsup_loss = torch.tensor(0.0, device=image_l.device)
        pseudo_loss = torch.tensor(0.0, device=image_l.device)
        consistency_loss = torch.tensor(0.0, device=image_l.device)
        pseudo_ratio = 0.0
        pseudo_boxes_per_img = 0.0
        unsup_weight = 0.0
        
        if epoch >= sup_only_epoch:
            # Ramp up unsupervised signal after warmup epochs.
            # lambda_u tăng dần theo epoch để tránh sốc gradient ở đầu train.
            unsup_weight = float(loss_weight) * sigmoid_rampup(
                epoch - sup_only_epoch,
                rampup_epochs,
            )

            with torch.no_grad():
                model_teacher.eval()
                # Teacher dự đoán trên weak view.
                teacher_preds = model_teacher(image_u_weak)

            yolo_model.train()
            # Student học trên strong view.
            student_preds_strong = yolo_model(image_u_aug)

            consistency_loss = compute_consistency_loss(
                student_preds=student_preds_strong,
                teacher_preds=teacher_preds,
                mode=consistency_mode,
                temperature=consistency_temp,
            )

            pseudo_batch, pseudo_ratio, pseudo_weight, pseudo_boxes_per_img = build_pseudo_batch(
                teacher_preds=teacher_preds,
                images=image_u_aug,
                conf_thres=pseudo_conf_thres,
                iou_thres=pseudo_iou_thres,
                max_det=pseudo_max_det,
                class_conf_thres=class_conf_thres,
                score_weighting=pseudo_score_weighting,
                score_power=pseudo_score_power,
                topk_per_image=pseudo_topk_per_image,
            )

            if pseudo_batch['batch_idx'].numel() > 0:
                try:
                    pseudo_loss_items, _ = yolo_model.loss(pseudo_batch, student_preds_strong)
                    pseudo_loss = pseudo_loss_items.sum() * float(pseudo_weight)
                except RuntimeError as e:
                    if logger is not None and rank == 0:
                        logger.warning(f"Unsupervised pseudo-label loss failed: {e}")
                    pseudo_loss = torch.tensor(0.0, device=image_l.device)

            unsup_raw = pseudo_loss + float(consistency_weight) * consistency_loss
            unsup_loss = unsup_raw * float(unsup_weight)
            if not torch.isfinite(unsup_loss):
                unsup_loss = torch.tensor(0.0, device=image_l.device)
                pseudo_loss = torch.tensor(0.0, device=image_l.device)
                consistency_loss = torch.tensor(0.0, device=image_l.device)
            
        pseudo_ratios.update(pseudo_ratio)
        pseudo_boxes_per_img_meter.update(pseudo_boxes_per_img)
        unsup_weights.update(unsup_weight)

        # ============ Total Loss ============
        # Loss cuối để backprop student.
        loss = sup_loss + unsup_loss

        # Update student model
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()

        # Update teacher model with EMA
        with torch.no_grad():
            if epoch >= sup_only_epoch:
                ema_decay = min(
                    1 - 1 / (i_iter - len(loader_l) * sup_only_epoch + 1),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                # EMA: theta_t = m * theta_t + (1 - m) * theta_s
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)

        # Gather losses from all GPUs
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        reduced_pseudo_loss = pseudo_loss.clone().detach()
        dist.all_reduce(reduced_pseudo_loss)
        pseudo_losses.update(reduced_pseudo_loss.item() / world_size)

        reduced_cons_loss = consistency_loss.clone().detach()
        dist.all_reduce(reduced_cons_loss)
        cons_losses.update(reduced_cons_loss.item() / world_size)

        # Logging
        batch_times.update(time.time() - batch_start)
        
        if step in print_freq_lst and rank == 0:
            logger.info(
                f"Epoch/Iter [{cfg['trainer']['epochs']}:{epoch:3}/{step:3}].  "
                f"Sup:{sup_losses.val:.3f}({sup_losses.avg:.3f})  "
                f"Uns:{uns_losses.val:.3f}({uns_losses.avg:.3f})  "
                f"Pseudo:{pseudo_losses.val:.3f}({pseudo_losses.avg:.3f})  "
                f"Cons:{cons_losses.val:.3f}({cons_losses.avg:.3f})  "
                f"Lambda:{unsup_weights.val:.3f}  "
                f"Pseudo:{pseudo_ratios.avg:.1f}  "
                f"PBox/Img:{pseudo_boxes_per_img_meter.avg:.2f}  "
                f"Time:{batch_times.avg:.2f}  "
                f"LR:{learning_rates.val:.5f}"
            )
            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("Pseudo Loss", pseudo_losses.avg, i_iter)
                tb_logger.add_scalar("Consistency Loss", cons_losses.avg, i_iter)
                tb_logger.add_scalar("Unsup Lambda", unsup_weights.avg, i_iter)
                tb_logger.add_scalar("Pseudo Ratio", pseudo_ratios.avg, i_iter)
                tb_logger.add_scalar("Pseudo Boxes per Img", pseudo_boxes_per_img_meter.avg, i_iter)
    
    return sup_losses.avg, uns_losses.avg


def validate_yolo(model, data_loader, epoch, logger, cfg, prefix=""):
    """
    Validation function for YOLO detection computing P, R, mAP50, mAP50-95.

    Returns:
        dict with keys: 'P', 'R', 'mAP50', 'mAP50-95'
    """
    try:
        from ultralytics.utils.metrics import box_iou
    except ImportError:
        def box_iou(box1, box2):
            area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
            area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
            inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
            inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
            inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
            inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
            inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            union_area = area1[:, None] + area2 - inter_area
            return inter_area / (union_area + 1e-10)

    yolo_model = model.module if hasattr(model, 'module') else model
    yolo_model.eval()

    data_loader.sampler.set_epoch(epoch)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # IoU thresholds for mAP computation
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # [0.50, 0.55, ..., 0.95]
    conf_thres = 0.001   # Low threshold for proper PR curve
    nms_iou_thres = 0.65
    zero_metrics = {'Precision': 0.0, 'Recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}

    stats = []  # list of (correct[N,10], conf[N], pred_cls[N], target_cls[M])

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        batch_size, _, img_h, img_w = images.shape

        with torch.no_grad():
            preds = yolo_model(images)
            nms_preds = apply_nms(preds, conf_thres=conf_thres, iou_thres=nms_iou_thres, max_det=300)

        for bi, pred in enumerate(nms_preds):
            # Get ground truth for this image
            if len(labels) > 0:
                target_boxes = labels[labels[:, 0] == bi]
                if len(target_boxes) > 0:
                    target_cls = target_boxes[:, 1].cpu()
                    tboxes_raw = target_boxes[:, 2:6].clone()
                    cx, cy, w, h = tboxes_raw[:, 0], tboxes_raw[:, 1], tboxes_raw[:, 2], tboxes_raw[:, 3]
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    tboxes = torch.stack([x1, y1, x2, y2], dim=1).cpu()
                else:
                    target_cls = torch.zeros(0)
                    tboxes = torch.zeros((0, 4))
            else:
                target_cls = torch.zeros(0)
                tboxes = torch.zeros((0, 4))

            nl = len(target_cls)

            if len(pred) == 0:
                if nl:
                    stats.append((
                        np.zeros((0, len(iou_thresholds)), dtype=bool),
                        np.zeros(0),
                        np.zeros(0),
                        target_cls.numpy()
                    ))
                continue

            pred_boxes = pred[:, :4].cpu()
            pred_conf = pred[:, 4].cpu()
            pred_cls_t = pred[:, 5].cpu()

            # Match predictions to GT at multiple IoU thresholds
            correct = np.zeros((len(pred), len(iou_thresholds)), dtype=bool)

            if nl:
                iou = box_iou(pred_boxes, tboxes)  # [N_pred, N_gt]
                correct_class = pred_cls_t[:, None] == target_cls[None, :]  # [N_pred, N_gt]

                for ti, thr in enumerate(iou_thresholds):
                    matches = torch.nonzero(
                        (iou >= thr) & correct_class, as_tuple=False
                    )  # [K, 2] -> (pred_idx, gt_idx)

                    if matches.shape[0]:
                        # Sort by IoU descending to prioritize best matches
                        match_ious = iou[matches[:, 0], matches[:, 1]]
                        sorted_idx = match_ious.argsort(descending=True)
                        matches = matches[sorted_idx]

                        # Each prediction matched at most once (keep highest IoU)
                        _, unique_pred_idx = np.unique(
                            matches[:, 0].cpu().numpy(), return_index=True
                        )
                        matches = matches[unique_pred_idx]

                        # Each ground truth matched at most once
                        _, unique_gt_idx = np.unique(
                            matches[:, 1].cpu().numpy(), return_index=True
                        )
                        matches = matches[unique_gt_idx]

                        correct[matches[:, 0].cpu().numpy(), ti] = True

            stats.append((
                correct,
                pred_conf.numpy(),
                pred_cls_t.numpy(),
                target_cls.numpy()
            ))

    # Gather validation stats across all ranks so AP/mAP are computed globally.
    # Nếu chạy DDP, cần gộp stats từ mọi GPU rồi mới tính metric để tránh lệch kết quả.
    if world_size > 1:
        gathered_stats = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_stats, stats)
        merged_stats = []
        for rank_stats in gathered_stats:
            if rank_stats:
                merged_stats.extend(rank_stats)
        stats = merged_stats

    # Return zeros if no stats
    if len(stats) == 0:
        if rank == 0 and logger:
            logger.info(f" [{prefix}Val] No predictions/targets for evaluation")
        return zero_metrics

    # Concatenate all stats
    tp_all = np.concatenate([s[0] for s in stats], axis=0)      # [total_pred, 10]
    conf_all = np.concatenate([s[1] for s in stats], axis=0)    # [total_pred]
    pred_cls_all = np.concatenate([s[2] for s in stats], axis=0)  # [total_pred]
    target_cls_all = np.concatenate([s[3] for s in stats], axis=0)  # [total_gt]

    if len(tp_all) == 0 or len(target_cls_all) == 0:
        if rank == 0 and logger:
            logger.info(f" [{prefix}Val] Empty stats")
        return zero_metrics

    # Compute per-class AP
    p, r, ap, unique_classes = ap_per_class(tp_all, conf_all, pred_cls_all, target_cls_all)

    # Mean across classes
    mp = p.mean() if len(p) else 0.0        # Mean Precision (at max-F1, IoU=0.5)
    mr = r.mean() if len(r) else 0.0        # Mean Recall    (at max-F1, IoU=0.5)
    map50 = ap[:, 0].mean() if len(ap) else 0.0      # mAP@0.5
    map50_95 = ap.mean() if len(ap) else 0.0          # mAP@0.5:0.95

    if rank == 0 and logger:
        logger.info(
            f" [{prefix}Val] Precision:{mp:.4f}  Recall:{mr:.4f}  mAP50:{map50:.4f}  mAP50-95:{map50_95:.4f}  "
            f"(classes:{len(unique_classes)}, preds:{len(tp_all)}, targets:{len(target_cls_all)})"
        )

    return {'Precision': float(mp), 'Recall': float(mr), 'mAP50': float(map50), 'mAP50-95': float(map50_95)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised YOLO Detection Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
