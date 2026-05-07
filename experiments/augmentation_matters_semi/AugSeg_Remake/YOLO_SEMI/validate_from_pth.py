import argparse
import atexit

import torch
import yaml

from augseg.dataset.yolodata import build_yololoader
from augseg.utils.dist_helper import setup_distributed
from augseg.utils.utils import set_random_seed
from train_semi import setup_yolo_model, validate_yolo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a .pth checkpoint with the custom train_semi validation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config_semi.yaml used for training.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to checkpoint .pth file (for example: ckpt_best.pth).",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="teacher_state",
        choices=["teacher_state", "model_state"],
        help="Checkpoint key to load before validation.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index shown in logs and used by sampler.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--prefix", type=str, default="PTH")
    return parser.parse_args()


def load_state_dict_from_pth(model, weights_path, state_key):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if state_key not in checkpoint:
        keys = ", ".join(checkpoint.keys()) if isinstance(checkpoint, dict) else "<non-dict>"
        raise KeyError(f"Cannot find '{state_key}' in checkpoint. Available keys: {keys}")

    state_dict = checkpoint[state_key]
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, world_size = setup_distributed(port=args.port)

    def _cleanup_dist():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    atexit.register(_cleanup_dist)

    # validate_yolo in this repo does not all-gather metrics, so keep single process for correct results.
    if world_size != 1:
        if rank == 0:
            raise RuntimeError(
                "Please run with single process: torchrun --nproc_per_node=1 ... validate_from_pth.py"
            )
        return

    model = setup_yolo_model(cfg["net"]["encoder"]["pretrain"])
    model.cuda().eval()

    load_state_dict_from_pth(model, args.weights, args.state_key)

    val_loader = build_yololoader("val", cfg, seed=args.seed)
    metrics = validate_yolo(
        model=model,
        data_loader=val_loader,
        epoch=args.epoch,
        logger=None,
        cfg=cfg,
        prefix=args.prefix,
    )

    if rank == 0:
        print("Validation done with custom pipeline")
        print(
            f"Precision={metrics['Precision']:.6f} "
            f"Recall={metrics['Recall']:.6f} "
            f"mAP50={metrics['mAP50']:.6f} "
            f"mAP50-95={metrics['mAP50-95']:.6f}"
        )


if __name__ == "__main__":
    main()
