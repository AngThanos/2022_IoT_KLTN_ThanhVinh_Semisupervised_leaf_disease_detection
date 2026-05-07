import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a custom .pth checkpoint (teacher/model state) to Ultralytics .pt"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to checkpoint .pth (for example: ckpt_best.pth).",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        required=True,
        help="Base Ultralytics .pt model to initialize architecture.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output .pt path. Default: same folder as weights with suffix _teacher.pt/_student.pt",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="teacher_state",
        choices=["teacher_state", "model_state"],
        help="Checkpoint key to export.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    yolo = YOLO(args.pretrain)

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    if args.state_key not in checkpoint:
        keys = ", ".join(checkpoint.keys()) if isinstance(checkpoint, dict) else "<non-dict>"
        raise KeyError(f"Cannot find '{args.state_key}' in checkpoint. Available keys: {keys}")

    src_state = checkpoint[args.state_key]
    cleaned = {}
    for k, v in src_state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = yolo.model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")

    if args.out:
        out_path = Path(args.out)
    else:
        in_path = Path(args.weights)
        suffix = "teacher" if args.state_key == "teacher_state" else "student"
        out_path = in_path.with_name(f"{in_path.stem}_{suffix}.pt")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
