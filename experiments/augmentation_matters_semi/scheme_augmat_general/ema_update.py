import torch
import argparse
from ultralytics.nn.tasks import DetectionModel


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"]
    return ckpt, model, model.state_dict()


def save_ckpt(model, template, out):
    new = {
        "model": model,
        "ema": model,
        "epoch": template.get("epoch", -1),
        "updates": template.get("updates", 0),
        "train_args": template.get("train_args", {}),
    }
    torch.save(new, out)


def ema(student_sd, teacher_sd, decay):
    for k in student_sd:
        if k in teacher_sd:
            teacher_sd[k] = teacher_sd[k].float() * decay + \
                            student_sd[k].float() * (1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decay", type=float, default=0.999)
    args = ap.parse_args()

    stu_ckpt, stu_model, stu_sd = load_ckpt(args.student)

    if args.teacher == "None":
        save_ckpt(stu_model, stu_ckpt, args.out)
        return

    tea_ckpt, tea_model, tea_sd = load_ckpt(args.teacher)
    ema(stu_sd, tea_sd, args.decay)
    tea_model.load_state_dict(tea_sd)
    save_ckpt(tea_model, stu_ckpt, args.out)


if __name__ == "__main__":
    main()
