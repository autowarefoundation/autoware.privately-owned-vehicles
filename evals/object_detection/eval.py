#!/usr/bin/env python3
# evals/object_detection/eval_object_detection.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

from evals.utils.helpers import (
    ensure,
    resolve_checkpoint,
    build_cfg_and_trainer_segmentation,
)

from evals.utils.trt_engine import TensorRTWrapper


# ============================================================
# Paths
# ============================================================
EVAL_ROOT = Path("runs/evals/object_detection")


# ============================================================
# Visualization utils
# ============================================================

COLOR_MAP = {
    0: (255, 255, 255),
    1: (0, 0, 255),     # red
    2: (0, 255, 255),   # yellow
    3: (255, 255, 0),   # cyan
}


def draw_predictions(img_bgr, predictions):
    """
    predictions: list of [x1,y1,x2,y2,conf,class]
    """
    for x1, y1, x2, y2, conf, cls in predictions:
        cls = int(cls)
        color = COLOR_MAP.get(cls, (255, 255, 255))

        cv2.rectangle(
            img_bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )
    return img_bgr


# ============================================================
# Post-processing (copiato / adattato dal tuo codice)
# ============================================================

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(preds, iou_thres=0.45):
    from torchvision.ops import nms as tv_nms
    if preds.numel() == 0:
        return torch.empty(0, 6, device=preds.device)
    boxes = preds[:, :4]
    scores = preds[:, 4]
    keep = tv_nms(boxes, scores, iou_thres)
    return preds[keep]


def post_process_predictions(
    raw_predictions,
    conf_thres=0.6,
    iou_thres=0.45,
):
    """
    raw_predictions: [1, C, N] or [1, N, C] (YOLO-like)
    returns tensor [K,6]
    """

    preds = raw_predictions.squeeze(0).permute(1, 0)
    boxes = preds[:, :4]
    class_probs = preds[:, 4:]

    scores, class_ids = torch.max(class_probs.sigmoid(), dim=1)
    mask = scores > conf_thres

    if mask.sum() == 0:
        return torch.empty(0, 6, device=preds.device)

    boxes_xyxy = xywh2xyxy(boxes[mask])

    combined = torch.cat([
        boxes_xyxy,
        scores[mask].unsqueeze(1),
        class_ids[mask].float().unsqueeze(1)
    ], dim=1)

    return nms(combined, iou_thres)


# ============================================================
# Evaluation loop
# ============================================================
@torch.no_grad()
def run_object_detection_eval(
    model,
    dataloader,
    device,
    out_dir,
    viz=10,
):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_idx = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)

        outputs = model(images)

        for b in range(images.shape[0]):
            if img_idx >= viz:
                return

            preds = post_process_predictions(outputs[b:b+1])

            # =====================================================
            # De-normalize ImageNet tensor -> uint8 BGR
            # =====================================================
            img = images[b].detach().cpu()
            img = img.permute(1, 2, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            img = (img * std + mean)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            preds_cpu = preds.cpu().numpy().tolist()
            vis = draw_predictions(img_bgr, preds_cpu)

            cv2.imwrite(
                str(out_dir / f"img_{img_idx:06d}.png"),
                vis
            )

            img_idx += 1



# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser("Object detection evaluation (PyTorch / TensorRT)")

    ap.add_argument("--run", default="runs/training/segmentation/dlv3plus_efficientnetb1_v2_finetuning_idda_5/")
    ap.add_argument("--engine", help="TensorRT engine (.engine)")
    ap.add_argument("--checkpoint")
    ap.add_argument("--best", action="store_true")

    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--viz", type=int, default=10)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()
    device = torch.device(args.device)

    run_dir = Path(args.run)
    engine_name = Path(args.engine).name if args.engine else ""
    engine_name = engine_name.replace(engine_name.split(".")[-1], "")

    #remove the trailing dot
    engine_name = engine_name[:-1]
    if engine_name:
        print(f"[INFO] Using TensorRT engine: {engine_name}")

    ensure(run_dir, "run folder")

    suffix = "trt" if args.engine else ""
    out_dir = EVAL_ROOT / f"{engine_name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {out_dir}")

    # ============================================================
    # Build config + dataloaders
    # ============================================================
    ckpt_path = args.checkpoint or resolve_checkpoint(run_dir, args.best)

    cfg, trainer = build_cfg_and_trainer_segmentation(
        run_dir=run_dir,
        datasets=args.datasets,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        checkpoint_path=ckpt_path,
    )

    val_loaders = trainer.val_loaders

    # ============================================================
    # Build backend
    # ============================================================
    if args.engine:
        model = TensorRTWrapper(args.engine, device=device)
    else:
        model = trainer.model.to(device).eval()

    # ============================================================
    # Run evaluation
    # ============================================================
    for ds_name, loader in val_loaders.items():
        print(f"\n=== Object detection on dataset: {ds_name} ===")

        ds_out = out_dir / ds_name
        run_object_detection_eval(
            model=model,
            dataloader=loader,
            device=device,
            out_dir=ds_out,
            viz=args.viz,
        )

    print(f"\n✔ Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
