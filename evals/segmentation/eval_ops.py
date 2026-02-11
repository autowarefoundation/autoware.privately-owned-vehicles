#!/usr/bin/env python3
# evals/segmentation/eval_segmentation.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2


from utils.segmentation import validate_segmentation

from evals.utils.helpers import (
    ensure,
    resolve_checkpoint,
    build_cfg_and_trainer_segmentation,
)

from training.segmentation.deeplabv3plus_trainer import DeepLabV3PlusTrainer
from evals.utils.trt_engine import TensorRTWrapper


# ============================================================
# Paths
# ============================================================
EVAL_ROOT = Path("runs/evals/segmentation")



# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    ap = argparse.ArgumentParser("Segmentation evaluation (PyTorch / TensorRT)")

    ap.add_argument("--run", required=True,
                    help="Training run folder (config + dataloader)")

    ap.add_argument("--engine",
                    help="TensorRT engine (.engine). If set, overrides checkpoint")

    ap.add_argument("--checkpoint",
                    help="Optional checkpoint override (ignored if --engine is set)")

    ap.add_argument("--best", action="store_true")

    # Data
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)

    # Runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz", type=int, default=10)

    args = ap.parse_args()
    device = torch.device(args.device)

    # ============================================================
    # Setup
    # ============================================================

    run_dir = Path(args.run)
    ensure(run_dir, "run folder")

    suffix = "trt" if args.engine else ""
    out_dir = EVAL_ROOT / f"{run_dir.name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {out_dir}")

    # ============================================================
    # Build config + dataloaders (ALWAYS)
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
    loss_fn     = trainer.loss_fn
    loss_cfg    = cfg.get("loss", {})
    model = trainer.model
    # ============================================================
    # Build model backend
    # ============================================================

    # resolve checkpoint
    if args.engine:
        #override model with TensorRT engine
        model = TensorRTWrapper(args.engine, device=device)


    # ============================================================
    # Evaluation loop (shared)
    # ============================================================

    from torchinfo import summary

    model = model.to(device)
    model.eval()

    # dummy input for torchinfo summary
    h = args.height if args.height else 320
    w = args.width  if args.width  else 640

    print("\n==============================")
    print("TORCHINFO SUMMARY")
    print("==============================")

    summary(
        model,
        input_size=(1, 3, h, w),
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        device=device.type,
    )


if __name__ == "__main__":
    main()
