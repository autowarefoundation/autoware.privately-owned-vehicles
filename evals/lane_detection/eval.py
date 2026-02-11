#!/usr/bin/env python3
# evals/lane_detection/eval.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2

from utils.lanes import validate_lanes
from utils.training import load_yaml, set_global_seed

from evals.utils.helpers import (
    build_validator_egolanes,
    ensure,
    resolve_checkpoint,
    build_cfg_and_trainer_lanes
)

from training.lane_detection.deeplabv3plus_trainer import DeepLabV3PlusTrainer
from evals.utils.trt_engine import TensorRTWrapper


# ============================================================
# Paths
# ============================================================
EVAL_ROOT = Path("runs/evals/lanes")

# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    ap = argparse.ArgumentParser("Lane detection evaluation (PyTorch / TensorRT / EgoLanes)")

    ap.add_argument("--run", help="Training run folder")
    ap.add_argument("--model", default="deeplabv3plus",
                    help="deeplabv3plus | egolanes")

    ap.add_argument("--engine", help="TensorRT engine (.engine)")
    ap.add_argument("--checkpoint", help="Optional checkpoint override")
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
    # EgoLanes path (UNCHANGED, as requested)
    # ============================================================

    if args.model == "egolanes":
        print("[INFO] Using EgoLanes validator")
        trainer = build_validator_egolanes(args)
        model = trainer.model

        suffix = "egolanes"
        out_dir = EVAL_ROOT / suffix
        out_dir.mkdir(parents=True, exist_ok=True)

        val_loaders = trainer.val_loaders
        loss_fn     = trainer.loss_fn

    else:
        # ========================================================
        # DeeplabV3+ lanes
        # ========================================================

        if args.run is None:
            raise ValueError("--run is required for deeplabv3plus")

        run_dir = Path(args.run)
        ensure(run_dir, "run folder")

        suffix = "trt" if args.engine else ""
        out_dir = EVAL_ROOT / f"{run_dir.name}_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Output dir: {out_dir}")

        ckpt_path = args.checkpoint or resolve_checkpoint(run_dir, args.best)

        cfg, trainer = build_cfg_and_trainer_lanes(
            run_dir=run_dir,
            datasets=args.datasets,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            checkpoint_path=ckpt_path,
        )

        model = trainer.model
        model.eval()


        val_loaders = trainer.val_loaders
        loss_fn     = trainer.loss_fn


        # --------------------------------------------------------
        # Build model backend
        # --------------------------------------------------------

        if args.engine:
            print("[INFO] Using TensorRT engine for inference")
            model = TensorRTWrapper(args.engine, device=device)


    # ============================================================
    # Evaluation loop
    # ============================================================

    all_totals = []

    for ds_name, loader in val_loaders.items():

        print(f"\n=== Evaluating dataset: {ds_name} ===")

        results, vis_images = validate_lanes(
            model=model,
            dataloader=loader,
            loss_fn=loss_fn,
            device=device,
            dataset_name=ds_name,
            vis_count=args.viz,
        )

        print(f"✔ {ds_name}")
        print(f"  Mean IoU  : {results['mean_iou']:.4f}")
        print(f"  Pixel Acc: {results['pixel_acc']:.4f}")
        print(f"  Val Loss : {results['loss']:.4f}")

        all_totals.append(results)

        for i, vis in enumerate(vis_images):
            cv2.imwrite(
                str(out_dir / f"{ds_name}_sample_{i:03d}.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

    # ============================================================
    # Summary
    # ============================================================

    print("\n=== Overall Results ===")
    print(f"Mean IoU   : {np.nanmean([r['mean_iou'] for r in all_totals]):.4f}")
    print(f"Pixel Acc : {np.nanmean([r['pixel_acc'] for r in all_totals]):.4f}")
    print(f"Val Loss  : {np.nanmean([r['loss'] for r in all_totals]):.4f}")
    print(f"✔ Results written to {out_dir}")


if __name__ == "__main__":
    main()
