#!/usr/bin/env python3
# evals/lane_detection/eval.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2

from utils.training import load_yaml, set_global_seed

from evals.utils.helpers import (
    build_validator_egolanes,
    ensure,
    lanes_overlay_simple,
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
    ap.add_argument(
        "--image",
        type=str,
        help="Path to a single PNG/JPG image for visual-only inference (no GT required)"
    )
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

        # --------------------------------------------------------
        # Build model backend
        # --------------------------------------------------------

        if args.engine:
            print("[INFO] Using TensorRT engine for inference")
            model = TensorRTWrapper(args.engine, device=device)


    # ============================================================
    # Single image inference (VISUAL ONLY)
    # ============================================================
    if args.image:
        img_path = Path(args.image)
        run_single_image_inference(
            model=model,
            image_path=img_path,
            device=device,
            out_dir=out_dir,
            cfg=cfg,
        )
        return


def run_single_image_inference(
    model,
    image_path: Path,
    device,
    out_dir: Path,
    cfg,
):
    """
    Visual-only segmentation inference on a single image.
    Saves RGB image + predicted mask visualization.
    """

    assert image_path.exists(), f"Image not found: {image_path}"

    # ---- load image ----
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H0, W0 = img_rgb.shape[:2]

    # ---- resize if requested (same logic as eval) ----
    aug_cfg = cfg["dataset"].get("augmentations", {})
    if aug_cfg.get("rescaling", {}).get("enabled", False):
        mode = aug_cfg["rescaling"]["mode"]
        if mode == "fixed_resize":
            H = aug_cfg["rescaling"]["height"]
            W = aug_cfg["rescaling"]["width"]
            img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    # ---- normalize (ImageNet, SAME AS TRAINING) ----
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    # HWC → CHW
    img = img.transpose(2, 0, 1)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    # ---- forward ----
    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        
    out_image = lanes_overlay_simple(img_t[0], logits[0])

    # ---- save outputs ----
    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_lanes.png"),
                cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))

    full_path = out_dir / f"{stem}_lanes.png"
    print(f"[OK] Single image inference saved:")
    print(f"     - {full_path}")


if __name__ == "__main__":
    main()
