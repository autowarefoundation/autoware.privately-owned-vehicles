#!/usr/bin/env python3
# evals/segmentation/eval_segmentation.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2


from utils.segmentation import validate_segmentation, mask_to_cityscapes

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
    ap.add_argument(
        "--image",
        type=str,
        help="Path to a single PNG/JPG image for visual-only inference (no GT required)"
    )

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

    model = trainer.model
    # ============================================================
    # Build model backend
    # ============================================================

    # resolve checkpoint
    if args.engine:
        #override model with TensorRT engine
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
        pred = torch.argmax(logits, dim=1)[0]   # [H,W]

    pred_np = pred.cpu().numpy()

    # ---- visualization ----
    pred_color = mask_to_cityscapes(pred_np)

    # resize back to original image size
    pred_color = cv2.resize(
        pred_color, (W0, H0), interpolation=cv2.INTER_NEAREST
    )

    overlay = (0.6 * img_rgb + 0.4 * pred_color).astype(np.uint8)

    # ---- save outputs ----
    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_input.png"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_pred.png"),
                cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[OK] Single image inference saved:")
    print(f"     - {stem}_input.png")
    print(f"     - {stem}_pred.png")
    print(f"     - {stem}_overlay.png")




if __name__ == "__main__":
    main()
