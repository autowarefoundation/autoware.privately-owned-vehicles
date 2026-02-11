#!/usr/bin/env python3
# evals/object_detection/eval_singleimage_object_detection.py

import argparse
from pathlib import Path

from ultralytics import YOLO  # for type hints only
import torch
import numpy as np
import cv2

from evals.utils.helpers import (
    ensure,
    resolve_checkpoint,
    build_cfg_and_trainer_segmentation,  # (tuo helper: costruisce cfg + trainer + dataloader)
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
    0: (255, 255, 255),  # white
    1: (0, 0, 255),      # red
    2: (0, 255, 255),    # yellow
    3: (255, 255, 0),    # cyan
}


def draw_predictions(img_bgr, predictions, class_names=None, draw_scores=True):
    """
    predictions: list of [x1,y1,x2,y2,conf,class]
    img_bgr: uint8 BGR
    """
    out = img_bgr.copy()

    for x1, y1, x2, y2, conf, cls in predictions:
        cls_i = int(cls)
        color = COLOR_MAP.get(cls_i, (255, 255, 255))

        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)

        label = None
        if class_names is not None and 0 <= cls_i < len(class_names):
            label = class_names[cls_i]
        else:
            label = f"cls={cls_i}"

        if draw_scores:
            label = f"{label} {float(conf):.2f}"

        # draw label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(0, y1i - th - baseline - 4)
        cv2.rectangle(out, (x1i, y0), (x1i + tw + 6, y0 + th + baseline + 4), color, -1)
        cv2.putText(
            out,
            label,
            (x1i + 3, y0 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# Post-processing (YOLO-like)
# ============================================================

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N,4] in xywh (center-x, center-y, w, h)
    returns: [N,4] in xyxy
    """
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(preds: torch.Tensor, iou_thres: float = 0.45) -> torch.Tensor:
    """
    preds: [N,6] -> [x1,y1,x2,y2,score,cls]
    """
    from torchvision.ops import nms as tv_nms

    if preds.numel() == 0:
        return torch.empty(0, 6, device=preds.device)

    boxes = preds[:, :4]
    scores = preds[:, 4]
    keep = tv_nms(boxes, scores, iou_thres)
    return preds[keep]


def post_process_predictions(
    raw_predictions: torch.Tensor,
    num_classes: int = 10,
    conf_thres: float = 0.6,
    iou_thres: float = 0.45,
) -> torch.Tensor:
    """
    Generic YOLO-like postprocessing.

    Supports outputs:
      - [1, C, N] or [1, N, C]
    where:
      C = 4 + num_classes
      layout = [cx, cy, w, h, cls0, cls1, ..., cls_{K-1}]

    Returns:
      Tensor [K, 6] -> [x1, y1, x2, y2, conf, cls]
    """

    assert raw_predictions.ndim == 3, \
        f"Expected 3D tensor, got {raw_predictions.shape}"

    # --------------------------------------------------
    # Normalize layout → [N, C]
    # --------------------------------------------------
    if raw_predictions.shape[1] < raw_predictions.shape[2]:
        # [1, C, N] → [N, C]
        preds = raw_predictions.squeeze(0).permute(1, 0)
    else:
        # [1, N, C] → [N, C]
        preds = raw_predictions.squeeze(0)

    C = preds.shape[1]
    expected_C = 4 + num_classes
    if C != expected_C:
        raise ValueError(
            f"Invalid output shape: got C={C}, expected {expected_C} "
            f"(4 bbox + {num_classes} classes)"
        )

    # --------------------------------------------------
    # Split bbox + class logits
    # --------------------------------------------------
    boxes_xywh = preds[:, :4]
    class_logits = preds[:, 4:]  # [N, num_classes]

    # --------------------------------------------------
    # Class scores
    # --------------------------------------------------
    class_scores = class_logits.sigmoid()          # [N, K]
    scores, class_ids = torch.max(class_scores, 1)  # [N]

    # --------------------------------------------------
    # Confidence threshold
    # --------------------------------------------------
    keep = scores > conf_thres
    if keep.sum() == 0:
        return torch.empty(0, 6, device=preds.device)

    boxes_xyxy = xywh2xyxy(boxes_xywh[keep])

    detections = torch.cat(
        [
            boxes_xyxy,
            scores[keep].unsqueeze(1),
            class_ids[keep].float().unsqueeze(1),
        ],
        dim=1,
    )

    # --------------------------------------------------
    # NMS
    # --------------------------------------------------
    return nms(detections, iou_thres)



# ============================================================
# Pre/Post helpers
# ============================================================

def apply_fixed_resize_if_enabled(img_rgb: np.ndarray) -> np.ndarray:
    """
    Applies the SAME rescaling logic you use in eval (fixed_resize only),
    so single-image path matches training/eval.
    """
    aug_cfg = cfg.get("dataset", {}).get("augmentations", {})
    res_cfg = aug_cfg.get("rescaling", {})
    if res_cfg.get("enabled", False):
        mode = res_cfg.get("mode", "")
        if mode == "fixed_resize":
            H = int(res_cfg["height"])
            W = int(res_cfg["width"])
            img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return img_rgb


def normalize_imagenet(img_rgb: np.ndarray) -> np.ndarray:
    """
    img_rgb: uint8 RGB
    returns: float32 CHW normalized
    """
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # CHW
    return img


def scale_boxes_xyxy(preds_xyxy: np.ndarray, from_hw, to_hw) -> np.ndarray:
    """
    Scale boxes from (H_from, W_from) to (H_to, W_to).
    preds_xyxy: (N,4) in pixels for from_hw
    """
    Hf, Wf = from_hw
    Ht, Wt = to_hw
    sx = Wt / max(Wf, 1)
    sy = Ht / max(Hf, 1)

    out = preds_xyxy.copy()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    return out


# ============================================================
# Single image inference
# ============================================================

@torch.no_grad()
def run_single_image_inference(
    model,
    image_path: Path,
    device,
    out_dir: Path,
    conf_thres: float,
    iou_thres: float,
    class_names=None,
):
    """
    Single image object detection inference (PyTorch or TensorRTWrapper).
    Saves:
      - *_input.png
      - *_pred.png (bboxes drawn)
    """
    assert image_path.exists(), f"Image not found: {image_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load image ----
    img_bgr0 = cv2.imread(str(image_path))
    if img_bgr0 is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb0.shape[:2]

    # ---- apply same resize logic as eval/training (if enabled) ----
    H1, W1 = img_rgb0.shape[:2]

    # ---- normalize (ImageNet) ----
    chw = normalize_imagenet(img_rgb0)
    img_t = torch.from_numpy(chw).unsqueeze(0).to(device)

    # ---- forward ----
    model.eval()
    outputs = model(img_t)

    # ---- post-process ----
    preds = post_process_predictions(
        outputs,
        num_classes=len(class_names) if class_names is not None else 10,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )


    preds_cpu = preds.detach().cpu().numpy()  # (K,6)
    preds_list = preds_cpu.tolist()

    # ---- scale boxes back to original resolution if we resized ----
    if len(preds_list) > 0 and (H1 != H0 or W1 != W0):
        xyxy = preds_cpu[:, :4]
        xyxy_scaled = scale_boxes_xyxy(xyxy, from_hw=(H1, W1), to_hw=(H0, W0))
        preds_cpu[:, :4] = xyxy_scaled
        preds_list = preds_cpu.tolist()

    # ---- draw on original image ----
    vis_bgr = draw_predictions(img_bgr0, preds_list, class_names=class_names, draw_scores=True)

    # ---- save outputs ----
    stem = image_path.stem
    in_path = out_dir / f"{stem}_input.png"
    pr_path = out_dir / f"{stem}_pred.png"

    cv2.imwrite(str(in_path), img_bgr0)
    cv2.imwrite(str(pr_path), vis_bgr)

    print(f"[OK] Single image inference saved:")
    print(f"     - {in_path}")
    print(f"     - {pr_path}")

    if len(preds_list) == 0:
        print("[INFO] No detections above conf_thres.")
    else:
        print(f"[INFO] Detections: {len(preds_list)}")
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(preds_list[:25]):
            print(f"  [{i:02d}] cls={int(cls)} conf={conf:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Object detection SINGLE IMAGE inference (PyTorch / TensorRT)")

    ap.add_argument(
        "--run",
        required=True,
        help="Training run folder (config + dataloader). Used to load cfg & checkpoint unless --engine is set.",
    )

    ap.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path to a single PNG/JPG image for inference",
    )
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument(
        "--engine",
        help="TensorRT engine (.engine). If set, overrides checkpoint",
    )

    ap.add_argument(
        "--checkpoint",
        help="Optional checkpoint override (ignored if --engine is set)",
    )

    ap.add_argument("--best", action="store_true")

    # still required by your build_cfg_and_trainer_* helper (to instantiate cfg/trainer consistently)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--height", type=int)
    ap.add_argument("--width", type=int)

    # postprocess thresholds
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.45)

    # runtime
    ap.add_argument("--device", default="cuda")

    # output
    ap.add_argument("--out-dir", type=str, default=None)

    args = ap.parse_args()
    device = torch.device(args.device)

    run_dir = Path(args.run)
    ensure(run_dir, "run folder")

    # ============================================================
    # Output dir
    # ============================================================
    suffix = "trt" if args.engine else "pth"
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = EVAL_ROOT / f"{run_dir.name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")

    # ============================================================
    # Build config + trainer (ALWAYS) for consistent preprocessing
    # ============================================================
    ckpt_path = args.checkpoint or resolve_checkpoint(run_dir, args.best)

    # ============================================================
    # Build backend
    # ============================================================
    if args.engine:
        print(f"[INFO] Using TensorRT engine: {args.engine}")
        model = TensorRTWrapper(args.engine, device=device)
    else:
        model = YOLO(ckpt_path)
        model.to(device)

    # ============================================================
    # Single image inference
    # ============================================================
    run_single_image_inference(
        model=model,
        image_path=Path(args.image),
        device=device,
        out_dir=out_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
        class_names=None,  # se hai i nomi classi, passa una lista qui
    )


if __name__ == "__main__":
    main()
