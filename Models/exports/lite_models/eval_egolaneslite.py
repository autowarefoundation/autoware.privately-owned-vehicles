#!/usr/bin/env python3

import argparse
from pathlib import Path
from sympy import true
import torch
import numpy as np
import cv2

from Models.training.ego_lanes_lite_trainer import EgoLanesLiteTrainer
from Models.data_utils.lite_models.helpers.lanes import validate_lanes
from Models.data_utils.lite_models.helpers.training import set_global_seed
from Models.exports.lite_models.helpers import ensure, EGOLANESLITE_DEFAULT_CONFIG
import onnxruntime as ort


# ============================================================
# ONNX Wrapper
# ============================================================

class ONNXWrapper:
    def __init__(self, onnx_path, device="cuda"):
        providers = ["CUDAExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def eval(self):
        return self

    def __call__(self, x):
        x_np = x.detach().cpu().numpy()
        out = self.session.run(None, {self.input_name: x_np})[0]
        return torch.from_numpy(out).to(x.device)


# ============================================================
# MAIN
# ============================================================

@torch.no_grad()
def main():

    ap = argparse.ArgumentParser("EgoLanes Lite evaluation (checkpoint-based)")

    # Required
    ap.add_argument("--checkpoint", help=".pth checkpoint")
    ap.add_argument("--onnx", help=".onnx model (overrides checkpoint)")
    ap.add_argument("--datasets", nargs="+", required=True)

    # Network override
    ap.add_argument("--backbone", default="efficientnet_b1")
    ap.add_argument("--output-stride", type=int, default=16)
    ap.add_argument("--decoder-channels", type=int, default=64)
    ap.add_argument("--head-upsampling", type=int, default=1)

    # Resize
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=1)

    # Runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/eval/egolanes_lite_eval")
    ap.add_argument("--viz", type=int, default=10)

    args = ap.parse_args()
    device = torch.device(args.device)

    # ============================================================
    # Build config
    # ============================================================

    cfg = EGOLANESLITE_DEFAULT_CONFIG.copy()

    cfg["dataset"]["validation_sets"] = args.datasets
    cfg["dataset"]["augmentations"]["rescaling"]["height"] = args.height
    cfg["dataset"]["augmentations"]["rescaling"]["width"] = args.width

    cfg["network"]["backbone"]["type"] = args.backbone
    cfg["network"]["backbone"]["output_stride"] = args.output_stride
    cfg["network"]["decoder"]["deeplabv3plus_decoder_channels"] = args.decoder_channels
    cfg["network"]["head"]["head_upsampling"] = args.head_upsampling

    cfg["dataloader"]["batch_size_val"] = args.batch_size

    set_global_seed(cfg["experiment"]["seed"])

    # ============================================================
    # Backend selection
    # ============================================================

    if args.onnx:
        print("[INFO] Using ONNX backend")
        model = ONNXWrapper(args.onnx, device=args.device)

        trainer = EgoLanesLiteTrainer(cfg)
        val_loaders = trainer.val_loaders
        loss_fn = trainer.loss_fn

    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --onnx must be provided")

        ensure(Path(args.checkpoint), "checkpoint")

        cfg["checkpoint"]["load_from"] = args.checkpoint

        trainer = EgoLanesLiteTrainer(cfg)
        model = trainer.model.to(device).eval()

        val_loaders = trainer.val_loaders
        loss_fn = trainer.loss_fn

    # ============================================================
    # Evaluation
    # ============================================================

    all_totals = []

    for ds_name, loader in val_loaders.items():

        print(f"\n=== Evaluating: {ds_name} ===")

        results, vis_images = validate_lanes(
            model=model,
            dataloader=loader,
            loss_fn=loss_fn,
            device=device,
            dataset_name=ds_name,
            vis_count=args.viz,
        )

        print(f"Mean IoU  : {results['mean_iou']:.4f}")
        print(f"Pixel Acc : {results['pixel_acc']:.4f}")
        print(f"Val Loss  : {results['loss']:.4f}")

        all_totals.append(results)

        for i, vis in enumerate(vis_images):
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(Path(args.out_dir) / f"{ds_name}_sample_{i:03d}.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

    print("\n=== Overall ===")
    print(f"Mean IoU  : {np.nanmean([r['mean_iou'] for r in all_totals]):.4f}")
    print(f"Pixel Acc : {np.nanmean([r['pixel_acc'] for r in all_totals]):.4f}")
    print(f"Val Loss  : {np.nanmean([r['loss'] for r in all_totals]):.4f}")


if __name__ == "__main__":
    main()
