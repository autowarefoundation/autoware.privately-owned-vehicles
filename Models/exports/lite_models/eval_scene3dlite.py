#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import onnxruntime as ort

from Models.training.scene3d_lite_trainer import Scene3DLiteTrainer
from Models.data_utils.lite_models.helpers.depth import validate_depth
from Models.data_utils.lite_models.helpers.training import set_global_seed
from Models.exports.lite_models.helpers import ensure, SCENE3DLITE_DEFAULT_CONFIG


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

    ap = argparse.ArgumentParser("Scene3D Lite depth evaluation (checkpoint-based)")

    # Required
    ap.add_argument("--checkpoint", help=".pth checkpoint")
    ap.add_argument("--onnx", help=".onnx model (overrides checkpoint)")
    ap.add_argument("--datasets", nargs="+", required=True)

    # Network override
    ap.add_argument("--backbone", default="efficientnet_b1")
    ap.add_argument("--decoder-channels", type=int, default=256)
    ap.add_argument("--head-depth", type=int, default=1)
    ap.add_argument("--head-mid-channels", type=int, default=None)
    ap.add_argument("--head-kernel-size", type=int, default=3)
    ap.add_argument("--head-upsampling", type=int, default=4)

    # Resize
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=1)

    # Runtime
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz", type=int, default=10)
    ap.add_argument("--pseudo", type=bool, default=True, help="Enable pseudo labeling")
    ap.add_argument("--out-dir", default="runs/eval/scene3d_lite_eval")

    args = ap.parse_args()
    device = torch.device(args.device)

    # ============================================================
    # Build config
    # ============================================================

    cfg = SCENE3DLITE_DEFAULT_CONFIG.copy()

    # dataset override
    cfg["dataset"]["training_sets"] = []
    cfg["dataset"]["validation_sets"] = args.datasets
    cfg["dataset"]["pseudo_labeling"] = args.pseudo

    cfg["dataset"]["augmentations"]["rescaling"]["height"] = args.height
    cfg["dataset"]["augmentations"]["rescaling"]["width"] = args.width

    # network override
    cfg["network"]["backbone"]["type"] = args.backbone

    cfg["network"]["decoder"]["deeplabv3plus_decoder_channels"] = args.decoder_channels

    cfg["network"]["head"]["head_depth"] = args.head_depth
    cfg["network"]["head"]["head_mid_channels"] = args.head_mid_channels
    cfg["network"]["head"]["head_kernel_size"] = args.head_kernel_size
    cfg["network"]["head"]["head_upsampling"] = args.head_upsampling

    cfg["dataloader"]["batch_size_val"] = args.batch_size
    cfg["experiment"]["wandb"]["enabled"] = False
    cfg["experiment"]["name"] = "val"

    set_global_seed(cfg["experiment"]["seed"])

    # ============================================================
    # Backend selection
    # ============================================================

    if args.onnx:
        print("[INFO] Using ONNX backend")
        model = ONNXWrapper(args.onnx, device=args.device)

        trainer = Scene3DLiteTrainer(cfg)
        val_loaders = trainer.val_loaders
        loss_fn = trainer.loss_fn_val
        pseudo_labeler = trainer.pseudo_labeler if args.pseudo else None

    else:
        if args.checkpoint is None:
            raise ValueError("Either --checkpoint or --onnx must be provided")

        ensure(Path(args.checkpoint), "checkpoint")
        cfg["checkpoint"]["load_from"] = args.checkpoint

        trainer = Scene3DLiteTrainer(cfg)
        model = trainer.model.to(device).eval()

        val_loaders = trainer.val_loaders
        loss_fn = trainer.loss_fn_val
        pseudo_labeler = trainer.pseudo_labeler if args.pseudo else None

    # ============================================================
    # Output directory
    # ============================================================

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Evaluation
    # ============================================================

    all_totals = []

    for ds_name, loader in val_loaders.items():

        print(f"\n=== Evaluating dataset: {ds_name} ===")

        total, mAE, edge, absrel, vis_images = validate_depth(
            model=model,
            dataloader=loader,
            loss_module=loss_fn,
            device=device,
            logger=None,
            step=0,
            dataset_name=ds_name,
            pseudo_label_generator_model=pseudo_labeler,
        )

        print(f"  total  : {total:.6f}")
        print(f"  mAE    : {mAE:.6f}")
        print(f"  edge   : {edge:.6f}")
        print(f"  absrel : {absrel:.6f}")

        all_totals.append(total)

        for i, vis in enumerate(vis_images):
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(out_dir / f"{ds_name}_sample_{i:03d}.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            )

    print("\n=== Overall ===")
    print(f"Mean total over datasets: {float(np.mean(all_totals)):.6f}")
    print(f"✔ Results written to {out_dir}")


if __name__ == "__main__":
    main()
