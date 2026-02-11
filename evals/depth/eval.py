
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2


from evals.utils.helpers import remap_scene3d_checkpoint, ensure



import argparse
from utils.training import load_yaml, set_global_seed
from training.depth.scene3d_trainer import Scene3DTrainer
from training.depth.deeplabv3plus_trainer import DeepLabV3PlusTrainer
from training.depth.unetplusplus_trainer import UnetPlusPlusTrainer
from utils.depth import validate_depth
# Legacy model
from evals.utils.helpers import (
    remove_value_layers,
    CHECKPOINT_ROOT,
    CONFIG_ROOT,
)

# ============================================================
# Paths
# ============================================================
TRAIN_ROOT   = Path("runs/training/depth")
EVAL_ROOT    = Path("runs/evals/depth")
        
from evals.utils.trt_engine import TensorRTWrapper

# ============================================================
# Main
# ============================================================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Generic depth evaluation (SMP + custom)")
    
    ap.add_argument(
        "--run", 
        help="Path containng the folder used for training. Example : \"/home/sergey/DEV/AI/AEI/runs/training/depth/deeplabv3plus_efficientnetb0_v0\""
    )
    ap.add_argument("--model", type=str, default="scene3d", help="Model to train: scene3d | deeplabv3plus | unetplusplus")
    ap.add_argument("--best", action="store_true", help="Use best checkpoint instead of last")
    ap.add_argument("--checkpoint", type=str, default="", help="Path to specific checkpoint to eval. It is used instead of --run and --best")
    ap.add_argument("--engine", type=str, default="", help="Path to TensorRT engine file to eval. It is used instead of --run and --best")


    # data
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)

    # eval
    ap.add_argument("--pseudo", default=True, action="store_false", help="Use DepthAnythingV2 for pseudo-labeling instead of ground-truth")
    ap.add_argument("--viz", type=int, default=10)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    #get the name of the last directory in the run path
    run_name = args.run.split("/")[-1] if args.run is not None else "eval_depth_model"
    out_dir_name = f"{run_name}_best" if args.best else f"{run_name}_last"
    out_dir = EVAL_ROOT / out_dir_name

    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.run is not None:
        print(f"Using the experiment : {args.run}")
        run_config_path = Path(args.run) / CONFIG_ROOT 
        print(f"Looking for config.yaml file at : {run_config_path}")
        cfg = load_yaml(run_config_path)

        #remove every intermediate dict layer called "value" from every cfg entry
        cfg = remove_value_layers(cfg)

        #replace the training and validation sets used in the config file. (Remove the ones in the training and modify the ones specified by the current CLI by arguments).
        #removing the training sets
        cfg["dataset"]["training_sets"] = list()            #empty
        cfg["dataset"]["validation_sets"] = args.datasets   #use the CLI specified ones

        #change experiment name to "val" to avoid the saving
        cfg["experiment"]["name"] = "val"

        #experiment wanbd enabled to false
        cfg["experiment"]["wandb"]["enabled"] = False


        seed = cfg.get("experiment", {}).get("seed", 42)
        set_global_seed(seed)

        print(f"Model being used : {args.model}")

        if args.model == "scene3d":
            validator = Scene3DTrainer(cfg)
        elif args.model == "deeplabv3plus":
            validator = DeepLabV3PlusTrainer(cfg)
        elif args.model == "unetplusplus":
            validator = UnetPlusPlusTrainer(cfg)
        else:
            raise ValueError(f"Unknown model type: {args.model}")
    
        aug_cfg = cfg["dataset"]["augmentations"]
        print(aug_cfg)

    # elif args.run is None and ensure(args.checkpoint):
    #     print(f"run path not specified: proceeding with checkpoint : {args.checkpoint}")
    elif args.engine is not None:
        engine_path = Path(args.engine)
        ensure(engine_path, "TensorRT engine")

        print("[INFO] Using TensorRT engine for inference")
        model = TensorRTWrapper(engine_path)
    
    else:
        ValueError("Run or checkpints files are not valid. Exiting")


    for ds_name, loader in validator.val_loaders.items():
        all_totals = []
        v_total, v_mAE, v_edge, v_absrel, vis_images = validate_depth(
            model=validator.model if validator else model,
            dataloader=loader,
            loss_module=validator.loss_fn_val,
            device=validator.device,
            logger=None,
            step=0,
            dataset_name=ds_name,
            pseudo_label_generator_model=validator.pseudo_labeler if validator.pseudo_labeling else None,
        )
        
        all_totals.append(v_total)
        print(f"  {ds_name}: total={v_total:.6f}, mAE={v_mAE:.6f}, edge={v_edge:.6f}, absrel={v_absrel:.6f}")


        mean_val = float(np.mean(all_totals))
        print(f"\n==> Mean total over datasets: {mean_val:.6f}")

        #save the vis images on disk
        for i, vis in enumerate(vis_images):
            img_name = out_dir / f"{ds_name}_sample_{i:03d}.png"
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_name), vis)


    
    #modify the vis images to cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    print(f"\n✔ Results saved to {out_dir}")


if __name__ == "__main__":
    main()
