# evals/utils.py
import torch
import sys
from network.awf.scene_seg_network import SceneSegNetwork
import numpy as np
from collections.abc import Mapping
from pathlib import Path
from training.lane_detection.egolanes_trainer import EgoLanesTrainer
from training.segmentation.deeplabv3plus_trainer import DeepLabV3PlusTrainer as DeepLabV3PlusTrainer_seg
from training.lane_detection.deeplabv3plus_trainer import DeepLabV3PlusTrainer as DeepLabV3PlusTrainer_lanes

from evals.utils.trt_engine import TensorRTWrapper
from utils.training import load_yaml, set_global_seed
# ============================================================
# Paths
# ============================================================
DATASET_ROOT = Path("/home/sergey/DEV/AI/datasets")
CONFIG_ROOT  = Path("logs/wandb/latest-run/files/config.yaml")             #relative to the --run folder
CHECKPOINT_ROOT = Path("checkpoints")                                   #relative to the --run folder



#change every backbone name from "Backbone.encoder.6.3.block.1.1.weight" to "Backbone.stages.6.3.block.1.1.weight"
def adaptBackboneStateDict(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        # caso da rinominare
        if k.startswith("Backbone.encoder."):
            k = k.replace(
                "Backbone.encoder.",
                "Backbone.stages.",
                1
            )

        # TUTTO il resto va mantenuto
        new_state[k] = v

    return new_state



def remap_scene3d_checkpoint(state_dict):
    """
    Fix key mismatch caused by old pretrainedBackBone nesting.
    """
    new_state = {}

    for k, v in state_dict.items():
        # Caso 1: pretrainedBackBone.stages.*  --> Backbone.stages.*
        if k.startswith("pretrainedBackBone."):
            new_k = "Backbone." + k.replace("pretrainedBackBone.", "")
            new_state[new_k] = v

        # Caso 2: Backbone.pretrainedBackBone.stages.* --> Backbone.stages.*
        elif k.startswith("Backbone.pretrainedBackBone."):
            new_k = "Backbone." + k.replace("Backbone.pretrainedBackBone.", "")
            new_state[new_k] = v

        else:
            new_state[k] = v

    return new_state

def normalize_class_names(cfg: dict) -> None:
    """
    Normalize cfg["loss"]["class_names"] when coming from W&B.

    W&B serializes YAML dict keys as strings:
        {"0": "road", "1": "sidewalk", ...}

    This function converts it to:
        ["road", "sidewalk", ...]

    The function mutates cfg in-place.
    """

    loss_cfg = cfg.get("loss", {})
    class_names = loss_cfg.get("class_names", None)

    # Nothing to do
    if class_names is None:
        return

    # Already in correct format
    if isinstance(class_names, list):
        return

    # Convert dict → ordered list
    if isinstance(class_names, dict):
        print("[INFO] Normalizing loss.class_names from dict → list")

        try:
            keys_sorted = sorted(class_names.keys(), key=lambda k: int(k))
            class_names_list = [class_names[k] for k in keys_sorted]

            # Optional sanity check
            num_classes = loss_cfg.get("num_classes", None)
            if num_classes is not None and len(class_names_list) != int(num_classes):
                raise ValueError(
                    f"class_names length ({len(class_names_list)}) "
                    f"!= num_classes ({num_classes})"
                )

            cfg["loss"]["class_names"] = class_names_list

            print(f"[INFO] Normalized class_names ({len(class_names_list)} classes)")

        except Exception as e:
            raise RuntimeError(
                f"Failed to normalize loss.class_names: {class_names}"
            ) from e

    else:
        raise TypeError(
            f"Unsupported type for loss.class_names: {type(class_names)}"
        )



def fatal(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def ensure(path, name):
    if not path.exists():
        fatal(f"{name} not found: {path}")


def remove_value_layers(obj):
    if isinstance(obj, Mapping):
        if set(obj.keys()) == {"value"}:
            return remove_value_layers(obj["value"])
        return {k: remove_value_layers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_value_layers(v) for v in obj]
    else:
        return obj


# ============================================================
# Legacy builder
# ============================================================

def build_legacy_sceneseg(checkpoint_path: Path, device: torch.device):

    print("[INFO] Building legacy SceneSeg model")

    model = SceneSegNetwork(is_testing=True)

    try:
        model.load_state_dict(torch.load(checkpoint_path)["model_state"])
        print("[INFO] Loaded model_state from checkpoint")
    except Exception:
        print("[INFO] Adapting backbone state dict...")
        state_dict = adaptBackboneStateDict(torch.load(checkpoint_path))
        model.load_state_dict(state_dict)
        print("[INFO] Loaded adapted state dict")

    model = model.to(device).eval()
    return model


# ============================================================
# EgoLanes validator builder
# ============================================================
def build_validator_egolanes(args) -> EgoLanesTrainer:


    args.height = args.height or 320
    args.width  = args.width  or 640
    # build minimal config for EgoLanesTrainer.
    #build dummy configs for scheduler and optimizer
    cfg = {
        "experiment": {
            "name": "val",
            "wandb": {
                "enabled": False,
            },
            "seed": 42,
        },
        "dataloader": {
            "batch_size": args.batch_size,
            "batch_size_val": 1,                    #1 since original EgoLanes does not support batching
            "num_workers": 4,
            "pin_memory": True,
        },
        "dataset": {
            "tusimple_root": str(DATASET_ROOT / "TUSimple"),
            "curvelanes_root": str(DATASET_ROOT / "Curvelanes"),
            "training_sets": list(),            # empty
            "validation_sets": args.datasets,   # use the CLI specified ones
            "augmentations": {
                "rescaling": {
                    "enabled": True,
                    "mode": "fixed_resize",
                    "height": args.height,
                    "width": args.width,
                    "scale_range": [1, 1],
                },
                "normalize": {
                    "enabled": False, # EgoLanes has its own normalization, so we disable it here
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
        },
        "scheduler": {
            "type": "warmup_cosine",
            "warmup_steps": 1000,
            "min_lr": 5e-6,
            "step_size": 30,
            "gamma": 0.1,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
        "training": {
            "grad_accum_steps": 1,
            "logging": {
                "log_every_steps": 50,
            },
            "max_epochs": 40,
            "max_steps": 80000,
            "mode": "steps",
            "save_best": True,
            "save_last": True,
            "validation": {
                "every_n_epochs": 1,
                "every_n_steps": 2000,
                "mode": "steps",
            },
        },
    }


    validator = EgoLanesTrainer(cfg)
    
    return validator



def resolve_checkpoint(run_dir: Path, best: bool) -> Path:
    ckpt_dir = run_dir / CHECKPOINT_ROOT
    if best:
        best_ckpts = list(ckpt_dir.glob("*best*.pth"))
        if not best_ckpts:
            fatal("No best checkpoint found")
        if len(best_ckpts) > 1:
            print("[WARNING] Multiple best checkpoints found, using first")
        return best_ckpts[0]
    return ckpt_dir / "last.pth"


def build_cfg_and_trainer_segmentation(
    run_dir: Path,
    datasets,
    batch_size,
    height,
    width,
    checkpoint_path=None,
):
    cfg_path = run_dir / CONFIG_ROOT
    ensure(cfg_path, "config.yaml")

    cfg = load_yaml(cfg_path)
    cfg = remove_value_layers(cfg)
    normalize_class_names(cfg)

    # Override datasets
    cfg["dataset"]["training_sets"]   = []
    cfg["dataset"]["validation_sets"] = datasets

    # Optional resize
    if height is not None and width is not None:
        cfg["dataset"]["augmentations"]["rescaling"] = {
            "enabled": True,
            "mode": "fixed_resize",
            "height": height,
            "width": width,
        }

    cfg["dataloader"]["batch_size_val"] = batch_size

    # Disable logging
    cfg["experiment"]["name"] = "val"
    cfg["experiment"]["wandb"]["enabled"] = False

    if checkpoint_path is not None:
        cfg["checkpoint"] = {
            "load_from": str(checkpoint_path),
            "strict_load": True,
        }

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = DeepLabV3PlusTrainer_seg(cfg)

    return cfg, trainer


def build_cfg_and_trainer_lanes(
    run_dir: Path,
    datasets,
    batch_size,
    height,
    width,
    checkpoint_path=None,   
):
    cfg_path = run_dir / CONFIG_ROOT
    ensure(cfg_path, "config.yaml")

    cfg = load_yaml(cfg_path)
    cfg = remove_value_layers(cfg)

    cfg["dataset"]["training_sets"]   = []
    cfg["dataset"]["validation_sets"] = datasets

    if height is not None and width is not None:
        cfg["dataset"]["augmentations"]["rescaling"] = {
            "enabled": True,
            "mode": "fixed_resize",
            "height": height,
            "width": width,
        }

    cfg["dataloader"]["batch_size_val"] = batch_size

    cfg["experiment"]["name"] = "val"
    cfg["experiment"]["wandb"]["enabled"] = False

    if checkpoint_path is not None:
        cfg["checkpoint"] = {
            "load_from": str(checkpoint_path),
            "strict_load": True,
        }

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    trainer = DeepLabV3PlusTrainer_lanes(cfg)
    return cfg, trainer

# ============================================================
import cv2

LANE_COLORS = np.array([
    [0, 255, 255],   # ego-left  (cyan)
    [255, 0, 200],   # ego-right (magenta)
    [0, 255, 145],   # other     (green)
], dtype=np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def lanes_overlay_simple(
    image_chw: torch.Tensor,
    logits_chw: torch.Tensor,
    threshold: float = 0.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    image_chw : [3,H,W] float, ImageNet normalized
    logits_chw: [3,h,w] float (possibly lower-res)
    returns   : [H,W,3] uint8 overlay image
    """

    # -------------------------
    # Denormalize image
    # -------------------------
    img = image_chw.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)  # H,W,3
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    H, W = img.shape[:2]

    # -------------------------
    # Binary masks from logits
    # -------------------------
    masks = (logits_chw.detach().cpu().numpy() > threshold)
    masks = masks.transpose(1, 2, 0)  # h,w,3

    h, w, _ = masks.shape

    # -------------------------
    # Upsample masks if needed
    # -------------------------
    if (h != H) or (w != W):
        masks_up = np.zeros((H, W, 3), dtype=bool)
        for c in range(3):
            masks_up[..., c] = cv2.resize(
                masks[..., c].astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        masks = masks_up

    # -------------------------
    # Apply colors
    # -------------------------
    colored = img.copy()
    for c in range(3):
        colored[masks[..., c]] = LANE_COLORS[c]

    # -------------------------
    # Alpha overlay
    # -------------------------
    out = cv2.addWeighted(colored, alpha, img, 1 - alpha, 0)
    return out
