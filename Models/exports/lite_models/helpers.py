# evals/utils.py
import sys
import numpy as np
from pathlib import Path


def fatal(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def ensure(path, name):
    if not path.exists():
        fatal(f"{name} not found: {path}")




# ============================================================
# EGO LANES DEFAULT SETTINGS
# ============================================================

DATASET_ROOT = Path("/home/sergey/DEV/AI/datasets")

EGOLANESLITE_DEFAULT_CONFIG = {

    "experiment": {
        "name": "val",
        "wandb": {"enabled": False},
        "seed": 42,
    },

    "dataset": {
        "tusimple_root": str(DATASET_ROOT / "TUSimple"),
        "curvelanes_root": str(DATASET_ROOT / "Curvelanes"),
        "training_sets": [],
        "validation_sets": [],
        "augmentations": {
            "rescaling": {
                "enabled": True,
                "mode": "fixed_resize",
                "height": 320,
                "width": 640,
                "scale_range": [1, 1],
            },
            "normalize": {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    },

    "dataloader": {
        "batch_size_val": 1,
        "num_workers": 2,
    },

    "network": {
        "model": "deeplabv3plus",
        "type": "custom",
        "output_channels": 3,
        "backbone": {
            "type": "efficientnet_b1",
            "pretrained": True,
            "encoder_depth": 5,
            "output_stride": 16,
            "load_encoder": False,
            "freeze_encoder": False,
        },
        "decoder": {
            "aspp_dilations": [12, 24, 36],
            "deeplabv3plus_decoder_channels": 64,
        },
        "head": {
            "head_activation": None,
            "head_depth": 1,
            "head_mid_channels": None,
            "head_upsampling": 1,
            "head_kernel_size": 3,
        },
    },

    "checkpoint": {
        "load_from": None,
        "strict_load": True,
    },
}



SCENESEGLITE_DEFAULT_CONFIG = {

    "experiment": {
        "name": "val",
        "wandb": {"enabled": False},
        "seed": 42,
    },

    "dataset": {
        "acdc_root":  str(DATASET_ROOT / "acdc"),
        "mapillary_root": str(DATASET_ROOT / "mapillary"),
        "muses_root": str(DATASET_ROOT / "MUSES"),
        "bdd100k_root": str(DATASET_ROOT / "BDD100K"),
        "cityscapes_root": str(DATASET_ROOT / "cityscapes"),
        "training_sets": [],
        "validation_sets": [],
        "augmentations": {
            "rescaling": {
                "enabled": True,
                "mode": "fixed_resize",
                "height": 320,
                "width": 640,
                "scale_range": [1, 1],
            },
            "normalize": {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    },
    "loss": {
        "type": "cross_entropy",
        "ignore_index": 255,

        "num_classes": 19,
        "class_names": {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "traffic_light",
            7: "traffic_sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motorcycle",
            18: "bicycle",
        },
    },

    "dataloader": {
        "batch_size_val": 1,
        "num_workers": 2,
    },

    "network": {
        "model": "deeplabv3plus",
        "type": "custom",
        "output_channels": 19,
        "backbone": {
            "type": "efficientnet_b1",
            "pretrained": True,
            "encoder_depth": 5,
            "output_stride": 16,
            "load_encoder": False,
            "freeze_encoder": False,
        },
        "decoder": {
            "aspp_dilations": [12, 24, 36],
            "deeplabv3plus_decoder_channels": 256,
        },
        "head": {
            "head_activation": None,
            "head_depth": 1,
            "head_mid_channels": None,
            "head_upsampling": 4,
            "head_kernel_size": 1,
        },
    },

    "checkpoint": {
        "load_from": None,
        "strict_load": True,
    },
}


SCENE3DLITE_DEFAULT_CONFIG = {

    "experiment": {
        "name": "val",
        "wandb": {"enabled": False},
        "seed": 42,
    },

    "dataset": {
        "pseudo_labeling": True,
        "acdc_root":  str(DATASET_ROOT / "acdc"),
        "mapillary_root": str(DATASET_ROOT / "mapillary"),
        "muses_root": str(DATASET_ROOT / "MUSES"),
        "bdd100k_root": str(DATASET_ROOT / "BDD100K"),
        "cityscapes_root": str(DATASET_ROOT / "cityscapes"),
        "training_sets": [],
        "validation_sets": [],
        "augmentations": {
            "rescaling": {
                "enabled": True,
                "mode": "fixed_resize",
                "height": 320,
                "width": 640,
                "scale_range": [1, 1],
            },
            "normalize": {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    },

    
    "dataloader": {
        "batch_size_val": 1,
        "num_workers": 2,
    },

    "network": {
        "model": "deeplabv3plus",
        "type": "custom",
        "output_channels": 1,
        "backbone": {
            "type": "efficientnet_b1",
            "pretrained": True,
            "encoder_depth": 5,
            "output_stride": 16,
            "load_encoder": False,
            "freeze_encoder": False,
        },
        "decoder": {
            "aspp_dilations": [12, 24, 36],
            "deeplabv3plus_decoder_channels": 256,
        },
        "head": {
            "head_activation": None,
            "head_depth": 1,
            "head_mid_channels": None,
            "head_upsampling": 4,
            "head_kernel_size": 3,
        },
    },

    "checkpoint": {
        "load_from": None,
        "strict_load": True,
    },

    "training": {
        "pseudo_labeler_generator" : "vitl"
    },

}