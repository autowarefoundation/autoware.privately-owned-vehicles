# utils/utils_training.py
import os
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from Models.data_parsing.lite_models.acdc.ACDCDataset import ACDCDataset
from Models.data_parsing.lite_models.mapillary.MapillaryDataset import MapillaryDataset
from Models.data_parsing.lite_models.muses.MUSESDataset import MUSESDataset
from Models.data_parsing.lite_models.idda.IDDADataset import IDDADataset
from Models.data_parsing.lite_models.bdd100k.BDD100KDataset import BDD100KDataset
from Models.data_parsing.lite_models.cityscapes.CityscapesDataset import CityscapesDataset
from Models.data_parsing.lite_models.curvelanes.CurveLanesDataset import CurveLanesDataset
from Models.data_parsing.lite_models.tusimple.TUSimpleDataset import TUSimpleDataset
from Models.data_parsing.lite_models.idda.IDDADataset import IDDADataset


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # good for conv nets


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)



def build_dataloader(dataset, cfg_dl, mode: str):
    # print("[Dataloader] config:", cfg_dl)
    if cfg_dl is None:
        #used for evaluation
        cfg_dl = {"batch_size": 2, "shuffle_train": True, "shuffle_val": False, "num_workers": 2, "pin_memory": True, "drop_last": mode == "train", "persistent_workers": True if mode == "train" else False, "prefetch_factor": 2 if mode == "train" else 1}

    if "mapillary" in dataset.__class__.__name__.lower():
        # MapillaryDataset needs batch_size=1 due to variable image sizes. otherwise usually OOM
        val_batch_size = 1
    elif "batch_size_val" in cfg_dl:
        # use specific val batch size if provided
        val_batch_size = cfg_dl["batch_size_val"]
    else:
        val_batch_size = 4

    return DataLoader(
        dataset,
        batch_size=cfg_dl.get("batch_size", 8) if mode == "train" else val_batch_size,
        shuffle=cfg_dl.get("shuffle_train", True) if mode == "train" else cfg_dl.get("shuffle_val", False),
        num_workers=cfg_dl.get("num_workers", 4),
        pin_memory=cfg_dl.get("pin_memory", True),
        drop_last=cfg_dl.get("drop_last", mode == "train"),
        persistent_workers=cfg_dl.get("persistent_workers", True) if mode == "train" else False,   #true only for training
        prefetch_factor=cfg_dl.get("prefetch_factor", 2) if mode == "train" else 1                 #default 2 only for training
    )


def build_single_dataset(name: str, dataset_root: str, aug_cfg: dict, mode: str, data_type:str = "SEGMENTATION", pseudo_labeling: bool = False):
    """
    Factory loader for each dataset.
    Loads config YAML and constructs the dataset instance.
    """

    name = name.lower()

    if data_type not in ["SEGMENTATION", "DEPTH", "LANE_DETECTION"]:
        raise ValueError(f"Unsupported data_type '{data_type}' in build_single_dataset()")

    if name == "acdc":
        return ACDCDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

    elif name == "mapillary":
        return MapillaryDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
    
    elif name == "muses":
        return MUSESDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

    elif name == "idda" or name == "iddav2":
        return IDDADataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

    elif name == "carla":
        return CarlaDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

    elif name == "bdd100k":
        return BDD100KDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
    
    elif name == "cityscapes":
        return CityscapesDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
    
    elif name == "curvelanes":
        return CurveLanesDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type)

    elif name == "tusimple":
        return TUSimpleDataset(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type)
    else:
        raise ValueError(f"Unsupported dataset '{name}' in build_single_dataset()")


def get_unique_experiment_dir(root_out, exp_name):
    """
    If root_out/exp_name exists, create exp_name_1, exp_name_2, ...
    Returns (final_exp_name, final_out_dir)
    """
    base_dir = os.path.join(root_out, exp_name)
    if not os.path.exists(base_dir):
        return exp_name, base_dir

    idx = 1
    while True:
        new_name = f"{exp_name}_{idx}"
        new_dir = os.path.join(root_out, new_name)
        if not os.path.exists(new_dir):
            return new_name, new_dir
        idx += 1
