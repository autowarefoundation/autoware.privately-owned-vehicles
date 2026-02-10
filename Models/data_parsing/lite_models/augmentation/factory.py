# dataloader/augmentations/factory.py

from Models.data_parsing.lite_models.augmentation.segmentation import SegmentationAugmentation
from Models.data_parsing.lite_models.augmentation.depth import DepthAugmentation



def build_aug(data_type: str, cfg: dict, mode: str, pseudo_labeling: bool = False):
    data_type = data_type.upper()

    if data_type == "SEGMENTATION":
        return SegmentationAugmentation(mode, cfg)
    elif data_type == "DEPTH":
        return DepthAugmentation(mode, cfg, pseudo_labeling=pseudo_labeling)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
