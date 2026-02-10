# dataloader/CurveLanesDataset.py

import os
import glob

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset


"""
CurveLanes Dataset (processed version).

Expected structure:

CurveLanes/processed/
├── image/            (*.jpg)
├── mask/             (*.png)
├── visualization/
└── drivable_path.json

Each image must have a corresponding mask with the same basename.
Example:
    image/000123.jpg
    mask/000123.png
"""

class CurveLanesDataset(BaseDataset):

    def __init__(
        self,
        dataset_root: str,
        aug_cfg: dict = {},
        mode: str = "train",
        data_type: str = "LANE_DETECTION",
        pseudo_labeling: bool = False,
    ):
        super().__init__(
            dataset_root,
            aug_cfg=aug_cfg,
            mode=mode,
            data_type=data_type,
            pseudo_labeling=pseudo_labeling,
        )

        self.root = dataset_root

        # Force processed version
        if "processed" not in os.path.basename(self.root):
            print(
                "[CurveLanesDataset] WARNING: dataset_root does not point to 'processed/'. "
                "Appending '/processed'."
            )
            self.root = os.path.join(self.root, "processed")

        self.split = mode.lower()   # "train" | "val" | "test"
        self.dataset_name = "Curvelanes"

        if self.data_type != "LANE_DETECTION":
            raise ValueError(
                f"[CurveLanesDataset] Unsupported data_type: {self.data_type}. "
                "Only 'LANE_DETECTION' is supported."
            )

        # ---- Build file list ----
        self.samples = self._build_file_list()

    # ------------------------------------------------------------
    # Build file list (replicates authors' split logic)
    # ------------------------------------------------------------
    def _build_file_list(self):

        MAX_VAL_SAMPLES = 500       #limit max number of validation samples to 500 (otherwise they would be )

        """
        Replicates the original authors' splitting logic:
            - Sort all frames deterministically
            - Every 10th sample goes to validation
            - The rest goes to training

        Returns:
            samples: list[(img_path, gt_path)]
        """

        print(
            f"[CurveLanesDataset] Building file list for split='{self.split}', "
            f"data_type='{self.data_type}'"
        )

        image_root = os.path.join(self.root, "image")
        mask_root  = os.path.join(self.root, "mask")

        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Missing mask directory: {mask_root}")

        # --------------------------------------------------
        # Collect and sort all images
        # --------------------------------------------------
        img_files = sorted(
            glob.glob(os.path.join(image_root, "*.jpg"))
        )

        if len(img_files) == 0:
            raise RuntimeError(f"[CurveLanesDataset] No images found in {image_root}")

        print(f"[CurveLanesDataset] Found {len(img_files)} images total.")

        samples = []

        # --------------------------------------------------
        # Deterministic split (authors logic)
        # --------------------------------------------------
        for idx, img_path in enumerate(img_files):

            basename = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(mask_root, f"{basename}.png")

            if not os.path.isfile(gt_path):
                print(f"[CurveLanesDataset] WARNING: Missing GT mask for {img_path}")
                continue

            is_val = (idx % 10 == 0)

            if self.split == "train" and not is_val:
                samples.append((img_path, gt_path))

            elif self.split == "val" and is_val and (len(samples) < MAX_VAL_SAMPLES):
                #cap validation samples to MAX_VAL_SAMPLES
                samples.append((img_path, gt_path))
            


        print(
            f"[CurveLanesDataset] Loaded {len(samples)} samples for split='{self.split}'."
        )

        if len(samples) == 0:
            raise RuntimeError(
                f"[CurveLanesDataset] Empty dataset split='{self.split}'. "
                "Check dataset path and split logic."
            )

        return samples
