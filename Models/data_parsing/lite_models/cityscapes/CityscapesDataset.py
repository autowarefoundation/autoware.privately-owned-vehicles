import os
import cv2
import numpy as np
import glob

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset



"""
Cityscapes Dataset class for segmentation
Dataset structure:
.
├── gtCoarse
│   ├── train
│   ├── train_extra
│   └── val
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    ├── train_extra
    └── val

Choose between fine and coarse by changing the root folder.
splits : train, val, test
"""






invalid_paths = [
    "/home/sergey/DEV/AI/datasets/cityscapes/leftImg8bit/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png",
]

class CityscapesDataset(BaseDataset):
    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):

        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

        
        self.root = dataset_root
        self.split = mode
        self.cityconfig = "fine"  # "fine" or "coarse"

        self.dataset_name = "cityscapes"

        self.pseudo_labeling = pseudo_labeling

        # ---- Build file lists ----
        self.samples = self._build_file_list()


        self.cityscapes_lut = self._build_cityscapes_lut()


    def cityscapes_id_to_trainid(self, mask):
        return self.cityscapes_lut[mask]


    def _build_file_list(self):
        print(
            f"[Cityscapes] Building file list: "
            f"config={self.cityconfig}, split={self.split}, data_type={self.data_type}"
        )

        # -------------------------
        # Label configuration
        # -------------------------
        if self.data_type == "SEGMENTATION":
            inner_folder = "gtFine" if self.cityconfig == "fine" else "gtCoarse"
            label_suffix = f"_{inner_folder}_labelIds"
            label_ext = ".png"
        elif self.data_type == "DEPTH":
            inner_folder = "depth"
            label_suffix = f"_{inner_folder}_depth"
            label_ext = ".npy"
        else:
            raise ValueError(f"[Cityscapes] ERROR: unsupported data_type: {self.data_type}")

        # Coarse uses train_extra
        split = self.split
        if split == "train" and self.cityconfig == "coarse":
            split = "train_extra"

        img_root = os.path.join(self.root, "leftImg8bit", split)
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"[Cityscapes] ERROR: image folder not found: {img_root}")

        # -------------------------
        # RECURSIVE image search
        # -------------------------
        img_files = sorted(
            glob.glob(
                os.path.join(img_root, "**", "*_leftImg8bit.png"),
                recursive=True,
            )
        )

        print(f"[Cityscapes] Found {len(img_files)} images under: {img_root}")

        need_gt = split in ["train", "val", "train_extra"]

        gt_root = os.path.join(self.root, inner_folder, split)

        samples = []
        missing_gt = 0
        blacklisted = 0

        for img_path in img_files:

            # --- blacklist ---
            if img_path in invalid_paths:
                blacklisted += 1
                print(
                    "[Cityscapes] WARNING: blacklisted image skipped\n"
                    f"  image: {img_path}"
                )
                continue

            # city name = parent folder
            city = os.path.basename(os.path.dirname(img_path))
            fname = os.path.basename(img_path)

            if not need_gt:
                samples.append((img_path, None))
                continue

            # Build GT filename
            gt_name = fname.replace("_leftImg8bit.png", "") + label_suffix + label_ext
            gt_path = os.path.join(gt_root, city, gt_name)


            #check GT existence only if not pseudo-labeling or if depth in val/test
            if not os.path.isfile(gt_path) and self.pseudo_labeling is False:
                missing_gt += 1
                print(
                    "[Cityscapes] WARNING: missing GT\n"
                    f"  image: {img_path}\n"
                    f"  expected: {gt_path}"
                )
                return

            samples.append((img_path, gt_path))

        print(f"[Cityscapes] Final dataset summary for split='{split}':")
        print(f"  ✔ valid samples:        {len(samples)}")
        print(f"  ✖ missing GT skipped:  {missing_gt}")
        print(f"  ✖ blacklisted images:  {blacklisted}")

        if need_gt and len(samples) == 0:
            raise RuntimeError("[Cityscapes] ERROR: 0 valid samples found.")

        return samples



    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # --------------------------------------------------
        # 1) LOAD RAW IMAGE (BGR → RGB)
        # --------------------------------------------------
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # 2) LOAD / FAKE GT
        # --------------------------------------------------
        if self.pseudo_labeling is False:
            if self.data_type == "SEGMENTATION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                label = self.cityscapes_id_to_trainid(label)
            elif self.data_type == "DEPTH" and os.path.isfile(gt_path):
                label = np.load(gt_path)
            else:
                raise ValueError(
                    f"[CityscapesDataset] ERROR: unsupported data_type: {self.data_type}"
                )
        else:
            # fake GT (placeholder)
            if self.data_type == "SEGMENTATION":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            elif self.data_type == "DEPTH":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                raise ValueError(
                    f"[CityscapesDataset] ERROR: unsupported data_type: {self.data_type}"
                )

        # --------------------------------------------------
        # 3) AUGMENTATION PIPELINE
        # --------------------------------------------------
        if self.pseudo_labeling and self.data_type == "DEPTH":
            image, label = self.aug.apply_augmentation(
                image, label, dataset_name=self.dataset_name
            )
        else:
            image, label = self.aug.apply_augmentation(
                image, label, dataset_name=self.dataset_name
            )

        # --------------------------------------------------
        # 4) FINAL CAST FOR MODEL
        # --------------------------------------------------
        image = image.astype(np.float32)
        label = label.astype(np.int64)

        image = np.transpose(image, (2, 0, 1))  # CHW

        # --------------------------------------------------
        # 5) RETURN SAMPLE (NON-BREAKING)
        # --------------------------------------------------
        sample = {
            "image": image,
            "gt": label,
        }

        return sample



    def _build_cityscapes_lut(self):
        #assign to ignore all, then refine
        lut = np.ones(256, dtype=np.uint8) * 255
        mapping = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
            17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13,
            27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }
        for k, v in mapping.items():
            lut[k] = v
        return lut
