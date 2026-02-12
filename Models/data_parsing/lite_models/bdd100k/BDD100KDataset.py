import os
import glob

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset



"""
BDD100K Dataset class for segmentation
Dataset structure:
.
├── 100k
│   ├── test
│   ├── train
│   └── val
├── 10k
│   ├── test
│   ├── train
│   └── val
├── color_labels
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

Choose between 10k and 100k by changing the root folder.
splits : train, val, test

invalid paths (due to wrong 90 degrees rotation) are hardcoded in self.invalid_paths
self.invalid_paths = {
    /home/sergey/DEV/AI/datasets/BDD100K/10k/train/3d581db5-2564fb7e.jpg
    /home/sergey/DEV/AI/datasets/BDD100K/10k/train/52e3fd10-c205dec2.jpg
    /home/sergey/DEV/AI/datasets/BDD100K/10k/train/781756b0-61e0a182.jpg
    /home/sergey/DEV/AI/datasets/BDD100K/10k/train/78ac84ba-07bd30c2.jpg
    /home/sergey/DEV/AI/datasets/BDD100K/10k/val/80a9e37d-e4548ac1.jpg
    /home/sergey/DEV/AI/datasets/BDD100K/10k/val/9342e334-33d167eb.jpg
}
"""

#invalid paths due to pitch black images
invalid_paths = [
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/train/3d581db5-2564fb7e.jpg",
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/train/52e3fd10-c205dec2.jpg",
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/train/781756b0-61e0a182.jpg",
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/train/78ac84ba-07bd30c2.jpg",
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/val/80a9e37d-e4548ac1.jpg",
    "/home/sergey/DEV/AI/datasets/BDD100K/10k/val/9342e334-33d167eb.jpg",
]

class BDD100KDataset(BaseDataset):
    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling: bool = False):

        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)

        self.root = dataset_root
        self.split = mode  # "train", "val", "test"
        self.bdd_config = "10k"  # "10k" or "100k"

        self.dataset_name = "BDD100K"

        # ---- Build file lists ----
        self.samples = self._build_file_list()


    def _build_file_list(self):
        print(
            f"[BDD100KDataset] Building file list: "
            f"config={self.bdd_config}, split={self.split}, data_type={self.data_type}"
        )

        if self.data_type == "SEGMENTATION":
            label_dir = "labels"
            label_suffix = "_train_id"
            label_ext = ".png"
        elif self.data_type == "DEPTH":
            label_dir = "depth"
            label_suffix = "_depth"
            label_ext = ".npy"
        else:
            raise ValueError(f"[BDD100KDataset] ERROR: unsupported data_type: {self.data_type}")

        img_root = os.path.join(self.root, self.bdd_config, self.split)
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"[BDD100KDataset] ERROR: image folder not found: {img_root}")

        img_files = glob.glob(os.path.join(img_root, "*.jpg"))
        # img_files += glob.glob(os.path.join(img_root, "*.png"))

        img_files = sorted(img_files)

        print(f"[BDD100KDataset] Found {len(img_files)} images under: {img_root}")

        need_gt = (self.split in ["train", "val"])
        gt_root = os.path.join(self.root, label_dir, self.split)

        samples = []
        missing_gt = 0
        blacklisted = 0

        for img_path in img_files:

            # --- fast skip via blacklist ---
            if img_path in invalid_paths:
                blacklisted += 1
                print(
                    "[BDD100KDataset] WARNING: blacklisted image skipped\n"
                    f"  image: {img_path}"
                )
                continue

            if not need_gt:
                samples.append((img_path, None))
                continue

            fname = os.path.basename(img_path)

            #build label name depending on the data type (segmentaion, depth)
            gt_name = os.path.splitext(fname)[0] + label_suffix + label_ext

            gt_path = os.path.join(gt_root, gt_name)

            if not os.path.isfile(gt_path):
                missing_gt += 1
                print(
                    "[BDD100KDataset] WARNING: missing GT\n"
                    f"  image: {img_path}\n"
                    f"  expected: {gt_path}"
                )
                continue

            samples.append((img_path, gt_path))

        print(f"[BDD100KDataset] Final dataset summary for split='{self.split}':")
        print(f"  ✔ valid samples:        {len(samples)}")
        print(f"  ✖ missing GT skipped:  {missing_gt}")
        print(f"  ✖ blacklisted images:  {blacklisted}")

        if need_gt and len(samples) == 0:
            raise RuntimeError("[BDD100KDataset] ERROR: 0 valid samples found.")

        return samples
