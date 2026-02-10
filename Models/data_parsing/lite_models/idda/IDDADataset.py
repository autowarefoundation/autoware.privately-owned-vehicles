import os
import glob
import cv2
import numpy as np

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset


class IDDADataset(BaseDataset):
    """
    Dataloader for the IDDAv2 dataset following the SAME structure as ACDCDataset.

    The labels are converted into the cityscapes format with 19 classes.

    - Only segmentation task supported.
    - We match image <-> label 1:1 by filename.
    - Handles gallery / queries_rain for both image and label folders.
    - Cameras (front, rear, left, right) included automatically.
    - Augmentations identical to ACDC pipeline.
    """

    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):
        """
        cfg structure:

        cfg = {
            "root": "/path/to/iddav2",
            "towns": ["town3", "town10"],   # optional: if not provided, auto-discover
            "splits": ["gallery", "queries_rain"],  # what folders to use
            "camera_groups": ["front_rear", "left_right"],
            "cameras": ["front", "rear", "left", "right"],
            "split": "train" or "val",
            "augmentations": {...}
        }
        """
        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
        self.root = dataset_root
        self.split = mode

        self.dataset_name = "IDDA"

        # If user does not list towns, auto-detect
        self.towns = ["town10", "town3"]

        # Folders to use for training/validation
        self.split_folders = ["gallery", "queries_rain"]  # default recommended training folders
        

        self.camera_groups = ["front_rear", "left_right"]
        

        self.cameras = ["front", "rear", "left", "right"]
        

        # ---- Build file list ----
        self.samples = self._build_file_list()


    # ----------------------------------------------------------------------
    # Build file list (matching image <-> mask by filename)
    # ----------------------------------------------------------------------
    def _build_file_list(self):
        samples = []
        print(f"[IDDAV2Dataset] Building file list for split '{self.split}', data_type='{self.data_type}'")

        val_count = 0         #take 500 images from the dataset to use for validation
        val_selected = 0
        val_limit = 500
        for town in self.towns:

            for folder in self.split_folders:
                img_root = os.path.join(
                    self.root, "images", town, folder
                )
                lbl_root = os.path.join(
                    self.root, "images", town, folder.replace("gallery", "gallery_labels")
                                            .replace("queries_rain", "queries_labels_rain")
                )

                for cam_group in self.camera_groups:
                    for cam in self.cameras:

                        img_dir = os.path.join(img_root, cam_group, cam)
                        lbl_dir = os.path.join(lbl_root, cam_group, cam)

                        if not os.path.isdir(img_dir):
                            continue
                        if not os.path.isdir(lbl_dir):
                            continue

                        # Collect all images inside the directory
                        img_files = glob.glob(
                            os.path.join(img_dir, "*.jpg"),
                            recursive=False
                        )

                        for img_path in img_files:
                            filename = os.path.basename(img_path)
                            base = filename.replace(".jpg", "")
                            
                            #TODO: support as well the depth maps 

                            lbl_path = os.path.join(lbl_dir, base + ".png")

                            #select for validastion only
                            if self.split == "val":
                                val_count += 1
                                if val_count % 10 != 0:
                                    continue
                                else:
                                    val_selected += 1

                            if os.path.isfile(lbl_path):
                                samples.append((img_path, lbl_path))
                            else:
                                print(f"[IDDAV2Dataset] WARNING: no label for {img_path}")

                            if self.split == "val" and val_selected >= val_limit:
                                break

        print(f"[IDDAV2Dataset] Loaded {len(samples)} samples for split '{self.split}'")
        return samples


    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)


    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # Load RGB image (OpenCV loads BGR)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #remove the alpha channel
        image = image[:, :, ::-1]  # change to BGR (discard alpha)

        # --------------------------------------------------
        # 2) LOAD / FAKE GT
        # --------------------------------------------------
        if self.pseudo_labeling is False:
            if self.data_type == "SEGMENTATION":
                # Load segmentation mask (bgr, and hten extract the 3rd channel)
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

                label = np.asarray(label, np.float32)[:, :, 2]
                #map the iddav2 training labels to cityscapes labels
                label = self.idda_cs_mapping(label)

            elif self.data_type == "DEPTH" and os.path.isfile(gt_path):
                label = np.load(gt_path)
            else:
                raise ValueError(
                    f"[IDDADataset] ERROR: unsupported data_type: {self.data_type}"
                )
        else:
            # fake GT (placeholder)
            if self.data_type == "SEGMENTATION":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            elif self.data_type == "DEPTH":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                raise ValueError(
                    f"[IDDADataset] ERROR: unsupported data_type: {self.data_type}"
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

    #from https://github.com/valeriopaolicelli/SegVPR.git, SegVPR/src/datasets.py
    def idda_cs_mapping(self, mask):
        mapping = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 16, 15, 12, 9, 10, 255, 255, 255, 255, 255, 14]
        mask_copy = mask.copy()
        for c in range(len(mapping)):
            mask_copy[mask == c] = mapping[c]
        return mask_copy


CITYSCAPES_COLORS = np.array([
    [128,  64,128],  # 0 - road
    [244,  35,232],  # 1 - sidewalk
    [ 70,  70, 70],  # 2 - building
    [102, 102,156],  # 3 - wall
    [190, 153,153],  # 4 - fence
    [153, 153,153],  # 5 - pole
    [250, 170, 30],  # 6 - traffic light
    [220, 220,  0],  # 7 - traffic sign
    [107, 142, 35],  # 8 - vegetation
    [152, 251,152],  # 9 - terrain
    [ 70, 130,180],  # 10 - sky
    [220,  20, 60],  # 11 - person
    [255,   0,  0],  # 12 - rider
    [  0,   0,142],  # 13 - car
    [  0,   0, 70],  # 14 - truck
    [  0,  60,100],  # 15 - bus
    [  0,  80,100],  # 16 - train
    [  0,   0,230],  # 17 - motorcycle
    [119,  11, 32],  # 18 - bicycle
    [  0,   0,   0] # 255 - unlabeled
], dtype=np.uint8)


def visualize_idda_to_cityscapes(image_rgb, raw_mask_idda, mapped_mask_cs, class_names=None, save_path="vis_debug.png"):
    """
    Visualize IDDA ➝ Cityscapes mapping.
    Shows:
        - original RGB image (left)
        - mapped segmentation mask in Cityscapes colors (center)
        - legend (right)
    """

    # -------------------------------------------------------------
    # Handle ignore label (255)
    # -------------------------------------------------------------
    ignore_mask = (mapped_mask_cs == 255)

    safe_mask = mapped_mask_cs.copy()
    safe_mask[ignore_mask] = 19  # index 19 just to allow lookup

    colored_mask = CITYSCAPES_COLORS[safe_mask]

    # -------------------------------------------------------------
    # 2. Prepare legend panel
    # -------------------------------------------------------------
    if class_names is None:
        class_names = [
            "road","sidewalk","building","wall","fence","pole",
            "traffic light","traffic sign","vegetation","terrain","sky",
            "person","rider","car","truck","bus","train","motorcycle","bicycle"
        ]

    H, W = mapped_mask_cs.shape
    legend_w = int(W * 0.35)

    legend = np.ones((H, legend_w, 3), dtype=np.uint8) * 255

    row_h = H // len(class_names)
    for i, cname in enumerate(class_names):

        color = CITYSCAPES_COLORS[i].tolist()

        y1 = i * row_h
        y2 = min(H, y1 + row_h)

        # color block
        cv2.rectangle(
            legend,
            (10, y1 + 5),
            (50, y2 - 5),
            color,  
            -1
        )

        # text label
        cv2.putText(
            legend, cname,
            (60, y1 + row_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,0),
            1,
            cv2.LINE_AA
        )

    # -------------------------------------------------------------
    # 3. Concatenate: image | mask | legend
    # -------------------------------------------------------------
    # Ensure all are same height
    image_rgb_resized = cv2.resize(image_rgb, (W, H))

    combined = np.concatenate([image_rgb_resized, colored_mask, legend], axis=1)

    # -------------------------------------------------------------
    # 4. Save / show result
    # -------------------------------------------------------------
    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    return combined
