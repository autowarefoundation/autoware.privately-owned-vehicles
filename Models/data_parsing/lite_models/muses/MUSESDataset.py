import os
import glob

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset


class MUSESDataset(BaseDataset):
    """
    MUSES segmentation / panoptic dataloader.
    Supports the following structure:

    ROOT/
    ├── RGB_Frame_Camera_trainvaltest/muses/frame_camera/{split}/{cond}/{day|night}/*.jpg
    ├── Semantic_Annotations_trainval/muses/gt_semantic/{split}/{cond}/{day|night}/*.png
    ├── Depth/muses/gt_depth/{split}/{cond}/{day|night}/*.npy
    ├── Panoptic_Annotations_trainval/muses/gt_panoptic/{train|val}/{cond}/{day|night}/*.png
    └── Uncertainty_Annotations_trainval/muses/gt_uncertainty/{train|val}/{cond}/{day|night}/*.png

    cfg:
        {
            "root": "...",
            "split": "train" or "val" or "test",
            "conditions": ["clear","fog"...],
            "time": ["day","night"] OR leave None for both,
            "label_type": "semantic" / "panoptic" / "uncertainty" / "depth",
            "augmentations": {...}
        }
    """

    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):
        
        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
        
        self.root = dataset_root
        self.split = mode                # train / val / test
        self.conditions = ["clear", "fog", "rain", "snow"]
        self.times = ["day", "night"]

        self.pseudo_labeling = pseudo_labeling

        self.dataset_name = "muses"
        # ---- Build file list ----
        self.samples = self._build_file_list()



    # ---------------------------------------------------------
    # BUILD FILE LIST
    # ---------------------------------------------------------
    def _build_file_list(self):
        samples = []
        print(f"[MUSESDataset] Building samples for split={self.split}, data_type={self.data_type}...")

        if self.data_type == "SEGMENTATION":
            inner_folder_1 = "Semantic_Annotations_trainval"
            inner_folder_2 = "gt_semantic"
            suffix = "gt_labelTrainIds"
            label_ext = ".png"
        elif self.data_type == "DEPTH":
            inner_folder_1 = "Depth_PseudoLabels"
            inner_folder_2 = "depth"
            suffix = "frame_camera_depth"
            label_ext = ".npy"
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")


        # 1) RGB ROOT
        rgb_root = os.path.join(
            self.root,
            "RGB_Frame_Camera_trainvaltest",
            "muses",
            "frame_camera",
            self.split
        )

        #LABEL ROOT, depending on label type (segmentation, depth)
        label_root = os.path.join(
            self.root,
            inner_folder_1,
            "muses",
            inner_folder_2,
            self.split
        )

        # -----------------------------------------------------
        # Loop on conditions + day/night folders
        # -----------------------------------------------------
        for cond in self.conditions:
            for time_of_day in self.times:

                # FULL PATHS
                rgb_dir = os.path.join(rgb_root, cond, time_of_day)
                gt_dir = os.path.join(label_root, cond, time_of_day)

                if not os.path.isdir(rgb_dir):
                    print(f"[MUSESDataset] WARNING: RGB path missing: {rgb_dir}")
                    continue
                if not os.path.isdir(gt_dir):
                    print(f"[MUSESDataset] WARNING: GT path missing: {gt_dir}")
                    continue

                # LOAD IMAGES
                img_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                if len(img_files) == 0:
                    print(f"[MUSESDataset] WARNING: No RGB images in {rgb_dir}")
                    continue

                # MATCH WITH GT
                for img_path in img_files:
                    base = os.path.splitext(os.path.basename(img_path))[0]
                    
                    #change name from REC0066_frame_497570_frame_camera.png to REC0066_frame_497570_gt_labelTrainIds.png in case of segmentation training
                    #change name from REC0066_frame_497570_frame_camera.png to REC0066_frame_497570_depth.npy in case of depth training
                    gt_filename = base.replace("frame_camera", suffix) + label_ext
                    gt_path = os.path.join(gt_dir, gt_filename)

                    if os.path.isfile(gt_path) or self.pseudo_labeling == True:
                        samples.append((img_path, gt_path))
                    else:
                        print(f"[MUSESDataset] WARNING: Missing GT for image: {img_path}")

        print(f"[MUSESDataset] Loaded {len(samples)} samples.")
        return samples