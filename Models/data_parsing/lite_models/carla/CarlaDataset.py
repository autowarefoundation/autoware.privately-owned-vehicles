import os
import glob
import cv2
import numpy as np
import json
import os

from Models.data_parsing.lite_models.BaseDataset.BaseDataset import BaseDataset

TASK_TO_FOLDER = {
    "SEGMENTATION": "semantic_id",
    "DEPTH": "depth_raw_mm",
    "LANES": "lanes",
    "BBOXES_2D": "bboxes_2d",
}


class CarlaDataset(BaseDataset):
    """
    Generic CARLA dataset loader.

    Structure:
        root/
            TownXX/
                time_of_day/
                    weather/
                        scene_xxxxx/
                            rgb/
                            semantic_id/
                            depth_raw_mm/
                            lanes/
                            bboxes_2d/
    """

    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling: bool = False):

        super().__init__(
            dataset_root,
            aug_cfg=aug_cfg,
            mode=mode,
            data_type=data_type,
            pseudo_labeling=pseudo_labeling,
        )

        self.root = dataset_root
        self.split = mode
        self.data_type = data_type.upper()

        assert self.data_type in TASK_TO_FOLDER, \
            f"Unsupported data_type: {self.data_type}"

        self.gt_folder = TASK_TO_FOLDER[self.data_type]

        self.towns = ["Town01", "Town02", "Town04", "Town06", "Town07", "Town10HD"]
        self.times = ["Day", "Evening", "Night"]
        self.weathers = ["Clear", "Fog", "Rain", "Snow"]

        self.dataset_name = "CARLA"

        self.samples = self._build_file_list()

    # ------------------------------------------------------------------

    def _build_file_list(self):
        """
        Build file list from frames.jsonl.

        Each line is expected to be a JSON object with:
            entry["files"]["rgb"]
            entry["files"][self.gt_folder]
        """
        samples = []

        print(f"[CarlaDataset] Building file list | split={self.split} | task={self.data_type}")

        json_path = os.path.join(self.root, self.split, "frames.jsonl")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"frames.jsonl not found: {json_path}")

        with open(json_path, "r", encoding="utf-8", errors="strict") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[CarlaDataset] ERROR parsing line {line_idx}: {e}")
                    continue

                files = entry.get("files", {})

                rgb_path = files.get("rgb", None)
                gt_path = files.get(self.gt_folder, None)

                # sanity checks
                if rgb_path is None:
                    print(f"[CarlaDataset] WARNING missing RGB at line {line_idx}")
                    continue

                if not os.path.isfile(rgb_path):
                    print(f"[CarlaDataset] WARNING RGB file not found: {rgb_path}")
                    continue

                # pseudo-labeling → GT può mancare
                if self.pseudo_labeling:
                    samples.append((rgb_path, gt_path))
                    continue

                # normal training → GT deve esistere
                if gt_path is None:
                    print(
                        f"[CarlaDataset] WARNING missing GT ({self.gt_folder}) "
                        f"for frame {entry.get('frame_id')}"
                    )
                    continue

                if not os.path.isfile(gt_path):
                    print(f"[CarlaDataset] WARNING GT file not found: {gt_path}")
                    continue

                samples.append((rgb_path, gt_path))

        print(f"[CarlaDataset] Loaded {len(samples)} samples")
        return samples


    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)
