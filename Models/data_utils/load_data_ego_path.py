#! /usr/bin/env python3

import json
import os
import pathlib
import numpy as np
from PIL import Image, ImageDraw
from typing import Literal, get_args
from check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    "BDD100K", 
    "COMMA2K19", 
    "CULANE", 
    "CURVELANES", 
    "ROADWORK", 
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))
VAL_SET_FRACTION = 0.1


class LoadDataEgoPath():
    def __init__(
            self, 
            labels_filepath: str,
            images_filepath: str,
            dataset: VALID_DATASET_LITERALS,
    ):
        
        # ================= Parsing param ================= #

        self.label_filepath = labels_filepath
        self.image_dirpath = images_filepath
        self.dataset_name = dataset

        # ================= Preliminary checks ================= #

        if not (self.dataset_name in VALID_DATASET_LIST):
            raise ValueError("Unknown dataset! Contact our team so we can work on this.")
        
        with open(self.label_filepath, "r") as f:
            self.labels = json.load(f)
        self.images = sorted([
            f for f in pathlib.Path(self.image_dirpath).glob("*.png")
        ])

        self.N_labels = len(self.labels)
        self.N_images = len(self.images)

        # Sanity check func by Mr. Zain
        checkData = CheckData(
            self.N_images,
            self.N_labels
        )
        
        # ================= Initiate data loading ================= #

        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        self.N_trains = 0
        self.N_vals = 0

        if (checkData.getCheck()):
            for set_idx, frame_id in enumerate(self.labels):
                if (set_idx % 10 == VAL_SET_FRACTION * 10):
                    # Slap it to Val
                    self.val_images.append(str(self.images[set_idx]))
                    self.val_labels.append(self.labels[frame_id])
                    self.N_vals += 1 
                else:
                    # Slap it to Train
                    self.train_images.append(str(self.images[set_idx]))
                    self.train_labels.append(self.labels[frame_id])
                    self.N_trains += 1

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")
        print(f"Val/Total = {(self.N_vals / (self.N_trains + self.N_vals))}")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals

    # ================= Get item at index ith, returning img and EgoPath ================= #
    
    # For train
    def getItemTrain(self, index):
        img_train = Image.open(str(self.train_images[index])).convert("RGB")
        label_train = self.train_labels[index]["drivable_path"]
        
        return np.array(img_train), label_train
    
    # For val
    def getItemVal(self, index):
        img_val = Image.open(str(self.val_images[index])).convert("RGB")
        label_val = self.val_labels[index]["drivable_path"]
        
        return np.array(img_val), label_val
    
    def sampleItemsAudit(
            self, 
            set_type: Literal["train", "val"], 
            vis_output_dir: str,
            n_samples: int = 100
    ):
        print(f"Now sampling first {n_samples} images from {set_type} set for audit...")

        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)

        for i in range(n_samples):

            # Fetch numpy image and ego path
            if set_type == "train":
                np_img, ego_path = self.getItemTrain(i)
            elif set_type == "val":
                np_img, ego_path = self.getItemVal(i)
            else:
                raise ValueError(f"sampleItemAudit() does not recognize set type {set_type}")
            
            # Convert back to image
            img = Image.fromarray(np_img)

            # Draw specs
            draw = ImageDraw.Draw(img)
            lane_color = (255, 255, 0)
            lane_w = 5
            img_width, img_height = img.size
            frame_name = str(i).zfill(5) + ".png"

            # Renormalize
            ego_path = [
                (point[0] * img_width, point[1] * img_height) 
                for point in ego_path
            ]

            # Now draw
            draw.line(ego_path, fill = lane_color, width = lane_w)

            # Then, save
            img.save(os.path.join(vis_output_dir, frame_name))

        print(f"Sampling all done, saved at {vis_output_dir}")


if __name__ == "__main__":
    # Testing cases here
    CurveLanesDataset = LoadDataEgoPath(
        labels_filepath = "/home/tranhuunhathuy/Documents/Autoware/pov_datasets/processed_CurveLanes/drivable_path.json",
        images_filepath = "/home/tranhuunhathuy/Documents/Autoware/pov_datasets/processed_CurveLanes/image",
        dataset = "CURVELANES",
    )
    CurveLanesDataset.sampleItemsAudit(
        "train",
        "/home/tranhuunhathuy/Documents/Autoware/pov_datasets/processed_CurveLanes/dataloader_sample"
    )


    # 1. Run this through all datasets to make sure they work (make sure it compatible with all JSON formats)
    # 2. Visualize the frames (first 100 samples) from getTrainVal/getItemVal
    # 3. Have a look at CurveLanes, then let Devang knows this as well