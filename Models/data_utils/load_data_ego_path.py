#! /usr/bin/env python3

import os
import json
import pathlib
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    "BDD100K", 
    "COMMA2K19", 
    "CULANE", 
    "CURVELANES", 
    "ROADWORK", 
    "TUSIMPLE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))


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
        
        # Load JSON labels, address the diffs of format across datasets
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

                # Check if there might be frame ID mismatch - happened to CULane before, just to make sure
                frame_id_from_img_path = str(self.images[set_idx]).split("/")[-1].replace(".png", "")
                if (frame_id == frame_id_from_img_path):

                    if (set_idx % 10 == 0):
                        # Slap it to Val
                        self.val_images.append(str(self.images[set_idx]))
                        self.val_labels.append(self.labels[frame_id])
                        self.N_vals += 1 
                    else:
                        # Slap it to Train
                        self.train_images.append(str(self.images[set_idx]))
                        self.train_labels.append(self.labels[frame_id])
                        self.N_trains += 1
                else:
                    raise ValueError(f"Mismatch data detected in {self.dataset_name}!")

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")
        print(f"Val/Total = {(self.N_vals / (self.N_trains + self.N_vals)):03f}")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals
    
    # Point/lane auto audit
    def dataAudit(self, label: list):
        # Convert all points into sublists in case they are tuples - yeah it DOES happens
        if (type(label[0]) == tuple):
            label = [[x, y] for [x, y] in label]

        # Trimming points whose y > 1.0
        fixed_label = label.copy()
        fixed_label = [
            point for point in fixed_label
            if point[1] <= 1.0
        ]

        # Make sure points are bottom-up
        start_y, end_y = fixed_label[0][1], fixed_label[-1][1]
        if (end_y > start_y):       # Top-to-bottom annotation, must reverse
            fixed_label.reverse()

        # Slightly decrease y if y == 1.0
        for i, point in enumerate(fixed_label):
            if (point[1] == 1.0):
                fixed_label[i][1] = 0.9999

        # Comma2k19's jumpy point (but can be happening to others as well)
        for i in range(1, len(fixed_label)):
            if (fixed_label[i][1] > fixed_label[i - 1][1]):
                fixed_label = fixed_label[0 : i]
                break

        return fixed_label
    
    # Get item at index ith, returning img and EgoPath
    def getItem(self, index, is_train: bool):
        if (is_train):
            img = Image.open(str(self.train_images[index])).convert("RGB")
            label = self.train_labels[index]["drivable_path"]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            label = self.val_labels[index]["drivable_path"]

        # Point/line auto audit
        label = self.dataAudit(label)
        
        return np.array(img), label