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
from network.data_utils.check_data import CheckData

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

            if(point[0] < 0):
                fixed_label[i][0] = 0

            if(point[1] < 0):
                fixed_label[i][1] = 0

        # Comma2k19's jumpy point (but can be happening to others as well)
        for i in range(1, len(fixed_label)):
            if (fixed_label[i][1] > fixed_label[i - 1][1]):
                fixed_label = fixed_label[0 : i]
                break

        return fixed_label
    

    def fit_cubic_bezier(self, label):

        # Fit a Cubic Bezier curve to raw data points

        # Adding a tiny amount of noise to the endpoints to 
        # ensure we don't have a singular matrix error
        # when fitting the bezier curve in case it is a perfectly
        # straight line in the ground truth label
        label[0][0] = label[0][0] + 1e-4
        label[0][1] = label[0][1] + 1e-4
        label[-1][0] = label[-1][0] + 1e-4
        label[-1][1] = label[-1][1] + 1e-4

        # Chord length parameterization
        distances = np.sqrt(np.sum(np.diff(label, axis=0)**2, axis=1))
        cumulative = np.insert(np.cumsum(distances), 0, 0)
        t = cumulative / cumulative[-1]

        # Bézier basis functions
        def bernstein_matrix(t):
            t = np.asarray(t)
            B = np.zeros((len(t), 4))
            B[:, 0] = (1 - t)**3
            B[:, 1] = 3 * (1 - t)**2 * t
            B[:, 2] = 3 * (1 - t) * t**2
            B[:, 3] = t**3
            return B

        B = bernstein_matrix(t)

        # Least squares fitting: B * P = points => P = (B^T B)^-1 B^T * points
        BTB = B.T @ B
        BTP = B.T @ label

        # Initializing control points list and individual control points
        control_points = 0
        p0 = 0.0
        p1 = 0.0
        p2 = 0.0
        p3 = 0.0

        # Flag to store whether or not we had a correct Bezier fit
        is_valid = True

        try:
            # Calculate Bezier curve control points using Least-Squares
            control_points = np.linalg.lstsq(BTB, BTP, rcond=None)
        except:
            # If we run into an optimization error - set validity flag to False
            is_valid = False

        if(is_valid):
            # Get control points for cubic bezier curve
            p0 = control_points[0][0]
            p1 = control_points[0][1]
            p2 = control_points[0][2]
            p3 = control_points[0][3]
        
        return is_valid, p0, p1, p2, p3
       
    # Get item at index ith, returning img and EgoPath
    def getItem(self, index, is_train: bool):
        if (is_train):
            img = Image.open(str(self.train_images[index])).convert("RGB")
            label = self.train_labels[index]["drivable_path"]
        else:
            img = Image.open(str(self.val_images[index])).convert("RGB")
            label = self.val_labels[index]["drivable_path"]

        # Flag to store whether data is valid or not
        is_valid = True

        # Keypoints
        bezier_curve = 0

        # If there are enough points to fit a cubic bezier curve
        if(len(label) >= 5):

            # Point/line auto audit
            label = self.dataAudit(label)
        
            if(len(label) >= 4):
                # Fit a cubic bezier curve to raw data points
                is_valid, p0, p1, p2, p3 = self.fit_cubic_bezier(label)

                p0_ref_arr = np.array(p0)
                p1_ref_arr = np.array(p1)
                p2_ref_arr = np.array(p2)
                p3_ref_arr = np.array(p3)

                bezier_curve = np.concatenate((p0_ref_arr, p1_ref_arr, p2_ref_arr, p3_ref_arr), axis=0)
                bezier_curve = np.float32(bezier_curve)
            else:
                is_valid = False
        else:
            # Data is not valid since we need at least 4 raw data
            # points to fit a cubic bezier curve
            is_valid = False

        # Convert image to OpenCV/Numpy format for augmentations
        img = np.array(img)
        
        return img, bezier_curve, is_valid
    
  