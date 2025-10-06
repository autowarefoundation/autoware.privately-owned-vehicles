#! /usr/bin/env python3

import os
import json
import pathlib
import numpy as np
import math
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '..',
    '..'
)))
from PIL import Image
from typing import Literal, get_args
from Models.data_utils.check_data import CheckData

# Currently limiting to available datasets only. Will unlock eventually
VALID_DATASET_LITERALS = Literal[
    # "BDD100K",
    # "COMMA2K19",
    #"CULANE",
    "CURVELANES",
    # "ROADWORK",
    "TUSIMPLE",
    "OPENLANE"
]
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))

FIXED_HOMOTRANS_DATASETS = [
    "CULANE",
    "TUSIMPLE"
]

DYNAMIC_HOMOTRANS_DATASETS = [
    "CURVELANES",
    "OPENLANE"
]


class LoadDataAutoSteer():
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

        # Load JSON labels, get homotrans matrix as well
        with open(self.label_filepath, "r") as f:
            json_data = json.load(f)
            if (self.dataset_name in FIXED_HOMOTRANS_DATASETS):
                homotrans_mat = json_data.pop("standard_homomatrix")
                self.BEV_to_image_transform = np.linalg.inv(homotrans_mat)
            else:
                self.BEV_to_image_transform = None
            self.labels = json_data

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
        self.train_ids = []
        self.val_images = []
        self.val_labels = []
        self.val_ids = []

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
                        self.val_ids.append(frame_id)
                        self.N_vals += 1 
                    else:
                        # Slap it to Train
                        self.train_images.append(str(self.images[set_idx]))
                        self.train_labels.append(self.labels[frame_id])
                        self.train_ids.append(frame_id)
                        self.N_trains += 1
                else:
                    raise ValueError(f"Mismatch data detected in {self.dataset_name}!")

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")

    # Get sizes of Train/Val sets
    def getItemCount(self):
        return self.N_trains, self.N_vals
    
    def calcSegMask(self, egoleft, egoright):
        
        # Left/Right ego lane points defining a contour
        contour_points = []

        # Parse egoleft lane
        for i in range (0, len(egoleft)):
            point = egoleft[i]
            contour_points.append([point[0]*640, point[1]*320])

        # Parse egoright lane
        for i in range (0, len(egoright)):
            point = egoright[len(egoright) - i - 1]
            contour_points.append([point[0]*640, point[1]*320])

        # Get binary segmentation mask
        contour = np.array(contour_points, dtype=np.int32)
        binary_segmenation = np.zeros([320, 640], np.uint8)
        cv2.drawContours(binary_segmenation,[contour],0,(255),-1)
        return binary_segmenation

    def calcData(self, ego_left, ego_right, ego_path):
        x_left_lane_offset = ego_left[0][0]
        x_right_lane_offset = ego_right[0][0]
        x_ego_path_offset = ego_path[0][0]

        start_angle = math.atan((ego_path[2][0]*640 - ego_path[0][0]*640)/abs((ego_path[2][1]*320 - ego_path[0][1]*320)))
        end_angle_denominator = abs((ego_path[-1][1]*320 - ego_path[0][1]*320))
        if (end_angle_denominator < 1e-5):
            end_angle = 0.0
        else:
            end_angle = math.atan((ego_path[-1][0]*640 - ego_path[0][0]*640) / end_angle_denominator)

        data = [
            x_left_lane_offset, 
            x_right_lane_offset, 
            x_ego_path_offset,
            start_angle,
            end_angle
        ]

        return data
       
    # Get item at index ith, returning img and EgoPath
    def getItem(self, index, is_train: bool):
        if (is_train):

            # BEV Image
            bev_img = Image.open(str(self.train_images[index])).convert("RGB")

            # Frame ID
            frame_id = self.train_ids[index]

            # Raw image path
            raw_img_path = (
                self.train_labels[index]["perspective_img_path"]
                if self.dataset_name in ["OPENLANE"]
                else None
            )

            # BEV-to-image transform
            bev_to_image_transform = (
                self.BEV_to_image_transform
                if (self.BEV_to_image_transform is not None)
                else np.linalg.inv(self.train_labels[index]["homomatrix"])
            )

            # BEV EgoPath
            bev_egopath = self.train_labels[index]["bev_egopath"]
            bev_egopath = [lab[0:2] for lab in bev_egopath]

            # Reprojected EgoPath
            reproj_egopath = self.train_labels[index]["reproj_egopath"]
            reproj_egopath = [lab[0:2] for lab in reproj_egopath]

            # BEV EgoLeft Lane
            bev_egoleft = self.train_labels[index]["bev_egoleft"]
            bev_egoleft = [lab[0:2] for lab in bev_egoleft]

            # Reprojected EgoLeft Lane
            reproj_egoleft = self.train_labels[index]["reproj_egoleft"]
            reproj_egoleft = [lab[0:2] for lab in reproj_egoleft]

            # BEV EgoRight Lane
            bev_egoright = self.train_labels[index]["bev_egoright"]
            bev_egoright = [lab[0:2] for lab in bev_egoright]

            # Reprojected EgoRight Lane
            reproj_egoright = self.train_labels[index]["reproj_egoright"]
            reproj_egoright = [lab[0:2] for lab in reproj_egoright]

            # Binary Segmentation Mask
            binary_seg = self.calcSegMask(reproj_egoleft, reproj_egoright)

            # Data vector
            data = self.calcData(reproj_egoleft, reproj_egoright, reproj_egopath)
           
        else:

            # BEV Image
            bev_img = Image.open(str(self.val_images[index])).convert("RGB")

            # Frame ID
            frame_id = self.val_ids[index]

            # Raw image path
            raw_img_path = (
                self.val_labels[index]["perspective_img_path"]
                if self.dataset_name in ["OPENLANE"]
                else None
            )

            # BEV-to-image transform
            bev_to_image_transform = (
                self.BEV_to_image_transform
                if (self.BEV_to_image_transform is not None)
                else np.linalg.inv(self.val_labels[index]["homomatrix"])
            )
            
            # BEV EgoPath
            bev_egopath = self.val_labels[index]["bev_egopath"]
            bev_egopath = [lab[0:2] for lab in bev_egopath]

            # Reprojected EgoPath
            reproj_egopath = self.val_labels[index]["reproj_egopath"]
            reproj_egopath = [lab[0:2] for lab in reproj_egopath]

            # BEV EgoLeft Lane
            bev_egoleft = self.val_labels[index]["bev_egoleft"]
            bev_egoleft = [lab[0:2] for lab in bev_egoleft]

            # Reprojected EgoLeft Lane
            reproj_egoleft = self.val_labels[index]["reproj_egoleft"]
            reproj_egoleft = [lab[0:2] for lab in reproj_egoleft]

            # BEV EgoRight Lane
            bev_egoright = self.val_labels[index]["bev_egoright"]
            bev_egoright = [lab[0:2] for lab in bev_egoright]
            
            # Reprojected EgoRight Lane
            reproj_egoright = self.val_labels[index]["reproj_egoright"]
            reproj_egoright = [lab[0:2] for lab in reproj_egoright]

            # Binary Segmentation Mask
            binary_seg = self.calcSegMask(reproj_egoleft, reproj_egoright)

            # Data vector
            data = self.calcData(reproj_egoleft, reproj_egoright, reproj_egopath)

        # Convert image to OpenCV/Numpy format for augmentations
        bev_img = np.array(bev_img)

        return [
            frame_id, bev_img, raw_img_path,
            binary_seg, data,
            self.BEV_to_image_transform,
            bev_egopath, reproj_egopath,
            bev_egoleft, reproj_egoleft,
            bev_egoright, reproj_egoright,
        ]

