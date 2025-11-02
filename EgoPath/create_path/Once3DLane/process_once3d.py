#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw


# ============================= Format functions ============================= #


PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]
Line = list[PointCoords] | list[ImagePointCoords]


def normalizeCoords(
    line: Line, 
    width: int, 
    height: int
):
    """
    Normalize the coords of line points.
    """

    return [
        (
            x / width, 
            y / height
        ) 
        for x, y in line
    ]


def round_line_floats(
    line: Line, 
    ndigits: int = 3
):
    """
    Round the coordinates of a line to a 
    specified number of decimal places.
    Default is 3 decimal places.
    """

    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)

    return line


# Custom warning format
def custom_warning_format(
    message, 
    category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def getLineAnchor(
    line: Line,
    verbose: bool = False
):
    """
    Determine "anchor" point of a line.
    This function assumes lines are sorted by descending y-coords.
    So better sort it in advance. These raw GTs are messy as hell.
    """

    (x2, y2) = line[0]
    (x1, y1) = line[1]
    if (verbose):
        print(f"Anchor points chosen: ({x1}, {y1}), ({x2}, {y2})")

    if (x1 == x2) or (y1 == y2):
        return (x1, None, None)

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (H - b) / a
    if (verbose):
        print(f"Anchor point computed: (x0 = {x0}, a = {a}, b = {b})")

    return (x0, a, b)


# ============================== Core functions ============================== #


def parseData(
    img_id      : int,
    label_data  : dict
):
    """
    Parse Once3DLane data entry.
    """

    

    return anno_entry


# ================================= MAIN RUN ================================= #


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    # FYI: https://once-3dlanes.github.io/

    # The downloaded dataset is ASSUMED to be organized as follows:
    #
    # Once3DLane dataset/
    # │
    # ├── images/
    # │   ├── 000027/
    # │   │   ├── cam01/
    # │   │   │   ├── <frame_id>.jpg
    # │   │   │   ├── <frame_id>.jpg
    # │   │   │   └── ...
    # │   │   └── cam03/
    # │   │       ├── <frame_id>.jpg
    # │   │       ├── <frame_id>.jpg
    # │   │       └── ...
    # │   ├── 000028/
    # │   │   ├── cam01/
    # │   │   │   ├── <frame_id>.jpg
    # │   │   │   ├── <frame_id>.jpg
    # │   │   │   └── ...
    # │   │   └── cam03/
    # │   │       ├── <frame_id>.jpg
    # │   │       ├── <frame_id>.jpg
    # │   │       └── ...
    # │   └── ...
    # │
    # └── infos/
    #       ├── 000027/
    #       │   └── 000027.json
    #       ├── 000028/
    #       │   └── 000028.json
    #       └── ...
    # 
    # └── lanes/
    #       ├── 000027/
    #       │   └── cam01/
    #       │       ├── <frame_id>.json
    #       │       ├── <frame_id>.json
    #       │       └── ...
    #       ├── 000028/
    #       │   └── cam01/
    #       │       ├── <frame_id>.json
    #       │       ├── <frame_id>.json
    #       │       └── ...
    #       └── ...

    IMG_DIR     = "images"
    LABEL_DIR   = "lanes"
    INFO_DIR    = "infos"
    CAM_DIR     = "cam01"  # Only GT for cam01 is provided

    # Size checked from sample images of cam01 (cam03 is similar too)
    W = 1920
    H = 1020

    # ============================== Argument parser ============================== #

    parser = argparse.ArgumentParser(
        description = "Process Once3DLane dataset - LaneSeg GT generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        "-d",
        type = str, 
        help = "Once3DLane directory, containing above structure of images, lanes and infos",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        "-o",
        type = str,
        help = "Output directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        "-e",
        type = int,
        help = "Num. files you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir
    IMG_DIR = os.path.join(dataset_dir, IMG_DIR)
    LABEL_DIR = os.path.join(dataset_dir, LABEL_DIR)
    INFO_DIR = os.path.join(dataset_dir, INFO_DIR)
    output_dir = args.output_dir

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    --output_dir
        |----image
        |----mask
        |----visualization
        |----drivable_path.json
    """
    list_subdirs = [
        "image", 
        "mask",
        "visualization"
    ]
    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    data_master = {}
    img_id_counter = -1

    for segment_id in tqdm(
        sorted(os.listdir(IMG_DIR)), 
        desc = "Processing segments: ",
        colour = "yellow"
    ):
        
        segment_img_dir     = os.path.join(IMG_DIR, segment_id, CAM_DIR)
        segment_label_dir   = os.path.join(LABEL_DIR, segment_id, CAM_DIR)
        segment_info_path   = os.path.join(
            INFO_DIR, 
            segment_id, 
            f"{segment_id}.json"
        )

        list_current_segment_imgs   = sorted(os.listdir(segment_img_dir))
        list_current_segment_labels = sorted(os.listdir(segment_label_dir))
        assert len(list_current_segment_imgs) == len(list_current_segment_labels), \
            f"Number of images and labels do not match in segment {segment_id}!"

        # Process frame-by-frame
        for i in range(len(list_current_segment_imgs)):

            # Early stopping
            if (
                (early_stopping) and 
                (img_id_counter == early_stopping - 1)
            ):
                break

            img_id_counter += 1
            img_filename    = list_current_segment_imgs[i]
            label_filename  = list_current_segment_labels[i]
            img_path        = os.path.join(segment_img_dir, img_filename)
            label_path      = os.path.join(segment_label_dir, label_filename)

            # Parsing
            img = Image.open(img_path).convert("RGB")
            with open(label_path, "r") as f:
                label_data = json.load(f)

            anno_entry = parseData(
                img_id_counter,
                label_data
            )