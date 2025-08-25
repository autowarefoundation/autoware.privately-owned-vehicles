#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw


# ============================= Format functions ============================= #


def round_line_floats(line, ndigits = 6):
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


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """
    return [(x / width, y / height) for x, y in line]


def getLineAnchor(line, new_img_height):
    """
    Determine "anchor" point of a line.
    Unlike other datasets, since the resolution of each line in this dataset is
    too fine, I'm taking first point and 11th point.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[10]

    for i in range(1, len(line) - 1, 1):
        if (line[i][0] != x2) & (line[i][1] != y2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (new_img_height - b) / a

    return (x0, a, b)


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    # FYI: https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md

    IMAGE_SPLITS = [
        "training", 
        "validation"
    ]
    IMG_DIR = "images"
    LABEL_SPLITS = {
        "lane3d_1000_training" : [
            "training",
        ],
        "lane3d_1000_validation_test" : [
            "validation",
            # "test" not included
        ]
    }

    # All 200k scenes have reso 1920 x 1280. I checked it.
    W = 1920
    H = 1280

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process OpenLane dataset - groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "OpenLane raw directory",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. files you wanna limit, instead of whole set.",
        required = False
    )

    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stopping after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    Due to the huge dataset size, and since we don't have to edit the raw images,
    I have decided to not outputing the raw image files, but instead only the
    visualizations and groundtruth JSON.

    --output_dir
        |----visualization
        |----drivable_path.json

    """

    list_subdirs = ["visualization"]

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

    for label_split, list_label_subdirs in LABEL_SPLITS.items():
        
        for subsplit in list_label_subdirs:
            subsplit_path = os.path.join(
                dataset_dir,
                label_split,
                subsplit
            )

            for segment in sorted(os.listdir(subsplit_path)):
                segment_path = os.path.join(subsplit_path, segment)

                for label_file in sorted(os.listdir(segment_path)):                    
                    label_file_path = os.path.join(segment_path, label_file)
                    with open(label_file_path, "r") as f:
                        this_label_data = json.load(f)

                    this_label_data = parseData(this_label_data)
                    annotateGT(this_label_data)