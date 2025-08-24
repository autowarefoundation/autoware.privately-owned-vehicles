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


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    # FYI: https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md

    IMAGE_SPLITS = [
        "training", 
        "validation"
    ]
    IMG_DIR = "images"
    LABEL_SPLITS = {
        "lane3d_1000_training" : [],
        "lane3d_1000_validation_test" : [
            "validation"
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

    