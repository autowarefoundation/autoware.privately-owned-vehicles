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


def parseData():


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
    CAM_DIR     = "cam01"  # Only GT for cam01 is provided.

    