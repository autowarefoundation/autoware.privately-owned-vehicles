#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from typing import Any
from PIL import Image, ImageDraw, ImageFont


# ============================= Format functions ============================= #


PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]
Line = list[PointCoords] | list[ImagePointCoords]


def round_line_floats(
    line: Line, 
    ndigits: int = 3
):
    """
    Round the coordinates of a line to a specified number of decimal places.
    """

    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)

    return line


def normalizeCoords(
    line: Line, 
    width: int, 
    height: int
):
    """
    Normalize the coords of line points.
    """
    return [
        (x / width, y / height) 
        for x, y in line
    ]


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
    """
    (x2, y2) = line[0]
    (x1, y1) = line[
        # int(len(line) / 5) 
        # if (
        #     len(line) > 5 and
        #     line[0][1] >= H * 0.8
        # ) else 1
        int(len(line) / 2)
    ]
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