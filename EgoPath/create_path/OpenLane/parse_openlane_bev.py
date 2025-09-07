#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw

PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]


# ============================== Format functions ============================== #


def round_line_floats(line, ndigits = 6):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def custom_warning_format(message, category, filename, lineno, line = None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def drawLine(
    img: np.ndarray, 
    line: list,
    color: tuple,
    thickness: int = 2
):
    for i in range(1, len(line)):
        pt1 = (
            int(line[i - 1][0]), 
            int(line[i - 1][1])
        )
        pt2 = (
            int(line[i][0]), 
            int(line[i][1])
        )
        cv2.line(
            img, 
            pt1, pt2, 
            color = color, 
            thickness = thickness
        )


def annotateGT(
    img: np.ndarray,
    orig_img: np.ndarray,
    frame_id: str,
    bev_egopath: list,
    reproj_egopath: list,
    bev_egoleft: list,
    reproj_egoleft: list,
    bev_egoright: list,
    reproj_egoright: list,
    raw_dir: str, 
    visualization_dir: str,
    normalized: bool
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # =========================== RAW IMAGE =========================== #
    
    # Save raw img in raw dir, as PNG
    cv2.imwrite(
        os.path.join(
            raw_dir,
            f"{frame_id}.png"
        ),
        img
    )

    # =========================== BEV VIS =========================== #

    img_bev_vis = img.copy()

    # Draw egopath
    if (normalized):
        renormed_bev_egopath = [
            (x * BEV_W, y * BEV_H)
            for x, y in bev_egopath
        ]
    else:
        renormed_bev_egopath = bev_egopath
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egopath,
        color = COLOR_EGOPATH
    )

    # Draw egoleft
    if (normalized):
        renormed_bev_egoleft = [
            (x * BEV_W, y * BEV_H)
            for x, y in bev_egoleft
        ]
    else:
        renormed_bev_egoleft = bev_egoleft
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw egoright
    if (normalized):
        renormed_bev_egoright = [
            (x * BEV_W, y * BEV_H)
            for x, y in bev_egoright
        ]
    else:
        renormed_bev_egoright = bev_egoright
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoright,
        color = COLOR_EGORIGHT
    )

    # Save visualization img in vis dir, as JPG (saving storage space)
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}.jpg"
        ),
        img_bev_vis
    )

    # =========================== ORIGINAL VIS =========================== #

    # Draw reprojected egopath
    if (normalized):
        renormed_reproj_egopath = [
            (x * W, y * H) 
            for x, y in reproj_egopath
        ]
    else:
        renormed_reproj_egopath = reproj_egopath
    drawLine(
        img = orig_img,
        line = renormed_reproj_egopath,
        color = COLOR_EGOPATH
    )
    
    # Draw reprojected egoleft
    if (normalized):
        renormed_reproj_egoleft = [
            (x * W, y * H) 
            for x, y in reproj_egoleft
        ]
    else:
        renormed_reproj_egoleft = reproj_egoleft
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw reprojected egoright
    if (normalized):
        renormed_reproj_egoright = [
            (x * W, y * H) 
            for x, y in reproj_egoright
        ]
    else:
        renormed_reproj_egoright = reproj_egoright
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoright,
        color = COLOR_EGORIGHT
    )

    # Save it
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}_orig.jpg"
        ),
        orig_img
    )


def calAngle(line: list[PointCoords]) -> float:
    """
    Calculate angle of a line with vertical axis at anchor point.
    - Vertical upward lane: 0°
    - Horizontal leftward lane: -90°
    - Horizontal rightward lane: +90°
    """
    return math.degrees(
        math.atan2(
            line[1][0] - line[0][0],
            -(line[1][1] - line[0][1])
        )
    )