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


def imagePointTuplize(point: PointCoords) -> ImagePointCoords:
    """
    Parse all coords of an (x, y) point to int, making it
    suitable for image operations.
    """
    return (int(point[0]), int(point[1]))


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


def interpX(line, y):
    """
    Interpolate x-value of a point on a line, given y-value
    """
    points = np.array(line)
    list_x = points[:, 0]
    list_y = points[:, 1]

    if not np.all(np.diff(list_y) > 0):
        sort_idx = np.argsort(list_y)
        list_y = list_y[sort_idx]
        list_x = list_x[sort_idx]

    return float(np.interp(y, list_y, list_x))


def polyfit_BEV(
    bev_line: list,
    order: int,
    y_step: int,
    y_limit: int
):
    valid_line = [
        point for point in bev_line
        if (
            (0 <= point[0] < BEV_W) and 
            (0 <= point[1] < BEV_H)
        )
    ]
    if (not valid_line):
        warnings.warn("No valid points in BEV line for polyfit.")
        return None, None, None
    
    x = [
        point[0] 
        for point in valid_line
    ]
    y = [
        point[1] 
        for point in valid_line
    ]

    z = np.polyfit(y, x, order)
    f = np.poly1d(z)
    y_new = np.linspace(
        0, y_limit, 
        int(y_limit / y_step) + 1
    )
    x_new = f(y_new)

    # Sort by decreasing y
    fitted_bev_line = sorted(
        tuple(zip(x_new, y_new)),
        key = lambda x: x[1],
        reverse = True
    )

    flag_list = [0] * len(fitted_bev_line)
    for i in range(len(fitted_bev_line)):
        if (not 0 <= fitted_bev_line[i][0] <= BEV_W):
            flag_list[i - 1] = 1
            break
    if (not 1 in flag_list):
        flag_list[-1] = 1

    validity_list = [1] * len(fitted_bev_line)
    last_valid_index = flag_list.index(1)
    for i in range(last_valid_index + 1, len(validity_list)):
        validity_list[i] = 0
    
    return fitted_bev_line, flag_list, validity_list


