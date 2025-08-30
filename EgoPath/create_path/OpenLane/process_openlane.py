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


PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]

def round_line_floats(
    line: list[PointCoords] | list[ImagePointCoords], 
    ndigits: int = 6
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


# Custom warning format
def custom_warning_format(
    message, 
    category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def normalizeCoords(
    line: list[PointCoords] | list[ImagePointCoords], 
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


def getLineAnchor(
    line: list[PointCoords] | list[ImagePointCoords], 
    new_img_height: int
):
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


def getDrivablePath(
    left_ego: list[PointCoords] | list[ImagePointCoords], 
    right_ego: list[PointCoords] | list[ImagePointCoords], 
    new_img_height: int,
    y_coords_interp: bool = False
):
    """
    Computes drivable path as midpoint between 2 ego lanes.
    """

    drivable_path = []

    # Interpolation among non-uniform y-coords
    if (y_coords_interp):

        left_ego = np.array(left_ego)
        right_ego = np.array(right_ego)
        y_coords_ASSEMBLE = np.unique(
            np.concatenate((
                left_ego[:, 1],
                right_ego[:, 1]
            ))
        )[::-1]
        left_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            left_ego[:, 1][::-1], 
            left_ego[:, 0][::-1]
        )
        right_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            right_ego[:, 1][::-1], 
            right_ego[:, 0][::-1]
        )
        mid_x = (left_x_interp + right_x_interp) / 2
        # Filter out those points that are not in the common vertical zone between 2 egos
        drivable_path = [
            [x, y] for x, y in list(zip(mid_x, y_coords_ASSEMBLE))
            if y <= min(left_ego[0][1], right_ego[0][1])
        ]

    else:
        # Get the normal drivable path from the longest common y-coords
        while (i <= len(left_ego) - 1 and j <= len(right_ego) - 1):
            if (left_ego[i][1] == right_ego[j][1]):
                drivable_path.append((
                    (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                    left_ego[i][1]
                ))
                i += 1
                j += 1
            elif (left_ego[i][1] > right_ego[j][1]):
                i += 1
            else:
                j += 1

    # Extend drivable path to bottom edge of the frame
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < new_img_height - 1)):
        x1, y1 = drivable_path[1]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (new_img_height - 1 - y2) / a
        drivable_path.insert(0, (x_bottom, new_img_height - 1))

    # Extend drivable path to be on par with longest ego line
    # By making it parallel with longer ego line
    y_top = min(
        left_ego[-1][1], 
        right_ego[-1][1]
    )

    if (
        (len(drivable_path) >= 2) and 
        (drivable_path[-1][1] > y_top)
    ):
        sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
        sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
        sign_val = sign_left_ego * sign_right_ego

        # 2 egos going the same direction
        if (sign_val > 0):
            longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
            if (
                (len(longer_ego) >= 2) and 
                (len(drivable_path) >= 2)
            ):
                x1, y1 = longer_ego[-1]
                x2, y2 = longer_ego[-2]
                if (x2 == x1):
                    x_top = drivable_path[-1][0]
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = drivable_path[-1][0] + (y_top - drivable_path[-1][1]) / a

                drivable_path.append((x_top, y_top))
        
        # 2 egos going opposite directions
        else:
            if (len(drivable_path) >= 2):
                x1, y1 = drivable_path[-1]
                x2, y2 = drivable_path[-2]

                if (x2 == x1):
                    x_top = x1
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = x1 + (y_top - y1) / a

                drivable_path.append((x_top, y_top))


    return drivable_path


# ============================== Core functions ============================== #


def annotateGT(
    raw_img: Image,
    anno_entry: dict,
    visualization_dir: str,
    normalized: bool = True,
):
    """
    Annotates and saves an image with:
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Define save name
    # Also save in PNG (EXTREMELY SLOW compared to jpg, for lossless quality)
    save_name = str(img_id_counter).zfill(6) + ".jpg"

    # Draw all lanes & lines
    draw = ImageDraw.Draw(raw_img)
    lane_colors = {
        "outer_red": (255, 0, 0), 
        "ego_green": (0, 255, 0), 
        "drive_path_yellow": (255, 255, 0)
    }
    lane_w = 5
    # Draw lanes
    for idx, line in enumerate(anno_entry["lanes"]):
        if (normalized):
            line = [
                (x * W, y * H) 
                for x, y in line
            ]
        if (idx in anno_entry["ego_indexes"]):
            # Ego lanes, in green
            draw.line(
                line, 
                fill = lane_colors["ego_green"], 
                width = lane_w
            )
        else:
            # Outer lanes, in red
            draw.line(
                line, 
                fill = lane_colors["outer_red"], 
                width = lane_w
            )
    # Drivable path, in yellow
    if (normalized):
        drivable_renormed = [
            (x * W, y * H) 
            for x, y in anno_entry["drivable_path"]
        ]
    else:
        drivable_renormed = anno_entry["drivable_path"]
    draw.line(
        drivable_renormed, 
        fill = lane_colors["drive_path_yellow"], 
        width = lane_w
    )

    # Save visualization img
    raw_img.save(os.path.join(visualization_dir, save_name))


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