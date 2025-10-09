#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from typing import Any
from PIL import Image, ImageDraw


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


# Custom warning format
def custom_warning_format(
    message, 
    category, filename, 
    lineno, line = None
):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# Log skipped images
def log_skipped_image(
    log_json: dict,
    reason: str,
    image_path: str
):
    if (reason not in log_json):
        log_json[reason] = []
    log_json[reason].append(image_path)


# Annotate skipped images
def annotate_skipped_image(
    image: Image,
    reason: str,
    save_path: str
):
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, 10), 
        reason, 
        fill = (255, 0, 0)
    )
    image.save(save_path)


# ============================== Helper functions ============================== #


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


def getLineAnchor(
    line: Line,
    verbose: bool = False
):
    """
    Determine "anchor" point of a line.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[
        int(len(line) / 5) 
        if (
            len(line) > 2 and
            line[0][1] >= H * 4/5
        ) else -1
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


def getDrivablePath(
    left_ego        : Line, 
    right_ego       : Line, 
    y_coords_interp : bool = False
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
            (x, y) for x, y in list(zip(mid_x, y_coords_ASSEMBLE))
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
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < H - 1)):
        x1, y1 = drivable_path[
            int(len(drivable_path) / 5)
            if (
                len(drivable_path) > 2 and
                drivable_path[0][1] >= H * 4/5
            ) else -1
        ]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (H - 1 - y2) / a
        drivable_path.insert(0, (x_bottom, H - 1))

    # Drivable path only extends to the shortest ego line
    drivable_path = [
        (x, y) for x, y in drivable_path
        if y >= max(left_ego[-1][1], right_ego[-1][1])
    ]

    # Extend drivable path to be on par with longest ego line
    # # By making it parallel with longer ego line
    # y_top = min(
    #     left_ego[-1][1], 
    #     right_ego[-1][1]
    # )

    # if (
    #     (len(drivable_path) >= 2) and 
    #     (drivable_path[-1][1] > y_top)
    # ):
    #     sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
    #     sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
    #     sign_val = sign_left_ego * sign_right_ego

    #     # 2 egos going the same direction
    #     if (sign_val > 0):
    #         longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
    #         if (
    #             (len(longer_ego) >= 2) and 
    #             (len(drivable_path) >= 2)
    #         ):
    #             x1, y1 = longer_ego[-1]
    #             x2, y2 = longer_ego[-2]
    #             if (x2 == x1):
    #                 x_top = drivable_path[-1][0]
    #             else:
    #                 a = (y2 - y1) / (x2 - x1)
    #                 x_top = drivable_path[-1][0] + (y_top - drivable_path[-1][1]) / a

    #             drivable_path.append((x_top, y_top))
        
    #     # 2 egos going opposite directions
    #     else:
    #         if (len(drivable_path) >= 2):
    #             x1, y1 = drivable_path[-1]
    #             x2, y2 = drivable_path[-2]

    #             if (x2 == x1):
    #                 x_top = x1
    #             else:
    #                 a = (y2 - y1) / (x2 - x1)
    #                 x_top = x1 + (y_top - y1) / a

    #             drivable_path.append((x_top, y_top))

    return drivable_path


# ============================== Core functions ============================== #


def parseData(
    json_data: dict[str: Any],
    lane_point_threshold: int = 20,
    verbose: bool = False
):
    """
    Parse raw JSON data from OpenLane dataset, then return a dict with:
        - "img_path"        : str, path to the image file.
        - "other_lanes"     : list of lanes [ (xi, yi) ] that are NOT ego lanes.
        - "egoleft_lane"    : egoleft lane in [ (xi, yi) ].
        - "egoright_lane"   : egoright lane in [ (xi, yi) ].
        - "drivable_path"   : drivable path in [ (xi, yi) ].

    Since each line in OpenLane has too many points, I implement `lane_point_threshold` 
    to determine approximately the maximum number of points allowed per lane.

    All coords are rounded to 2 decimal places (honestly we won't need more than that).
    All coords are NOT NORMALIZED (will do it right before saving to JSON).
    """

    img_path = json_data["file_path"]
    lane_lines = json_data["lane_lines"]
    egoleft_lane = None
    egoright_lane = None
    other_lanes = []

    # Walk thru each lane
    for i, lane in enumerate(lane_lines):

        if not len(lane["uv"][0]) == len(lane["uv"][1]):
            if (verbose):
                warnings.warn(
                    f"Inconsistent number of U and V coords:\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - u          : {len(lane['uv'][0])}\n \
                        - v          : {len(lane['uv'][1])}"
                )
            continue

        if not (len(lane["uv"][0]) >= 10):
            if (verbose):
                warnings.warn(
                    f"Lane with insufficient points detected (less than 10 points). Ignored.\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - num_points : {len(lane['uv'][0])}"
                )
            continue

        # There are adjacent points with the same y-coords. Filtering em out.
        raw_lane = sorted(
            [
                (
                    int(lane["uv"][0][j]), 
                    int(lane["uv"][1][j])
                )
                for j in range(
                    0, 
                    len(lane["uv"][0]), 
                    (
                        1 if (len(lane['uv'][0]) < lane_point_threshold) 
                        else len(lane['uv'][0]) // lane_point_threshold
                    )
                )
            ],
            key = lambda x: x[1],
            reverse = True
        )
        this_lane = [raw_lane[0]] if raw_lane else []
        for point in raw_lane[1:]:
            if (point[1] != this_lane[-1][1]):
                this_lane.append(point)

        if (len(this_lane) < 2):
            if (verbose):
                warnings.warn(
                    f"Lane with insufficient unique y-coords detected (less than 2 points). Ignored.\n \
                        - file_path  : {img_path}\n \
                        - lane_index : {i}\n \
                        - num_points : {len(this_lane)}"
                )
            continue

        # Add anchor to line, if needed
        if (this_lane and (this_lane[0][1] < H - 1)):
            this_lane.insert(0, (
                getLineAnchor(this_lane, verbose)[0],
                H - 1
            ))

        this_attribute = lane["attribute"]

        """
        "attribute":    <int>: left-right attribute of the lane
                            1: left-left
                            2: left (exactly egoleft)
                            3: right (exactly egoright)
                            4: right-right
        """
        if (this_attribute == 2):
            if (egoleft_lane and verbose):
                warnings.warn(
                    f"Multiple egoleft lanes detected. Please check! \n\
                        - file_path: {img_path}"
                )
            else:
                egoleft_lane = this_lane
        elif (this_attribute == 3):
            if (egoright_lane and verbose):
                warnings.warn(
                    f"Multiple egoright lanes detected. Please check! \n\
                        - file_path: {img_path}"
                )
            else:
                egoright_lane = this_lane
        else:
            other_lanes.append(this_lane)

    if (egoleft_lane and egoright_lane):
        drivable_path = getDrivablePath(
            left_ego = egoleft_lane,
            right_ego = egoright_lane,
            y_coords_interp = True
        )
    else:
        if (verbose):
            warnings.warn(f"Missing egolines detected: \n\
            - file_path: {img_path}")

            if (not egoleft_lane):
                print("\t- Left egoline missing!")
                missing_line = "left"
            if (not egoright_lane):
                print("\t- Right egoline missing!")
                missing_line = "right"
        
        # Log skipped image
        reason = f"Missing egolines detected: {missing_line}"
        log_skipped_image(
            log_json = {},
            reason = reason,
            image_path = img_path
        )
        annotate_skipped_image(
            image = Image.open(img_path).convert("RGB"),
            reason = reason,
            save_path = os.path.join(skipped_path, os.path.basename(img_path))
        )

        return None
    
    # Check drivable path validity
    THRESHOLD_EGOPATH_ANCHOR = 0.25

    if (len(drivable_path) < 2):
        if (verbose):
            warnings.warn(f"Drivable path with insufficient points detected (less than 2 points). Ignored.\n \
                - file_path  : {img_path}\n \
                - num_points : {len(drivable_path)}"
            )
        return None
    
    elif not (
        THRESHOLD_EGOPATH_ANCHOR * W <= drivable_path[0][0] <= (1 - THRESHOLD_EGOPATH_ANCHOR) * W
    ):
        if (verbose):
            warnings.warn(f"Drivable path anchor too close to edge of frame. Ignored.\n \
                - file_path  : {img_path}\n \
                - anchor_x   : {drivable_path[0][0]}\n \
                - anchor_y   : {drivable_path[0][1]}"
            )
        return None
    
    elif not (
        (egoleft_lane[0][0] < drivable_path[0][0] < egoright_lane[0][0]) and
        (egoleft_lane[-1][0] < drivable_path[-1][0] < egoright_lane[-1][0])
    ):
        if (verbose):
            warnings.warn(f"Drivable path not between 2 egolanes. Ignored.\n \
                - file_path      : {img_path}\n \
                - drivable_path  : {drivable_path}\n \
                - egoleft_lane   : {egoleft_lane}\n \
                - egoright_lane  : {egoright_lane}"
            )
        return None
    
    elif not (egoright_lane[0][0] - egoleft_lane[0][0] >= egoright_lane[-1][0] - egoleft_lane[-1][0]):
        if (verbose):
            warnings.warn(f"Ego lanes are not parallel logically. Ignored.\n \
                - file_path      : {img_path}\n \
                - egoleft_lane   : {egoleft_lane}\n \
                - egoright_lane  : {egoright_lane}"
            )
        return None

    # Assemble all data
    anno_entry = {
        "img_path"        : img_path,
        "other_lanes"     : other_lanes,
        "egoleft_lane"    : egoleft_lane,
        "egoright_lane"   : egoright_lane,
        "drivable_path"   : drivable_path
    }

    return anno_entry


def annotateGT(
    anno_entry: dict,
    img_dir: str,
    visualization_dir: str
):
    """
    Annotates and saves an image with:
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Define save name, now saving everything in JPG
    # to preserve my remaining disk space
    save_name = str(img_id_counter).zfill(6) + ".jpg"

    # Prepping canvas
    raw_img = Image.open(
        os.path.join(
            img_dir, 
            anno_entry["img_path"]
        )
    ).convert("RGB")
    draw = ImageDraw.Draw(raw_img)
    
    lane_colors = {
        "outer_red": (255, 0, 0), 
        "ego_green": (0, 255, 0), 
        "drive_path_yellow": (255, 255, 0)
    }
    lane_w = 5

    # Draw other lanes, in red
    for line in anno_entry["other_lanes"]:
        draw.line(
            line, 
            fill = lane_colors["outer_red"], 
            width = lane_w
        )
    
    # Draw drivable path, in yellow
    draw.line(
        anno_entry["drivable_path"],
        fill = lane_colors["drive_path_yellow"], 
        width = lane_w
    )

    # Draw ego lanes, in green
    if (anno_entry["egoleft_lane"]):
        draw.line(
            anno_entry["egoleft_lane"],
            fill = lane_colors["ego_green"],
            width = lane_w
        )
    if (anno_entry["egoright_lane"]):
        draw.line(
            anno_entry["egoright_lane"],
            fill = lane_colors["ego_green"],
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

    # All 200k scenes have reso 1920 x 1280. I checked it manually.
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
    os.makedirs(output_dir)

    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # Logging skipped images for auditing
    log_skipped_json = {}
    skipped_path = os.path.join(output_dir, "skipped")
    if (not os.path.exists(skipped_path)):
        os.makedirs(skipped_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    data_master = {}
    img_id_counter = 0
    flag_continue = True

    for label_split, list_label_subdirs in LABEL_SPLITS.items():
        if (not flag_continue):
            break
        print(f"\nPROCESSING LABEL SPLIT : {label_split}")
        
        for subsplit in list_label_subdirs:
            if (not flag_continue):
                break
            print(f"PROCESSING SUBSPLIT : {subsplit}")

            subsplit_path = os.path.join(
                dataset_dir,
                label_split,
                subsplit
            )

            for segment in tqdm(
                sorted(os.listdir(subsplit_path)), 
                desc = "Processing segments : ",
                colour = "green"
            ):
                if (not flag_continue):
                    break
                segment_path = os.path.join(subsplit_path, segment)

                for label_file in sorted(os.listdir(segment_path)):
                    if (not flag_continue):
                        break
                                    
                    label_file_path = os.path.join(segment_path, label_file)

                    with open(label_file_path, "r") as f:
                        this_label_data = json.load(f)

                    this_label_data = parseData(
                        json_data = this_label_data,
                        verbose = True if (img_id_counter == 450) else False
                    )
                    if (this_label_data):

                        annotateGT(
                            anno_entry = this_label_data,
                            img_dir = os.path.join(
                                dataset_dir,
                                IMG_DIR
                            ),
                            visualization_dir = os.path.join(
                                output_dir, 
                                "visualization"
                            )
                        )

                        img_index = str(str(img_id_counter).zfill(6))
                        data_master[img_index] = {
                            "img_path"      : this_label_data["img_path"],
                            "egoleft_lane"  : round_line_floats(
                                normalizeCoords(
                                    this_label_data["egoleft_lane"],
                                    W, H
                                )
                            ),
                            "egoright_lane" : round_line_floats(
                                normalizeCoords(
                                    this_label_data["egoright_lane"],
                                    W, H
                                )
                            ),
                            "drivable_path" : round_line_floats(
                                normalizeCoords(
                                    this_label_data["drivable_path"],
                                    W, H
                                )
                            )
                        }

                        img_id_counter += 1

                    # Early stopping check
                    if (
                        early_stopping and 
                        (img_id_counter >= early_stopping)
                    ):
                        flag_continue = False
                        print(f"Early stopping reached at {early_stopping} samples.")
                        break

                print(f"Segment {segment} done, with {len(os.listdir(segment_path))} samples collected.")

    # Save master data
    with open(
        os.path.join(output_dir, "drivable_path.json"), 
        "w"
    ) as f:
        json.dump(
            data_master, f, 
            indent = 4
        )