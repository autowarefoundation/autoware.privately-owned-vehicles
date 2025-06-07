#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import math
import warnings
import numpy as np
from PIL import Image, ImageDraw


anomaly_dict = {
    "one-sided-anchors" : {
        "count" : 0,
        "list" : []
    },
    "non-mid-egopath" : {
        "count" : 0,
        "list" : {}
    },
    "too-short-egopath" : {
        "count" : 0,
        "list" : {}
    },
    "too-steep-egopath" : {
        "count" : 0,
        "list" : {}
    },
}


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


def add_anomaly(frame_id, anomaly_code, drivable_path = None):
    anomaly_dict[anomaly_code]["count"] += 1
    if (drivable_path):
        anomaly_dict[anomaly_code]["list"][frame_id] = drivable_path
    else:
        anomaly_dict[anomaly_code]["list"].append(frame_id)


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line = None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """
    return [(x / width, y / height) for x, y in line]


def interpLine(line: list, points_quota: int):
    """
    Interpolates a line of (x, y) points to have at least `point_quota` points.
    This helps with CurveLanes since most of its lines have so few points, 2~3.
    """
    if len(line) >= points_quota:
        return line

    # Extract x, y separately then parse to interp
    x = np.array([pt[0] for pt in line])
    y = np.array([pt[1] for pt in line])
    interp_x = np.interp
    interp_y = np.interp

    # Here I try to interp more points along the line, based on
    # distance between each subsequent original points. 

    # 1) Use distance along line as param (t)
    # This is Euclidian distance between each point and the one before it
    distances = np.cumsum(np.sqrt(
        np.diff(x, prepend = x[0])**2 + \
        np.diff(y, prepend = y[0])**2
    ))
    # Force first t as zero
    distances[0] = 0

    # 2) Generate new t evenly spaced along original line
    evenly_t = np.linspace(distances[0], distances[-1], points_quota)

    # 3) Interp x, y coordinates based on evenly t
    x_new = interp_x(evenly_t, distances, x)
    y_new = interp_y(evenly_t, distances, y)

    return list(zip(x_new, y_new))


def getLineAnchor(line, new_img_height):
    """
    Determine "anchor" point of a line.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[1]

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


def getEgoIndexes(frame_id, anchors, new_img_width):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of line anchors.
    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= new_img_width / 2):
            if (i == 0):
                add_anomaly(frame_id, "one-sided-anchors")
                return "NO LINES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
        
    add_anomaly(frame_id, "one-sided-anchors")
    return "NO LINES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(
        frame_id,
        left_ego, right_ego, 
        new_img_height, new_img_width, 
        y_coords_interp = False
):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.
    """
    drivable_path = []

    # When it's CurveLanes and we need interpolation among non-uniform y-coords
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
    y_top = min(left_ego[-1][1], right_ego[-1][1])
    if ((len(drivable_path) >= 2) and (drivable_path[-1][1] > y_top)):
        sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
        sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
        sign_val = sign_left_ego * sign_right_ego
        # 2 egos going the same direction
        if (sign_val > 0):
            longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
            if len(longer_ego) >= 2 and len(drivable_path) >= 2:
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
            if len(drivable_path) >= 2:
                x1, y1 = drivable_path[-1]
                x2, y2 = drivable_path[-2]
                if (x2 == x1):
                    x_top = x1
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = x1 + (y_top - y1) / a

                drivable_path.append((x_top, y_top))

    # Now check drivable path's params for automatic auditting

    drivable_path_angle_deg = math.degrees(math.atan(float(
        abs(drivable_path[-1][1] - drivable_path[0][1]) / \
        abs(drivable_path[-1][0] - drivable_path[0][0])
    )))

    if not (new_img_width * LEFT_ANCHOR_BOUNDARY <= drivable_path[0][0] <= new_img_width * (1 - RIGHT_ANCHOR_BOUNDARY)):
        add_anomaly(frame_id, "non-mid-egopath", drivable_path)
        return f"Drivable path has anchor point out of heuristic boundary [{LEFT_ANCHOR_BOUNDARY} , {1 - RIGHT_ANCHOR_BOUNDARY}], ignoring this frame!"
    elif not (drivable_path[-1][1] < new_img_height * (1 - HEIGHT_BOUNDARY)):
        add_anomaly(frame_id, "too-short-egopath", drivable_path)
        return f"Drivable path has length not exceeding heuristic length of {HEIGHT_BOUNDARY * 100}% frame height, ignoring this frame!"
    elif not (drivable_path_angle_deg >= ANGLE_BOUNDARY):
        add_anomaly(frame_id, "too-steep-egopath", drivable_path)
        return f"Drivable path has angle not exceeding heuristic angle of {ANGLE_BOUNDARY} degrees, ignoring this frame!"

    return drivable_path


def annotateGT(
        raw_img, anno_entry,
        raw_dir, visualization_dir,
        init_img_width, init_img_height,
        normalized = True,
        resize = None,
        crop = None,
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Define save name
    # Also save in PNG (EXTREMELY SLOW compared to jpg, for lossless quality)
    save_name = str(img_id_counter).zfill(6) + ".png"

    # Load img
    raw_img = raw_img
    new_img_height = init_img_height
    new_img_width = init_img_width
    
    # Handle image resizing
    if (resize):
        new_img_height = int(new_img_height * resize)
        new_img_width = int(new_img_width * resize)
        raw_img = raw_img.resize((
            new_img_width, 
            new_img_height
        ))

    # Handle image cropping
    if (crop):
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]
        raw_img = raw_img.crop((
            CROP_LEFT, 
            CROP_TOP, 
            new_img_width - CROP_RIGHT, 
            new_img_height - CROP_BOTTOM
        ))
        new_img_height -= (CROP_TOP + CROP_BOTTOM)
        new_img_width -= (CROP_LEFT + CROP_RIGHT)

    # Copy raw img and put it in raw dir.
    raw_img.save(os.path.join(raw_dir, save_name))

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
                (x * new_img_width, y * new_img_height) 
                for x, y in line
            ]
        if (idx in anno_entry["ego_indexes"]):
            # Ego lanes, in green
            draw.line(line, fill = lane_colors["ego_green"], width = lane_w)
        else:
            # Outer lanes, in red
            draw.line(line, fill = lane_colors["outer_red"], width = lane_w)
    # Drivable path, in yellow
    if (normalized):
        drivable_renormed = [
            (x * new_img_width, y * new_img_height) 
            for x, y in anno_entry["drivable_path"]
        ]
    else:
        drivable_renormed = anno_entry["drivable_path"]
    draw.line(drivable_renormed, fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Save visualization img, same format with raw, just different dir, and jpg since it's vis 
    raw_img.save(os.path.join(visualization_dir, save_name.replace("png", "jpg")))


def parseAnnotations(
        frame_id,
        anno_path, 
        init_img_width,
        init_img_height,
        crop = None,
        resize = None,
    ):
    """
    Parses line annotations from raw img + anno files, then extracts normalized GT data.
    """
    # Read each line of GT text file as JSON
    with open(anno_path, "r") as f:
        read_data = json.load(f)["Lines"]
        if (len(read_data) < 2):    # Some files are empty, or having less than 2 lines
            warnings.warn(f"Parsing {anno_path} : insufficient line amount: {len(read_data)}")
            return None
        else:
            # Parse data from those JSON lines
            lines = [
                [(float(point["x"]), float(point["y"])) for point in line]
                for line in read_data
            ]

            # Interpolate each line in case it has too few points
            for i in range(len(lines)):
                if (len(lines[i]) < LINE_INTERP_THRESHOLD):
                    lines[i] = interpLine(lines[i], LINE_INTERP_THRESHOLD)

            new_img_height = init_img_height
            new_img_width = init_img_width

            # Handle image resizing
            if (resize):
                new_img_height = int(new_img_height * resize)
                new_img_width = int(new_img_width * resize)
                lines = [[
                    (x * resize, y * resize) 
                    for (x, y) in line
                ] for line in lines]

            # Handle image cropping
            if (crop):
                CROP_TOP = crop["TOP"]
                CROP_RIGHT = crop["RIGHT"]
                CROP_BOTTOM = crop["BOTTOM"]
                CROP_LEFT = crop["LEFT"]
                # Crop
                lines = [[
                    (x - CROP_LEFT, y - CROP_TOP) for x, y in line
                    if (
                        (CROP_LEFT <= x <= (new_img_width - CROP_RIGHT)) and 
                        (CROP_TOP <= y <= (new_img_height - CROP_BOTTOM))
                    )
                ] for line in lines]
                new_img_height -= (CROP_TOP + CROP_BOTTOM)
                new_img_width -= (CROP_LEFT + CROP_RIGHT)
            
            # Remove empty lanes
            lines = [line for line in lines if (line and len(line) >= 2)]   # Pick lines with >= 2 points
            if (len(lines) < 2):    # Ignore frames with less than 2 lines
                warnings.warn(f"Parsing {anno_path}: insufficient line amount after cropping: {len(lines)}")
                return None
            
            # Determine 2 ego lines via line anchors

            # First get the anchors
            # Here I attach index i before each anchor to support retracking after sort by x-coord)
            line_anchors = sorted(
                [(i, getLineAnchor(line, new_img_height)) for i, line in enumerate(lines)],
                key = lambda x : x[1][0],
                reverse = False
            )
                
            # Sort the lines by order of their corresponding anchors (which is also sorted)
            lines_sortedBy_anchor = [
                lines[anchor[0]]
                for anchor in line_anchors
            ]

            ego_indexes = getEgoIndexes(
                frame_id,
                [anchor[1] for anchor in line_anchors],
                new_img_width
            )

            if (type(ego_indexes) is str):
                if (ego_indexes.startswith("NO")):
                    warnings.warn(f"Parsing {anno_path}: {ego_indexes}")
                return None

            left_ego = lines_sortedBy_anchor[ego_indexes[0]]
            right_ego = lines_sortedBy_anchor[ego_indexes[1]]

            # Determine drivable path from 2 egos, and switch on interp cuz this is CurveLanes
            drivable_path = getDrivablePath(
                frame_id,
                left_ego, right_ego,
                new_img_height, new_img_width,
                y_coords_interp = True
            )

            if (type(drivable_path) is str):
                warnings.warn(f"Parsing {anno_path}: {drivable_path}")
            else:
                # Parse processed data, all coords normalized
                anno_data = {
                    "lanes" : [
                        normalizeCoords(line, new_img_width, new_img_height) 
                        for line in lines_sortedBy_anchor
                    ],
                    "ego_indexes" : ego_indexes,
                    "drivable_path" : normalizeCoords(drivable_path, new_img_width, new_img_height),
                    "egoleft_lane" : normalizeCoords(left_ego, new_img_width, new_img_height),
                    "egoright_lane" : normalizeCoords(right_ego, new_img_width, new_img_height),
                    "img_width" : new_img_width,
                    "img_height" : new_img_height,
                }

                return anno_data


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    ROOT_DIR = "Curvelanes"
    LIST_SPLITS = ["train", "valid"]
    IMG_DIR = "images"
    LABEL_DIR = "labels"

    # I got this result from `./EDA_imgsizes.ipynb`
    SIZE_DICT = {
        "beeg" : (2560, 1440),
        "half_beeg" : (1280, 720),
        "weird" : (1570, 660),
    }

    BEEG_Y_CROP_SUM = 320
    BEEG_X_CROP = 240
    BEEG_Y_CROP_TOP_RATIO = 0.75
    CROP_BEEG = {
        "TOP" : int(BEEG_Y_CROP_SUM * BEEG_Y_CROP_TOP_RATIO),
        "RIGHT" : BEEG_X_CROP,
        "BOTTOM" : int(BEEG_Y_CROP_SUM * (1 - BEEG_Y_CROP_TOP_RATIO)),
        "LEFT" : BEEG_X_CROP
    }
    
    WEIRD_Y_CROP = 130
    WEIRD_X_CROP = 385
    CROP_WEIRD = {
        "TOP" : WEIRD_Y_CROP,
        "RIGHT" : WEIRD_X_CROP,
        "BOTTOM" : WEIRD_Y_CROP,
        "LEFT" : WEIRD_X_CROP
    }

    # For interping lines with soooooooo few points, 2~3 or so
    LINE_INTERP_THRESHOLD = 5

    # ====== Heuristic boundaries of drivable path for automatic auditing ====== #

    LEFT_ANCHOR_BOUNDARY = RIGHT_ANCHOR_BOUNDARY = 0.25
    HEIGHT_BOUNDARY = 0.15
    ANGLE_BOUNDARY = 30

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process CurveLanes dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "CurveLanes directory (should contain exactly `Curvelanes` if you get it from Kaggle)",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory",
        required = True
    )
    parser.add_argument(
        "--sampling_step",
        type = int,
        help = "Sampling step for each split/class",
        required = False,
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. files each split/class you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Parse sampling step
    if (args.sampling_step):
        sampling_step = args.sampling_step
    else:
        sampling_step = 1
    print(f"Sampling step set to {sampling_step}.")

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, each split/class stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    --output_dir
        |----image
        |----visualization
        |----drivable_path.json
    """
    list_subdirs = [
        "image", 
        # "segmentation",   # Removed on 2025/06/05 
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

    # Parse data by batch
    data_master = {}
    img_id_counter = -1

    for split in LIST_SPLITS:
        print(f"\n==================== Processing {split} data ====================\n")
        raw_img_book = os.path.join(dataset_dir, ROOT_DIR, split, f"{split}.txt")
        with open(raw_img_book, "r") as f:
            list_raw_files = f.readlines()

            for i in range(0, len(list_raw_files), sampling_step):
                img_path = os.path.join(dataset_dir, ROOT_DIR, split, list_raw_files[i]).strip()
                img_id_counter += 1

                # Preload image file for multiple uses later
                raw_img = Image.open(img_path).convert("RGB")
                img_width, img_height = raw_img.size

                init_img_size = raw_img.size

                resize = None
                crop = None

                if (init_img_size == SIZE_DICT["beeg"]):
                    resize = 0.5
                    crop = CROP_BEEG
                elif (init_img_size == SIZE_DICT["half_beeg"]):
                    resize = None
                    crop = CROP_BEEG
                elif (init_img_size == SIZE_DICT["weird"]):
                    resize = None
                    crop = CROP_WEIRD

                anno_path = img_path.replace(".jpg", ".lines.json").replace(IMG_DIR, LABEL_DIR)

                this_data = parseAnnotations(
                    frame_id = os.path.abspath(img_path),
                    anno_path = anno_path,
                    init_img_width = img_width,
                    init_img_height = img_height,
                    resize = resize,
                    crop = crop
                )
                if (this_data is not None):

                    annotateGT(
                        raw_img = raw_img,
                        anno_entry = this_data,
                        raw_dir = os.path.join(output_dir, "image"),
                        visualization_dir = os.path.join(output_dir, "visualization"),
                        init_img_height = img_height,
                        init_img_width = img_width,
                        resize = resize,
                        crop = crop
                    )

                    # Save as 6-digit incremental index
                    img_index = str(str(img_id_counter).zfill(6))
                    data_master[img_index] = {}
                    data_master[img_index]["drivable_path"] = round_line_floats(this_data["drivable_path"])
                    data_master[img_index]["egoleft_lane"] = round_line_floats(this_data["egoleft_lane"])
                    data_master[img_index]["egoright_lane"] = round_line_floats(this_data["egoright_lane"])
                    data_master[img_index]["img_height"] = this_data["img_height"]
                    data_master[img_index]["img_width"] = this_data["img_width"]

                    # Early stopping, it defined
                    if (early_stopping and img_id_counter >= early_stopping - 1):
                        break

    # Save master data
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent = 4)

    # Save discarded data
    with open(os.path.join(output_dir, "anomaly.json"), "w") as f:
        json.dump(anomaly_dict, f, indent = 4)