#! /usr/bin/env python3

import argparse
import json
import os
import cv2
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


def calcLaneSegMask(
    lanes, 
    width, height,
    normalized: bool = True
):
    """
    Calculates binary segmentation mask for some lane lines.
    """

    # Create blank mask as new Image
    bin_seg = np.zeros(
        (height, width), 
        dtype = np.uint8
    )
    bin_seg_img = Image.fromarray(bin_seg)

    # Draw lines on mask
    draw = ImageDraw.Draw(bin_seg_img)
    for lane in lanes:
        if (normalized):
            lane = [
                (
                    x * width, 
                    y * height
                ) 
                for x, y in lane
            ]
        draw.line(
            lane, 
            fill = 255, 
            width = 4
        )

    # Convert back to numpy array
    bin_seg = np.array(
        bin_seg_img, 
        dtype = np.uint8
    )

    return bin_seg


# ============================== Core functions ============================== #


def parseData(
    img_id      : int,
    label_data  : dict,
    verbose     : bool = False
):
    """
    Parse Once3DLane data entry.
    """

    # Read GT info
    num_lanes       = label_data["lane_num"]
    lines_3d        = label_data["lanes"]
    cam_intrinsics  = label_data["calibration"]

    if (num_lanes < 2):
        if (verbose):
            print(f"Frame ID {img_id} has less than 2 lines. Skipping.")
        return None

    # Process lanes, project to 2D
    cam_intrinsics_T = np.array(cam_intrinsics).T.tolist()
    lines_2d = []

    for line_3d in lines_3d:

        line_3d = np.array(
            line_3d,
            dtype = np.float32
        )[:, :3]

        pcl_cam_homo = np.hstack(
            [
                line_3d,
                np.ones(
                    line_3d.shape[0], 
                    dtype = np.float32
                ).reshape((-1, 1))
            ]
        )
        pcl_img = np.dot(
            pcl_cam_homo,
            cam_intrinsics_T
        )
        pcl_img = pcl_img / pcl_img[:, [2]]
        line_2d = pcl_img[:, :2].tolist()

        if (len(line_2d) < 2):
            continue

        # Sort by descending y-coords
        line_2d = sorted(
            line_2d,
            key = lambda x: x[1],
            reverse = True
        )

        # Attach anchor to line
        line_2d = [[getLineAnchor(line_2d)[0], H - 1]] + line_2d

        lines_2d.append(line_2d)

    if (len(lines_2d) < 2):
        if (verbose):
            print(f"Image ID {img_id} after processing has insufficient lines. Skipping.")
        return None
        
    # Sort all lines by ascending x-coords of anchors
    lines_2d = sorted(
        lines_2d,
        key = lambda x: x[0][0],
        reverse = False
    )

    # Determine ego lines
    for i, line in enumerate(lines_2d):
        if (line[0][0] >= W / 2):
            if (i == 0):
                egoleft_lane = lines_2d[0]
                egoright_lane = lines_2d[1]
                other_lanes = [
                    line for j, line in enumerate(lines_2d) 
                    if j != 0 and j != 1
                ]
            else:
                egoleft_lane = lines_2d[i - 1]
                egoright_lane = lines_2d[i]
                other_lanes = [
                    line for j, line in enumerate(lines_2d) 
                    if j != i - 1 and j != i
                ]
            break
        else:
            # Traversed all lines but none is on the right half
            if (i == len(lines_2d) - 1):
                egoleft_lane = None
                egoright_lane = None

    # Skip if egolines not found
    if (not egoleft_lane) or (not egoright_lane):
        if (verbose):
            print(f"Image ID {img_id} has insufficient egolines. Skipping.")
        return None
    
    # Create segmentation masks:
    # Channel 1: egoleft lane
    # Channel 2: egoright lane
    # Channel 3: other lanes

    mask = np.zeros(
        (H, W, 3), 
        dtype = np.uint8
    )
    mask[:, :, 0] = calcLaneSegMask(
        [egoleft_lane], 
        W, H,
        normalized = False
    )
    mask[:, :, 1] = calcLaneSegMask(
        [egoright_lane], 
        W, H,
        normalized = False
    )
    mask[:, :, 2] = calcLaneSegMask(
        other_lanes, 
        W, H,
        normalized = False
    )

    # Final anno entry log
    anno_entry = {
        "other_lanes"     : other_lanes,
        "egoleft_lane"    : egoleft_lane,
        "egoright_lane"   : egoright_lane,
        "mask"            : mask,
    }

    return anno_entry


def annotateGT(
    raw_img: np.ndarray,
    anno_entry: dict,
    img_dir: str,
    mask_dir: str,
    visualization_dir: str
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
        - Segmentation mask image, in "output_dir/mask".
    """

    # Define save name, now saving everything in JPG
    # to preserve my remaining disk space
    save_name = str(img_id_counter).zfill(6)

    # Raw img, fetched from cv2 BGR format, convert to RGB
    raw_img = Image.fromarray(
        cv2.cvtColor(
            raw_img, 
            cv2.COLOR_BGR2RGB
        )
    )
    raw_img.save(os.path.join(img_dir, save_name + ".jpg"))

    # Fetch seg mask and save as RGB PNG
    mask_img = Image.fromarray(anno_entry["mask"]).convert("RGB")
    mask_img.save(os.path.join(mask_dir, save_name + ".png"))

    # Overlay mask on raw image, ratio 1:1
    overlayed_img = Image.blend(
        raw_img, 
        mask_img, 
        alpha = 0.5
    )

    # Save visualization img, JPG for lighter weight, just different dir
    overlayed_img.save(os.path.join(visualization_dir, save_name + ".jpg"))


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
    flag_continue = True

    for segment_id in tqdm(
        sorted(os.listdir(IMG_DIR)), 
        desc = "Processing segments: ",
        colour = "yellow"
    ):
        
        # Early stopping check on outer loop
        if (not flag_continue):
            break
        
        segment_img_dir     = os.path.join(IMG_DIR, segment_id, CAM_DIR)
        segment_label_dir   = os.path.join(LABEL_DIR, segment_id, CAM_DIR)
        segment_info_path   = os.path.join(
            INFO_DIR, 
            segment_id, 
            f"{segment_id}.json"
        )

        list_current_segment_imgs   = sorted(os.listdir(segment_img_dir))
        list_current_segment_labels = sorted(os.listdir(segment_label_dir))
        assert len(list_current_segment_imgs) == len(list_current_segment_labels) * 2, \
            f"Number of images and labels do not match in segment {segment_id}!"

        # Process frame-by-frame
        for i in range(len(list_current_segment_labels)):

            # Early stopping
            if (
                (early_stopping) and 
                (img_id_counter == early_stopping - 1)
            ):
                break

            img_id_counter += 1
            img_filename    = list_current_segment_imgs[i * 2]  # Every 2 images share the same label. Tricky ain't it?
            label_filename  = list_current_segment_labels[i]
            img_path        = os.path.join(segment_img_dir, img_filename)
            label_path      = os.path.join(segment_label_dir, label_filename)

            # Read data
            img = cv2.imread(img_path)
            with open(label_path, "r") as f:
                label_data = json.load(f)

            # Parse GTs
            anno_entry = parseData(
                img_id_counter,
                label_data
            )

            # Annotate GT
            if (anno_entry is not None):
                annotateGT(
                    raw_img = img,
                    anno_entry = anno_entry,
                    img_dir = os.path.join(
                        output_dir, 
                        "image"
                    ),
                    mask_dir = os.path.join(
                        output_dir, 
                        "mask"
                    ),
                    visualization_dir = os.path.join(
                        output_dir, 
                        "visualization"
                    )
                )

            # Log to master data
            img_index = str(str(img_id_counter).zfill(6))
            data_master[img_index] = {
                "img_path"      : os.path.join(
                    output_dir, 
                    "image",
                    img_index + ".jpg"
                ),
                "egoleft_lane"  : round_line_floats(
                    normalizeCoords(
                        anno_entry["egoleft_lane"],
                        W, H
                    )
                ),
                "egoright_lane" : round_line_floats(
                    normalizeCoords(
                        anno_entry["egoright_lane"],
                        W, H
                    )
                ),
                "other_lanes"   : [
                    round_line_floats(
                        normalizeCoords(
                            lane,
                            W, H
                        )
                    )
                    for lane in anno_entry["other_lanes"]
                ],
            }

            img_id_counter += 1

            # Early stopping check on inner loop
            if (
                early_stopping and 
                (img_id_counter >= early_stopping)
            ):
                flag_continue = False
                print(f"Early stopping reached at {early_stopping} samples.")
                break

    # Save master annotation file
    with open(
        os.path.join(output_dir, "drivable_path.json"), 
        "w"
    ) as f:
        json.dump(
            data_master, f, 
            indent = 4
        )

    print(f"Completed processing Once3DLane dataset.")
    print(f"Total samples: {img_id_counter}.")
    print(f"Output saved to {output_dir}.")