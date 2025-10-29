#! /usr/bin/env python3

import argparse
import json
import os
import re
import ast
import cv2
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


# Sort key to sort by text and num parts separately
def _natural_keys(text: str):
    """Split text into list of ints and lowercase text for natural sorting."""
    return [
        int(tok) 
        if tok.isdigit() 
        else tok.lower() 
        for tok in re.split(r'(\d+)', text)
    ]


# ============================== Helper functions ============================== #


def getLineAnchor(
    line: Line,
    verbose: bool = False
):
    """
    Determine "anchor" point of a line.
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
    gt_filepath: str,
    frame_idx: int,
    verbose: bool = False
) -> dict[str, Any] | None:

    # Parse GT file
    with open(gt_filepath, "r") as f:
        lines = f.readlines()
        if (not lines):
            if (verbose):
                print(f"GT file {gt_filepath} is empty, skipping frame {frame_idx}.")
            return None
    
    lane_lines = []
    for line in lines:

        line = line.strip()
        if (not line):                      # Deal with empty line case, like 0253/2779.txt
            continue

        line = line.split(":")[1].strip()   # Get only the coords part

        line = line.replace(
            ")(", 
            ")|("
        )                                   # My lil trick to separate points properly
        line = [
            ast.literal_eval(pt_str)
            for pt_str in line.split("|")   # Do you see the beauty of it?
        ]
        if (line):
            line = sorted(
                line,
                key = lambda pt: pt[1],     # Sort by y coords
                reverse = True
            )

        # Sanity checks
        if (
            (line and len(line) < 2) or
            (not line)
        ):
            if (verbose):
                print(f"Line with less than 2 points found in frame {frame_idx}, skipping this line.")
            continue

        lane_lines.append(line)

    if (len(lane_lines) < 2):
        if (verbose):
            print(f"Frame {frame_idx} has less than 2 lane lines, skipping frame.")
        return None
    
    # Determining egolines via anchors

    line_anchors = [
        getLineAnchor(
            line,
            verbose = verbose
        )
        for line in lane_lines
    ]

    for i, anchor in enumerate(line_anchors):
        if (anchor[0] >= W / 2):
            if (i == 0):
                egoleft_lane = lane_lines[0]
                egoright_lane = lane_lines[1]
                other_lanes = [
                    line for j, line in enumerate(lane_lines) 
                    if j != 0 and j != 1
                ]
            else:
                egoleft_lane = lane_lines[i - 1]
                egoright_lane = lane_lines[i]
                other_lanes = [
                    line for j, line in enumerate(lane_lines) 
                    if j != i - 1 and j != i
                ]
            break
        else:
            # Traversed all lanes but none is on the right half
            if (i == len(lane_lines) - 1):
                egoleft_lane = None
                egoright_lane = None

    # Skip frames with no sufficient egolines
    if (not egoleft_lane) or (not egoright_lane):
        if (verbose):
            print(f"Frame {frame_idx} has no egolines, skipping frame.")
        frame_idx += 1
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

    # Saving processed GTs
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

    # FYI: https://github.com/vonsj0210/Multi-Lane-Detection-Dataset-with-Ground-Truth

    # The downloaded dataset is structured as follows (if you do it the right way):
    # Jiqing_Dataset/
    # ├── Jiqing Expressway Video/
    # │   ├── IMG_0249.MOV
    # │   ├── IMG_0250.MOV
    # │   └── ...
    # └── Lane_Parameters/
    #     ├── 0249/
    #     │   ├── 1.txt
    #     │   ├── 2.txt
    #     │   └── ...
    #     ├── 0250/
    #     │   ├── 1.txt
    #     │   ├── 2.txt
    #     │   └── ...
    #     └── ...

    # All scenes have reso 1920 x 1080 at 30 FPS
    W = 1920
    H = 1080

    # ================================ Parsing args ================================ #

    parser = argparse.ArgumentParser(
        description = "Process Jinan-Qingdao Expressway Dataset - groundtruth generation"
    )
    parser.add_argument(
        "--video_dir", 
        "-iv",
        type = str, 
        help = "Jinan-Qingdao Expressway video directory (should contain .MOV files)",
        required = True
    )
    parser.add_argument(
        "--gt_dir", 
        "-igt",
        type = str, 
        help = "Jinan-Qingdao Expressway ground truth directory (should contain subfolders with same names as video files)",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        "-o",
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
    parser.add_argument(
        "--verbose",
        "-v",
        type = bool,
        help = "Verbose output.",
        required = False
    )

    args = parser.parse_args()

    # Parse dirs
    video_dir = args.video_dir
    gt_dir = args.gt_dir
    output_dir = args.output_dir

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stopping after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Parse verbose
    if (args.verbose):
        verbose = args.verbose
    else:
        verbose = False
    
    # Generate output structure

    list_subdirs = [
        "image",
        "mask",
        "visualization",
    ]

    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    data_master = {}
    img_id_counter = 0
    flag_continue = True

    list_videos  = sorted(
        os.listdir(video_dir),
        key = _natural_keys
    )
    list_gt_dirs = sorted(
        os.listdir(gt_dir),
        key = _natural_keys
    )

    assert len(list_videos) == len(list_gt_dirs), \
        f"Number of video files ({len(list_videos)}) and GT folders ({len(list_gt_dirs)}) do not match."
    
    # Main loop

    for i in tqdm(
        range(len(list_videos)) 
        if (early_stopping is None) 
        else range(min(early_stopping, len(list_videos))),
        desc = "Processing videos: ",
        unit = "video",
        colour = "green"
    ):

        # Early stopping check on outer loop
        if (not flag_continue):
            break
        
        this_video_name = list_videos[i]
        this_gt_dir = list_gt_dirs[i]

        assert this_gt_dir in this_video_name, \
            f"Video file ({this_video_name}) and GT folder ({this_gt_dir}) do not match."

        # Prepare raw GTs
        gt_files = sorted(
            os.listdir(
                os.path.join(
                    gt_dir, 
                    this_gt_dir
                )
            ),
            key = _natural_keys
        )
        
        # Read video frame-by-frame at 30 FPS
        video_path = os.path.join(
            video_dir, 
            this_video_name
        )
        video_name = os.path.basename(video_path).split(".")[0]
        cap = cv2.VideoCapture(video_path)

        # Go frame-by-frame
        # frame_idx = 0
        # while (
        #     (cap.isOpened()) and 
        #     (frame_idx < len(gt_files))
        # ):
        for frame_idx in tqdm(
            range(len(gt_files)),
            desc = f"Processing frames: ",
            unit = "frame",
            colour = "yellow",
        ):
            
            ret, frame = cap.read()
            if (not ret):
                if (verbose):
                    print(f"Frame {frame_idx} could not be read, stopping.")
                break

            # Corresponding GT file
            gt_file = gt_files[frame_idx]
            gt_filepath = os.path.join(
                gt_dir,
                this_gt_dir, 
                gt_file
            )
        
            # Parse raw data
            anno_entry = parseData(
                gt_filepath = gt_filepath,
                frame_idx = frame_idx,
                verbose = False
            )

            # Annotate GT
            if (anno_entry is not None):
                annotateGT(
                    raw_img = frame,
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

        cap.release()
        if (verbose):
            print(f"Finished processing video {video_name}.")


    # Save master annotation file
    with open(
        os.path.join(output_dir, "drivable_path.json"), 
        "w"
    ) as f:
        json.dump(
            data_master, f, 
            indent = 4
        )
    
    print(f"Completed processing Jiqing dataset.")
    print(f"Total samples: {img_id_counter}.")
    print(f"Output saved to {output_dir}.")