#! /usr/bin/env python3

import argparse
import json
import os
from PIL import Image, ImageDraw
import warnings
import numpy as np
import cv2
import shutil
from tqdm import tqdm

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

# ============================== Helper functions ============================== #


def roundLineFloats(line, ndigits = 4):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def sortByYDesc(line):
    return sorted(
        line,
        key = lambda p: p[1],
        reverse = True
    )


def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.

    """
    return [(x / width, y / height) for x, y in lane]


def getLineAnchor(line, verbose = False):
    """
    Determine "anchor" point of a lane.

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
        if (verbose):
            warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (H - 1 - b) / a

    return (x0, a, b)



def getEgoIndexes(anchors):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of lane anchors.

    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= W / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
    
    return "NO LANES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(left_ego, right_ego):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

    """
    i, j = 0, 0
    drivable_path = []
    while (i < len(left_ego) - 1 and j < len(right_ego) - 1):
        if (left_ego[i][1] == right_ego[j][1]):
            drivable_path.append((
                (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                left_ego[i][1]
            ))
            i += 1
            j += 1
        elif (left_ego[i][1] < right_ego[j][1]):
            i += 1
        else:
            j += 1

    # Extend drivable path to bottom edge of the frame
    if (len(drivable_path) >= 2):
        x1, y1 = drivable_path[-2]
        x2, y2 = drivable_path[-1]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (H - y2) / a
        drivable_path.append((x_bottom, H))

    # Extend drivable path to be on par with longest ego lane
    # By making it parallel with longer ego lane
    y_top = min(left_ego[0][1], right_ego[0][1])
    sign_left_ego = left_ego[0][0] - left_ego[1][0]
    sign_right_ego = right_ego[0][0] - right_ego[1][0]
    sign_val = sign_left_ego * sign_right_ego
    if (sign_val > 0):  # 2 egos going the same direction
        longer_ego = left_ego if left_ego[0][1] < right_ego[0][1] else right_ego
        if len(longer_ego) >= 2 and len(drivable_path) >= 2:
            x1, y1 = longer_ego[0]
            x2, y2 = longer_ego[1]
            if (x2 == x1):
                x_top = drivable_path[0][0]
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = drivable_path[0][0] + (y_top - drivable_path[0][1]) / a

            drivable_path.insert(0, (x_top, y_top))
    else:
        # Extend drivable path to be on par with longest ego lane
        if len(drivable_path) >= 2:
            x1, y1 = drivable_path[0]
            x2, y2 = drivable_path[1]
            if (x2 == x1):
                x_top = x1
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = x1 + (y_top - y1) / a

            drivable_path.insert(0, (x_top, y_top))

    return drivable_path


def annotateGT(
    anno_entry, 
    anno_raw_file, 
    raw_dir, 
    mask_dir,
    visualization_dir,
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Lane seg mask, in "output_dir/mask".
        - Annotated image with all lanes, in "output_dir/visualization".

    """

    # Load raw image
    raw_img = Image.open(anno_raw_file).convert("RGB")

    # Define save name
    save_name = str(img_id_counter).zfill(6)

    # Copy raw img and put it in raw dir.
    shutil.copy(
        anno_raw_file,
        os.path.join(raw_dir, save_name + ".jpg")
    )

    # # Draw all lanes & lines
    # draw = ImageDraw.Draw(raw_img)
    # lane_colors = {
    #     "outer_red": (255, 0, 0), 
    #     "ego_green": (0, 255, 0), 
    #     "drive_path_yellow": (255, 255, 0)
    # }
    # lane_w = 5
    # # Draw lanes
    # for idx, lane in enumerate(anno_entry["lanes"]):
    #     if (normalized):
    #         lane = [(x * W, y * H) for x, y in lane]
    #     if (idx in anno_entry["ego_indexes"]):
    #         # Ego lanes, in green
    #         draw.line(lane, fill = lane_colors["ego_green"], width = lane_w)
    #     else:
    #         # Outer lanes, in red
    #         draw.line(lane, fill = lane_colors["outer_red"], width = lane_w)
    # # Drivable path, in yellow
    # if (normalized):
    #     drivable_renormed = [(x * W, y * H) for x, y in anno_entry["drivable_path"]]
    # else:
    #     drivable_renormed = anno_entry["drivable_path"]
    # draw.line(drivable_renormed, fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Save mask, lossless PNG for accuracy
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


def parseAnnotations(
        item: dict,
        verbose: bool = False
):
    """
    Parses lane annotations from a dict structure read from a TuSimple label JSON file.

    """
    
    lanes = item["lanes"]
    h_samples = item["h_samples"]
    raw_file = item["raw_file"]

    # Decouple from {lanes: [xi1, xi2, ...], h_samples: [y1, y2, ...]} to [(xi1, y1), (xi2, y2), ...]
    # `lane_decoupled` is a list of sublists representing lanes, each lane is a list of (x, y) tuples.
    lanes_decoupled = [
        [(x, y) for x, y in zip(lane, h_samples) if x != -2]
        for lane in lanes if sum(1 for x in lane if x != -2) >= 2     # Filter out lanes < 2 points (there's actually a bunch of em)
    ]

    # Sort each lane by decreasing y
    lanes_decoupled = [
        sortByYDesc(lane) 
        for lane in lanes_decoupled
    ]

    # Determine 2 ego lanes
    lane_anchors = [
        getLineAnchor(lane) 
        for lane in lanes_decoupled
    ]
    ego_indexes = getEgoIndexes(lane_anchors)

    if (type(ego_indexes) is str):
        if (ego_indexes.startswith("NO")):
            if (verbose):
                warnings.warn(f"Parsing {raw_file}: {ego_indexes}")
            return None

    left_ego = lanes_decoupled[ego_indexes[0]]
    right_ego = lanes_decoupled[ego_indexes[1]]
    other_lanes = [
        lane for idx, lane in enumerate(lanes_decoupled)
        if idx not in ego_indexes
    ]

    # # Determine drivable path from 2 egos
    # drivable_path = getDrivablePath(left_ego, right_ego)

    # Create segmentation masks:
    # Channel 1: egoleft lane
    # Channel 2: egoright lane
    # Channel 3: other lanes
    mask = np.zeros(
        (H, W, 3), 
        dtype = np.uint8
    )
    mask[:, :, 0] = calcLaneSegMask(
        [left_ego], 
        W, H, 
        normalized = False
    )
    mask[:, :, 1] = calcLaneSegMask(
        [right_ego], 
        W, H, 
        normalized = False
    )
    mask[:, :, 2] = calcLaneSegMask(
        other_lanes, 
        W, H, 
        normalized = False
    )

    # Parse processed data, all coords normalized
    anno_data = {
        "other_lanes" : [
            roundLineFloats(
                normalizeCoords(
                    lane, 
                    W, 
                    H
                )
            ) for lane in other_lanes
        ],
        # "drivable_path" : roundLineFloats(
        #     normalizeCoords(
        #         drivable_path, 
        #         W, 
        #         H
        #     )
        # ),
        "egoleft_lane" : roundLineFloats(
            normalizeCoords(
                left_ego, 
                W, 
                H
            )
        ),
        "egoright_lane" : roundLineFloats(
            normalizeCoords(
                right_ego, 
                W, 
                H
            )
        ),
        "mask" : mask,
    }

    return anno_data

            
# ====================================== MAIN RUN ====================================== #

if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    root_dir = "TUSimple"
    train_dir = "train_set"
    test_dir = "test_set"

    train_clip_codes = ["0313", "0531", "0601"] # Train labels are split into 3 dirs
    test_file = "test_label.json"               # Test file name

    W = 1280
    H = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process TuSimple dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        "-i",
        type = str, 
        help = "TuSimple directory (right after extraction)"
    )
    parser.add_argument(
        "--output_dir",
        "-o", 
        type = str, 
        help = "Output directory"
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        "-e",
        type = int,
        help = "Num. files each split/class you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Generate output structure
    """
    --output_dir
        |----image
        |----mask
        |----visualization
        |----drivable_path.json
    """
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    if (os.path.exists(output_dir)):
        print(f"Output dir {output_dir} already exists, purged!")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    list_subdirs = ["image", "mask", "visualization"]
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # ============================== Parsing annotations ============================== #

    train_label_files = [
        os.path.join(dataset_dir, root_dir, train_dir, f"label_data_{clip_code}.json")
        for clip_code in train_clip_codes
    ]

    """
    Make no mistake: the file `TuSimple/test_set/test_tasks_0627.json` is NOT a label file.
    It is just a template for test submission. Kinda weird they put it in `test_set` while
    the actual label file is in `TuSimple/test_label.json`.
    """
    test_label_files = [os.path.join(dataset_dir, root_dir, test_file)]
    label_files = train_label_files + test_label_files

    # Parse data by batch
    data_master = {}

    img_id_counter = -1

    for anno_file in label_files:

        print(f"\n==================== Processing data in label file {anno_file} ====================\n")
        
        # Read each line of GT text file as JSON
        with open(anno_file, "r") as f:
            read_data = [json.loads(line) for line in f.readlines()]

        for frame_data in tqdm(
            read_data,
            desc = f"Parsing annotations of {os.path.basename(anno_file)}",
            colour = "green"
        ):

            # Conduct index increment
            img_id_counter += 1

            # Early stopping, it defined
            if (early_stopping and img_id_counter >= early_stopping - 1):
                break
            
            #set_dir = "/".join(anno_file.split("/")[ : -1]) # Slap "train_set" or "test_set" to the end  <-- Specific to linux hence used os.path.dirname command below
            set_dir= os.path.dirname(anno_file)
            set_dir = os.path.join(set_dir, test_dir) if test_file in anno_file else set_dir    # Tricky test dir

            # Parse annotations
            raw_file = frame_data["raw_file"]
            anno_entry = parseAnnotations(frame_data)
            if (anno_entry is None):
                continue
            
            # Annotate raw images
            annotateGT(
                anno_entry,
                anno_raw_file = os.path.join(set_dir, raw_file),
                raw_dir = os.path.join(output_dir, "image"),
                mask_dir =  os.path.join(output_dir, "mask"),
                visualization_dir = os.path.join(output_dir, "visualization")
            )

            # Change `raw_file` to 6-digit incremental index
            data_master[str(img_id_counter).zfill(6)] = {
                # "drivable_path" : anno_entry["drivable_path"],
                "egoleft_lane" : anno_entry["egoleft_lane"],
                "egoright_lane" : anno_entry["egoright_lane"],
                "other_lanes" : anno_entry["other_lanes"],
            }

        print(f"Processed {len(read_data)} entries in above file.\n")

    print(f"Done processing data with {len(data_master)} entries in total.\n")

    # Save master data
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent = 4)