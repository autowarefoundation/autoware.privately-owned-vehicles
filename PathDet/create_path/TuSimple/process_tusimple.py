#! /usr/bin/env python3

import argparse
import json
import os
import pathlib
from PIL import Image, ImageDraw
import warnings

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

# ============================== Dataset structure ============================== #

root_dir = "TUSimple"
train_dir = "train_set"
test_dir = "test_set"

train_clip_codes = ["0313", "0531", "0601"] # Train labels are split into 3 dirs
test_file = "test_label.json"               # Test file name

img_width = 1280
img_height = 720

# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.

    Parameters
    ----------
        lane (list of tuples):
            - list of (x, y) tuples representing 2D coords of lane points.
            - from here please be reminded that all these `lane` used in this script are in
              ascending order of y-coords, which means it starts from top to bottom.
        width (float): 
            - image width, 1280 for TuSimple.
        height (float):
            - image height, 720 for TuSimple.

    Returns
    -------
        normalized lane: 
            - list of (x, y) tuples with normalized coords.

    """
    return [(x / width, y / height) for x, y in lane]


def getLaneAnchor(lane):
    """
    Determine "anchor" point of a lane.

    Here I define, the anchor of a lane is the intersection point of a lane with the bottom edge
    of an image, determined by the lane's linear equation, defined by its 2 points:
        - (x2, y2): last point of the lane, closest to bottom edge (where y = `img_height` = 720).
        - (x1, y1): closest point to (x2, y2) but with different x-coord.

    With these 2 points, slope `a` and y-intercept `b` of the line equation `y = ax + b` can be
    derived, as well as anchor point `x0`.

    Parameters
    ----------
        lane (list of tuples):
            - list of (x, y) tuples representing 2D coords of lane points.

    Returns
    -------
        tuple (x0, a, b):
            - x0 (float): anchor point, representing (x0, y = 720).
            - a (float): slope.
            - b (float): y-intercept.

    """
    (x2, y2) = lane[-1]
    (x1, y1) = lane[-2]
    for i in range(len(lane) - 2, 0, -1):
        if (lane[i][0] != x2):
            (x1, y1) = lane[i]
            break
    if (x1 == x2):
        warnings.warn(f"Vertical lane detected: {lane}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (img_height - b) / a
    
    return (x0, a, b)


def getEgoIndexes(anchors):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of lane anchors.

    Basically, left and right ego lanes are the 2 lanes closest to the center of the frame.
    Leveraging those "anchor" points, I pick the 2 anchors closest to center point of bottom
    edge (640, 720), left and right. Their lanes are ego lanes.

    This is true like 99% of the time, and is a good heuristic for this dataset. Of course it
    might mess up if the car is not driving straight, but these datasets are mostly from a 
    car cruising on highways, so it's fine ig.

    Parameters
    ----------
        anchors (list of tuples):
            - list of (x, y) tuples representing the anchors of lanes.
            - In those labels, lanes are labeled from left to right, so anchors extracted from
              them are also sorted x-coords in ascending order.

    Returns
    -------
        tuple (left_ego_idx, right_ego_idx):
            - 2 indexes in the original lane list, indicating left and right ego lanes.
    
    Sometimes there's no lanes on one side of the frame, so I return a string to indicate that.

    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= img_width / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)
    
    return "NO LANES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(left_ego, right_ego):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

    Average is taken with points having same y-coord. If not, skip to ensure alignment.

    Parameters
    ----------
        left_ego (list of tuples):
            - list of (x, y) points representing left ego lane.
        right_ego (list of tuples):
            - same as above, for right ego lane.

    Returns
    -------
        drivable_path (list of tuples):
            - list of (x, y) points representing drivable path.
    
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

    return drivable_path


def annotateGT(
        anno_entry, anno_raw_file, 
        raw_dir, labeled_dir, 
        normalized = True
):
    """
    Annotates and saves an image with lane and drivable path overlays for GT generation.
    There are both raw and labeled images saved on 2 separated dirs.

    Parameters
    ----------
        anno_entry (dict):
            - an annotation entry containing:
                + `lanes` (list of list of tuples): a list of lane points, each represented as 
                  (x, y) tuples. Coords may be normalized (0 to 1) or absolute.
                + `ego_indexes` (list of int): indexes of ego lanes in the `lanes` list.
                + `drivable_path` (list of tuples): drivable path as a list of (x, y) tuples.
        anno_raw_file (str):
            - file path of raw input image to annotate.
        raw_dir (str):
            - directory to save raw (unlabeled) image copy.
        labeled_dir (str):
            - directory to save annotated (labeled) image.
        normalized (bool, optional):
            - defaults to `True`.
            - If `True`, all coords are scaled/normalized to (0, 1). Otherwise, absolute.

        In labeled image, lanes have different colors:
            - Outer lanes: red.
            - Ego lanes: green.
            - Drivable path: yellow.

    No returns
    ----------

    """

    # Load raw image
    raw_img = Image.open(anno_raw_file).convert("RGB")

    # Define save name
    # Keep original pathname (back to 4 levels) for traceability, but replace "/" with "-"
    save_name = "_".join(anno_raw_file.split("/")[-4 : ])

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
    for idx, lane in enumerate(anno_entry["lanes"]):
        if (normalized):
            lane = [(x * img_width, y * img_height) for x, y in lane]
        if (idx in anno_entry["ego_indexes"]):
            # Ego lanes, in green
            draw.line(lane, fill = lane_colors["ego_green"], width = lane_w)
        else:
            # Outer lanes, in red
            draw.line(lane, fill = lane_colors["outer_red"], width = lane_w)
    # Drivable path, in yellow
    if (normalized):
        anno_entry["drivable_path"] = [(x * img_width, y * img_height) for x, y in anno_entry["drivable_path"]]
    draw.line(anno_entry["drivable_path"], fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Save labeled img, same format with raw, just different dir
    raw_img.save(os.path.join(labeled_dir, save_name))


def parseAnnotations(anno_path):
    """
    Parses lane annotations from raw dataset file, then extracts normalized GT data.

    First, read raw annotation/label data, then filter and process lane info, then identify 2
    ego lanes, and calculate drivable path. All coords are normalized. Basically "main" function.

    Parameters
    ----------
        anno_path (str):
            - path to annotation file containing lane data in JSON lines format.

    Returns
    -------
        anno_data (dict):
            - dictionary mapping `raw_file` paths to their corresponding processed annotations.
            - each entry contains:
                + `lanes` (list of list of tuples): normalized lane points for each lane.
                + `ego_indexes` (tuple): indexes of 2 left and right ego lanes.
                + `drivable_path` (list of tuples): normalized points of the drivable path.
                + `img_size` (tuple): dimensions of images (width, height). TuSimple is 1280 x 720.

    Notes
    -----
        - Lanes with fewer than 2 valid points (x != 2) are ignored.
        - All coords are normalized, as requested by Mr. Zain.
        - Warnings are issued for frames with no lanes on one side, while finding ego indexes.

    """
    # Read em raw
    with open(anno_path, "r") as f:
        read_data = [json.loads(line) for line in f.readlines()]

    # Parse em through
    anno_data = {}
    for item in read_data:
        lanes = item["lanes"]
        h_samples = item["h_samples"]
        raw_file = item["raw_file"]

        # Decouple from {lanes: [xi1, xi2, ...], h_samples: [y1, y2, ...]} to [(xi1, y1), (xi2, y2), ...]
        lanes_decoupled = [
            [(x, y) for x, y in zip(lane, h_samples) if x != -2]
            for lane in lanes if sum(1 for x in lane if x != -2) >= 2     # Filter out lanes < 2 points (there's actually a bunch of em)
        ]

        # Determine 2 ego lanes
        lane_anchors = [getLaneAnchor(lane) for lane in lanes_decoupled]
        ego_indexes = getEgoIndexes(lane_anchors)

        if (type(ego_indexes) is str):
            if (ego_indexes.startswith("NO")):
                warnings.warn(f"Parsing {raw_file}: {ego_indexes}")
                continue

        left_ego = lanes_decoupled[ego_indexes[0]]
        right_ego = lanes_decoupled[ego_indexes[1]]

        # Determine drivable path from 2 egos
        drivable_path = getDrivablePath(left_ego, right_ego)

        # Parse processed data, all coords normalized
        anno_data[raw_file] = {
            "lanes": [normalizeCoords(lane, img_width, img_height) for lane in lanes_decoupled],
            "ego_indexes": ego_indexes,
            "drivable_path": normalizeCoords(drivable_path, img_width, img_height),
            "img_size": (img_width, img_height),
        }

    return anno_data

            
if __name__ == "__main__":
    """
    TuSimple dataset preprocessing script for PathDet.

    CMD line args
    -------------
        --dataset_dir : str
            - path to TuSimple dataset directory.
            - only accepts the dir right after extraction. So it should be `<smth>/tu_simple`
              if you tried to download it from Kaggle.
        --output_dir : str
            - path to output directory where processed files will be stored.
        
    Notes
    -----
        - These dirs can either be relative or absolute.

    
    Structure of `output dir`:
    ------------------------
        --train
            |----raw
            |----labeled
        --test
            |----raw
            |----labeled

    Example:
    --------
        python process_tusimple.py --dataset_dir /path/to/TuSimple --output_dir /path/to/output

    """

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process TuSimple dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "TuSimple directory (right after extraction)"
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory"
    )
    args = parser.parse_args()

    # Generate output structure
    """
    --train
        |----raw
        |----labeled
    --test
        |----raw
        |----labeled
    """
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    list_batches = ["train", "test"]
    list_subbatches = ["raw", "labeled"]
    for batch in list_batches:
        for subbatch in list_subbatches:
            sub_dir = os.path.join(output_dir, batch, subbatch)
            if (not os.path.exists(sub_dir)):
                os.makedirs(sub_dir, exist_ok = True)

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

    # Parse data by batch
    data_master = {
        "train": {
            "files": train_label_files,
            "data": {}
        },
        "test": {
            "files": test_label_files,
            "data": {}
        },
    }
    for batch in list_batches:
        print(f"\n================================== Processing {batch} data ==================================\n")
        for anno_file in data_master[batch]["files"]:
            print(f"Processing {batch} file {anno_file}...")
            this_data = parseAnnotations(anno_file)
            for raw_file, anno_entry in this_data.items():
                # Annotate raw images
                annotateGT(
                    anno_entry,
                    os.path.join(dataset_dir, root_dir, f"{batch}_set", raw_file), 
                    os.path.join(output_dir, batch, "raw"),
                    os.path.join(output_dir, batch, "labeled")
                )
            data_master[batch]["data"].update(this_data)
            print(f"Processed {len(this_data)} {batch} entries in above file.\n")
        print(f"Done processing {batch} data with {len(data_master[batch]['data'])} entries in total.\n")

    # Save master data
    with open(os.path.join(output_dir, "data_master.json"), "w") as f:
        json.dump(data_master, f, indent = 4)