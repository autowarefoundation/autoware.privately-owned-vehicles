#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import pathlib
from PIL import Image, ImageDraw
import warnings
from datetime import datetime

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

# ============================== Helper functions ============================== #

def normalizeCoords(lane, width, height):
    """
    Normalize the coords of lane points.

    """
    return [(x / width, y / height) for x, y in lane]


def getLaneAnchor(lane):
    """
    Determine "anchor" point of a lane.

    """
    (x2, y2) = lane[0]
    (x1, y1) = lane[1]
    for i in range(1, len(lane) - 1, 1):
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

    """
    i, j = 0, 0
    drivable_path = []
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
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < img_height)):
        x1, y1 = drivable_path[1]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (img_height - y2) / a
        drivable_path.insert(0, (x_bottom, img_height))

    # Extend drivable path to be on par with longest ego lane
    # By making it parallel with longer ego lane
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

    return drivable_path


def annotateGT(
        anno_entry, anno_raw_file, 
        raw_dir, visualization_dir, mask_dir,
        img_width, img_height,
        normalized = True,
        crop = None
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
        - Binary segmentation mask of drivable path, in "output_dir/segmentation".

    """

    # Load raw image
    raw_img = Image.open(anno_raw_file).convert("RGB")

    # Handle image cropping
    if (crop):
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]
        raw_img = raw_img.crop((
            CROP_LEFT, 
            CROP_TOP, 
            former_img_width - CROP_RIGHT, 
            former_img_height - CROP_BOTTOM
        ))

    # Define save name
    # Also save in PNG (EXTREMELY SLOW compared to jpg, for lossless quality)
    save_name = str(img_id_counter).zfill(6) + ".png"

    # Copy raw img and put it in raw dir.
    raw_img.save(os.path.join(raw_dir, save_name))
    
    # Draw all lanes & lines
    draw = ImageDraw.Draw(raw_img)
    lane_colors = {     
        "leftego_green": (0, 255, 0),
        "rightego_blue" : (0,0,255),
        "outer_yellow": (255, 255, 0)
    }
    lane_w = 5

    for idx,lane in enumerate(anno_entry["lanes"]):
        if(normalized):
            lane = [( x * img_width, y*img_height) for x,y in lane]

        # left ego lane in green
        if (idx == anno_entry["ego_indexes"][0]):           
            draw.line(lane, fill = lane_colors["leftego_green"], width = lane_w)

        # right ego lane in blue    
        elif (idx == anno_entry["ego_indexes"][1]):
            draw.line(lane, fill = lane_colors["rightego_blue"], width = lane_w)

        # Outer lanes, in yellow
        else:            
            draw.line(lane, fill = lane_colors["outer_yellow"], width = lane_w)    

    # Save visualization img, same format with raw, just different dir
    raw_img.save(os.path.join(visualization_dir, save_name))     


    # renormalize all lanes
    lanes_renormed=[]
    for lane in anno_entry["lanes"]:
        
        
            lane_renormed=[(x* img_width,y*img_height) for x,y in lane]
            lanes_renormed.append(lane_renormed)

    
    # Working on binary mask
    mask = Image.new("L", (img_width, img_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    #mask_draw.line(drivable_renormed, fill = 255, width = lane_w)

    for lane in lanes_renormed:
        mask_draw.line(lane,fill=255,width=lane_w)
 

    mask.save(os.path.join(mask_dir, save_name))  
   

def parseAnnotations(anno_path, crop = None):
    """
    Parses lane annotations from raw img + anno files, then extracts normalized GT data.

    """
    # Read each line of GT text file as JSON
    with open(anno_path, "r") as f:
        read_data = f.readlines()
        if (len(read_data) < 2):    # Some files are empty, or having less than 2 lines
            warnings.warn(f"Parsing {anno_path} : insufficient lane amount: {len(read_data)}")
            return None
        else:
            # Parse data from those JSON lines
            lanes = []
            for line in read_data:
                if line.strip():    # Not an empty line
                    points = line.strip().split(" ")
                    lane = [
                        (float(points[i]), float(points[i + 1]))
                        for i in range(0, len(points), 2)
                    ]
                    lanes.append(lane)

            # Handle image cropping
            if (crop):
                CROP_TOP = crop["TOP"]
                CROP_RIGHT = crop["RIGHT"]
                CROP_BOTTOM = crop["BOTTOM"]
                CROP_LEFT = crop["LEFT"]
                # Crop
                lanes = [[
                    (x - CROP_LEFT, y - CROP_TOP) for x, y in lane
                    if (CROP_LEFT <= x <= (former_img_width - CROP_RIGHT)) and (CROP_TOP <= y <= (former_img_height - CROP_BOTTOM))
                ] for lane in lanes]
                # Remove empty lanes
                lanes = [lane for lane in lanes if (lane and len(lane) >= 2)]   # Pick lanes with >= 2 points
                if (len(lanes) < 2):    # Ignore frames with less than 2 lanes
                    warnings.warn(f"Parsing {anno_path}: insufficient lane amount after cropping: {len(lanes)}")
                    return None

            # Determine 2 ego lanes
            lane_anchors = [getLaneAnchor(lane) for lane in lanes]
            ego_indexes = getEgoIndexes(lane_anchors)

            if (type(ego_indexes) is str):
                if (ego_indexes.startswith("NO")):
                    warnings.warn(f"Parsing {anno_path}: {ego_indexes}")
                return None

            left_ego = lanes[ego_indexes[0]]
            right_ego = lanes[ego_indexes[1]]

            # Determine drivable path from 2 egos
            drivable_path = getDrivablePath(left_ego, right_ego)

            # Parse processed data, all coords normalized
            anno_data = {
                "lanes" : [normalizeCoords(lane, img_width, img_height) for lane in lanes],
                "ego_indexes" : ego_indexes,
                "drivable_path" : normalizeCoords(drivable_path, img_width, img_height),
                "img_width" : img_width,
                "img_height" : img_height,
            }

            return anno_data

            
if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    """
    The raw annotations (not segmentation labels) for training&val set are not correct before April 16th 2018.
    To update to the right version, you could either download "annotations_new.tar.gz" and cover the original
    annotation files or download the training&val set again.
    """
    new_anno = "annotations_new"

    list_path = "list"
    test_classification = "test_split"

    img_width = 1640
    img_height = 590

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process CULane dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "CULane directory (contains all unzipped folders from GoogleDrive)",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        type = str, 
        help = "Output directory",
        required = True
    )
    parser.add_argument(
        "--crop",
        type = int,
        nargs = 4,
        help = "Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
        metavar = ("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default = [0, 390, 160, 390],    # 2 plus 2 is 4 minus 1 that's 3, quick maths
        required = False
    )
    parser.add_argument(
        "--sampling_step",
        type = int,
        help = "Sampling step for each split/class",
        required = False,
        default = 5
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
        print(f"Sampling step set to {args.sampling_step}.")
        sampling_step = args.sampling_step
    else:
        sampling_step = 1

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, each split/class stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None
    
    # Parse crop
    if (args.crop):
        print(f"Cropping image set with sizes:")
        print(f"\nTOP: {args.crop[0]},\tRIGHT: {args.crop[1]},\tBOTTOM: {args.crop[2]},\tLEFT: {args.crop[3]}")
        if (args.crop[0] + args.crop[2] >= img_height):
            warnings.warn(f"Cropping size: TOP = {args.crop[0]} and BOTTOM = {args.crop[2]} exceeds image height of {img_height}. Not cropping.")
            crop = None
        elif (args.crop[1] + args.crop[3] >= img_width):
            warnings.warn(f"Cropping size: LEFT = {args.crop[3]} and RIGHT = {args.crop[1]} exceeds image width of {img_width}. No cropping.")
            crop = None
        else:
            crop = {
                "TOP" : args.crop[0],
                "RIGHT" : args.crop[1],
                "BOTTOM" : args.crop[2],
                "LEFT" : args.crop[3],
            }
            former_img_height = img_height
            former_img_width = img_width
            img_height -= crop["TOP"] + crop["BOTTOM"]
            img_width -= crop["LEFT"] + crop["RIGHT"]
            print(f"New image size: {img_width}W x {img_height}H.\n")
    else:
        crop = None

    # Generate output structure
    """
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json
    """
    list_splits = ["train", "val", "test"]
    list_subdirs = ["image", "segmentation", "visualization"]
    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    list_train_file = os.path.join(dataset_dir, list_path, f"train.txt")    # Train set
    list_val_file = os.path.join(dataset_dir, list_path, f"val.txt")        # Val set
    # Test set, a lil bit tricky since it has 9 cats
    list_test_file = [
        os.path.join(dataset_dir, list_path, test_classification, filename)
        for filename in os.listdir(os.path.join(dataset_dir, list_path, test_classification))
    ]

    # Parse data by batch
    data_master = {}

    for split in list_splits:
        print(f"\n==================== Processing {split} data ====================\n")
        img_id_counter = -1
        if (split in ["train", "val"]):
            list_files = [os.path.join(dataset_dir, list_path, f"{split}.txt")]    # Train or Val set
        else:   # Test set and its lil more complicated
            list_test_classes = os.listdir(os.path.join(dataset_dir, list_path, test_classification))
            list_files = [
                os.path.join(dataset_dir, list_path, test_classification, filename)
                for filename in list_test_classes
            ]
        for label_file in list_files:
            with open(label_file, "r") as f:
                list_raw_files = f.readlines()

            for i in range(0, len(list_raw_files), sampling_step):
                img_path = list_raw_files[i]
                img_id_counter += 1
                img_path = img_path.strip()
                if (img_path[0] == "/"):
                    img_path = img_path[1 : ]     # Remove the leading "/" so that path join works
                anno_path = img_path.replace(".jpg", ".lines.txt")
                if (split == "test"):
                    anno_file = os.path.join(dataset_dir, anno_path)
                else:
                    anno_file = os.path.join(dataset_dir, new_anno, anno_path)

                this_data = parseAnnotations(anno_file, crop)
                if (this_data is not None):

                    print(f"Processing data in label file {anno_file}.")
                    annotateGT(
                        anno_entry = this_data,
                        anno_raw_file = os.path.join(dataset_dir, img_path),
                        raw_dir = os.path.join(output_dir, "image"),
                        visualization_dir = os.path.join(output_dir, "visualization"),
                        mask_dir = os.path.join(output_dir, "segmentation"),
                        img_height = img_height,
                        img_width = img_width,
                        crop = crop
                    )

                    # Save as 6-digit incremental index
                    img_index = str(str(img_id_counter).zfill(6))
                    data_master[img_index] = {}                    
                    data_master[img_index]["egoleft_lane"] = this_data["lanes"][this_data["ego_indexes"][0]]
                    data_master[img_index]["egoright_lane"] = this_data["lanes"][this_data["ego_indexes"][1]]
                    data_master[img_index]["outer_lane"]=[]

                    #identify the the outer lanes and save in data_master
                    for idx,lane in enumerate(this_data["lanes"]):
                        if idx in this_data["ego_indexes"]:
                            continue
                        else:
                            data_master[img_index]["outer_lane"].append(lane)
                    data_master[img_index]["img_height"] = img_height
                    data_master[img_index]["img_width"] = img_width
                    
                    # Early stopping, it defined
                    if (early_stopping and img_id_counter >= early_stopping - 1):
                        break

    # Save master data
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent = 4)