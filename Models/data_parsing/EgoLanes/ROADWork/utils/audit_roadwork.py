"""
@author: Sohel Mahmud
@date: 2025-01-20
@description: This script is used to generate the JSON file for the
Audit Dataset from visualization after currating the images.
Here the image names are same as the JSON ID. In case, if the 
image names are Image Id, we need to use a JSON which maps Image ID to JSON ID.
"""

import json
import glob
import os
import shutil
import argparse
import logging
from tqdm import tqdm

# Create Log files directory
log_filename = "/tmp/logs/audit_roadwork_data.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# Creating and configuring the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Creating Logging format
formatter = logging.Formatter("[%(asctime)s: %(name)s] %(levelname)s\t%(message)s")

# Creating file handler and setting the logging formats
file_handler = logging.FileHandler(log_filename, mode="a")
file_handler.setFormatter(formatter)

# Adding handlers into the logger
logger.addHandler(file_handler)


#### DIRECTORY HELPER FUNCTIONS ####


def create_output_subdirs(subdirs_list, output_dir):
    """
    Create subdirectories for the output directory
    Returns a dictionary having subdirectory paths
    """
    output_subdirs = {}

    for subdir in subdirs_list:
        subdir_path = os.path.join(output_dir, subdir)

        # Check or Create directory
        check_directory_exists(subdir_path)

        output_subdirs[subdir] = subdir_path

    return output_subdirs


def check_directory_exists(directory_path: str):
    """Check if a directory exists; if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info("Directory created: %s", directory_path)
    else:
        logger.info("Directory %s already exists.", directory_path)


#### JSON FILE HELPER FUNCTIONS ####


def generate_jsonID(indx, data_size):
    """
    Generate JSON ID from 00000 to 99999. The number of digits is
    less or equal to 5 if the data size is less than 100000. Otherwise,
    the number of digits is equal to the number of digits in the data size.
    """

    # Get the number of digits in the data size
    digits = len(str(data_size))
    zfill_num = max(digits, 5)

    return str(indx).zfill(zfill_num)


def create_drivable_path_json(traj_data, output_dir):
    """
    Generate JSON file for the drivable path trajectory
    """

    # The file name is `Hard Coded` as the name is fixed
    # Output file name
    out_file_name = "drivable_path.json"
    out_file_path = os.path.join(output_dir, out_file_name)

    # Process the trajectory data - traj_data is a list of dictionaries
    traj_dict = {k: v for i in traj_data for k, v in i.items()}

    # Create JSON Data Structure
    json_data = {"data": traj_dict}

    with open(out_file_path, "w") as fh:
        json.dump(json_data, fh, indent=4)

    # Log the result
    logger.info("%s successfully generated!", out_file_name)


def map_imageID_to_jsonID(image_map, output_dir):
    """
    Generate JSON file for mapping image ID to JSON ID
    """

    # The file name is `Hard Coded` as the name is fixed
    # Output file name
    out_file_name = "map_image_ID.json"
    out_file_path = os.path.join(output_dir, out_file_name)

    # Process the list of dictionaries to a single dictionary
    imageID_dict = {k: v for i in image_map for k, v in i.items()}

    # Create JSON Data Structure
    json_data = {"data": imageID_dict}

    with open(out_file_path, "w") as fh:
        json.dump(json_data, fh, indent=4)

    # Log the result
    logger.info("%s successfully generated!", out_file_name)


def read_json_file(json_file):
    """
    Read JSON file and return the data
    """
    with open(json_file, "r") as fh:
        jd = json.load(fh)["data"]

    return jd


def main(args):
    data_dir = args.dataset_dir
    output_dir = args.output_dir

    # Create output subdirectories
    subdirs_name = ["image", "visualization", "segmentation"]
    output_subdirs = create_output_subdirs(subdirs_name, output_dir)

    ### STEP 01: List all audited cropped overlay images
    image_path = os.path.join(data_dir, subdirs_name[1])
    image_ids = [i.split(".")[0] for i in os.listdir(image_path)]

    ### STEP 02: Read and parse JSON file
    # Retrieve JSON file path
    json_file = glob.glob(f"{data_dir}/*.json")[0]

    # Read JSON file
    json_data = read_json_file(json_file)

    # Get the size of the Dataset
    data_size = len(json_data)

    # Log the result
    logger.info("Dataset Size: %d", data_size)
    logger.info("Output subdirectories: %s", subdirs_name)

    ### STEP 03: Parse JSON data and create drivable path JSON file
    # List for all trajectory ponts
    traj_list = []

    # List to store a map of JSON ID to Image ID
    image_map = []

    for indx, id in tqdm(
        enumerate(image_ids),
        total=len(image_ids),
        position=0,
        ncols=100,
        desc="Audit ROADWork Dataset",
    ):
        # Generate JSON ID for the image
        json_id = generate_jsonID(indx, data_size)

        # Copy the images to the output directory
        for i, j in enumerate(subdirs_name):
            src = os.path.join(data_dir, j, f"{id}.png")
            dst = os.path.join(output_subdirs[j], f"{json_id}.png")
            shutil.copy(src, dst)

        # Append JSON Data to the list
        traj_list.append({json_id: json_data[id]})

        # Append the JSON ID and Image ID to the list
        image_map.append({id: json_id})
        tqdm.write(f"Processed {json_id} {id}")

    ### Step 04: Create drivable path JSON file
    create_drivable_path_json(traj_list, output_dir)

    ### Optional: Create JSON file for mapping image ID to JSON ID
    map_imageID_to_jsonID(image_map, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Audit ROADWork dataset - EgoPath groundtruth generation"
    )
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        required=True,
        help="""
        ROADWork Audited Cropped Image Dataset directory. 
        DO NOT include subdirectories or files.""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for JSON File",
    )
    args = parser.parse_args()

    main(args)
