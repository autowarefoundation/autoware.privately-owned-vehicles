"""
@Author: Sohel Mahmud
@Date: 12/16/2024
@Description: Process ROADWork dataset for PathDet groundtruth generation

* STEP 01: Create subdirectories for the output directory
* STEP 02: Read all JSON files and create a combined JSON data (list of dictionaries)
* STEP 03: Parse JSON data and create drivable path JSON file and Trajecory Images (RGB and Binary)
    * STEP 03(a): Convert JPG to PNG format and store in output directory
    * STEP 03(b): Read Trajectory and process the trajectory points as tuples
    * STEP 03(c): Create Trajectory Overlay and Mask, and save
    * STEP 03(d): Normalize the trajectory points
    * STEP 03(e): Create drivable path JSON file

Generate output structure
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json

"""

import numpy as np
import argparse
import os
import glob
import json
import logging
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline

# Create Log files directory
log_filename = '/tmp/logs/roadwork_data_processing.log'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# Creating and configuring the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Creating Logging format
formatter = logging.Formatter(
    "[%(asctime)s: %(name)s] %(levelname)s\t%(message)s")

# Creating file handler and setting the logging formats
file_handler = logging.FileHandler(log_filename, mode='a')
file_handler.setFormatter(formatter)

# Creating console handler with logging format
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Adding handlers into the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def create_output_subdirs(output_dir):
    """
    Create subdirectories for the output directory
    """
    subdirs_list = ["image", "visualization", "segmentation"]
    output_subdirs = []

    for subdir in subdirs_list:
        subdir_path = os.path.join(output_dir, subdir)
        check_directory_exists(subdir_path)
        output_subdirs.append(subdir_path)
    
    return output_subdirs
    
def check_directory_exists(directory_path: str):
    """Check if a directory exists; if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Directory created: {directory_path}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def normalize_coords(trajectory, img_shape):
    """
    Normalize the coordinates of trajectory points.
    """
    img_width, img_height = img_shape
    return [(x / img_width, y / img_height) for x, y in trajectory]

def process_trajectory(trajectory):
    """
    Returns list of trajectory points as tuple
    """
    
    return [(i['x'], i['y']) for i in trajectory]

def merge_json_files(json_dir):
    """
    Merge multiple JSON files into a single list of dictionaries
    Return: List of dictionaries
    """
    merged_data = []

    for json_file in glob.glob(f"{json_dir}/**/*.json"):
        with open(json_file, "r") as fh:
            merged_data += json.load(fh)
    
    return merged_data

def draw_trajectory(image_path, image_id, trajectory, output_subdirs):
    """
    Draw the trajectory line both on the image and the mask, and save
    """

    # Open the image and create a draw object
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        img_draw = ImageDraw.Draw(img)

        # Create a new image name
        new_img = f"{image_id}.png"

        # Create a new image with black background
        mask = Image.new("L", (img_width, img_height), 0)

        # Create a draw object
        mask_draw = ImageDraw.Draw(mask)

        # Split the trajectory into x and y coordinates
        x_coords = np.array([point[0] for point in trajectory])
        y_coords = np.array([point[1] for point in trajectory])

        # Perform cubic spline interpolation
        cs_x = CubicSpline(np.arange(len(x_coords)), x_coords, bc_type="natural")
        cs_y = CubicSpline(np.arange(len(y_coords)), y_coords, bc_type="natural")

        # Generate new interpolated points
        new_x = np.linspace(0, len(x_coords)-1, num=500)
        new_y = cs_y(new_x)  # Get y coordinates using the cubic spline

        # Draw the interpolated trajectory on the image
        for i in range(len(new_x) - 1):
            # Start and end point
            start_point = (int(cs_x(new_x[i])), int(new_y[i]))
            end_point = (int(cs_x(new_x[i + 1])), int(new_y[i + 1]))

            # Draw the line
            img_draw.line([start_point, end_point], fill='yellow', width=5)
            mask_draw.line([start_point, end_point], fill=255, width=5)

        # # Optional: Draw original points in blue
        # for x, y in zip(x_coords, y_coords):
        #     mask_draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red', outline='blue')

        ##### OVERLAY IMAGE #####
        # Save or show the modified image with the overlay
        img.save(os.path.join(output_subdirs[1], new_img))
        # Log the result for trajectory line overlay
        logger.info(f"Overlay Trajectory Image generated in: {output_subdirs[1]}")

        ##### MASK #####
        # Save the mask with trajectory line
        mask.save(os.path.join(output_subdirs[2], new_img))
        # Log the result for trajectory line mask
        logger.info(f"Trajectory Mask generated in: {output_subdirs[2]}")
        
def create_drivable_path_json(json_dir, traj_data, output_dir):
    """
    Generate JSON file for the drivable path trajectory
    """
    
    # The file name is `Hard Coded` as the name is fixed
    # Output file name
    out_file_name = "drivable_path.json"
    out_file_path = os.path.join(output_dir, out_file_name)

    
    # parent_dirs = "/".join(json_dir.split('/')[-2:])
    # json_files = [f"{parent_dirs}/{f}" for f in os.listdir(json_dir) if f.endswith('.json')]

    # Extract the annotation files name and their parent directories
    json_files = ["/".join(i.split('/')[-3:]) for i in glob.glob(f"{json_dir}/**/*.json")]


    # Process the trajectory data - traj_data is a list of dictionaries
    traj_dict = { k: v for i in traj_data for k, v in i.items()}

    # Create JSON Data Structure
    json_data = {
        "files": json_files,
        "data": traj_dict
    }

    with open(out_file_path, "w") as fh:
        json.dump(json_data, fh, indent=4)
        logger.info(f"{out_file_name} successfully generated!")

def convert_jpg2png(image_id, image_path, output_subdir):
    """
    Convert JPG image to PNG image
    """
    with Image.open(image_path) as image:
        # Create a new image name
        new_img = f"{image_id}.png"

        # Save the image in PNG format
        image.save(os.path.join(output_subdir, new_img))
    
        # Log the result
        logger.info(f"Converted JPG to PNG image: {new_img}")

def get_image_shape(image_path):
    """
    Get the shape of the image
    """
    with Image.open(image_path) as img:
        return img.size

def generate_jsonID(indx, data_size):
    """
    Generate JSON ID from 00000 to 99999. The number of digits is 
    less or equal to 5 if the data size is less than 100000. Otherwise,
    the number of digits is equal to the number of digits in the data size.
    """

    # Get the number of digits in the data size
    digits = len(str(data_size))

    if digits > 5:
        zfill_num = digits

    else:
        zfill_num = 5

    return str(indx).zfill(zfill_num)

def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    
    #### STEP 01: Create subdirectories for the output directory
    output_subdirs = create_output_subdirs(output_dir)
    print(output_subdirs)

    #### STEP 02: Read all JSON files and create JSON data (list of dictionaries)
    json_data = merge_json_files(json_dir)
    
    # Get the size of the Dataset
    data_size = len(json_data)
    print(data_size)

    ## STEP 04: Parse JSON data and create drivable path JSON file
    
    # List of all trajectory ponts
    traj_list = []

    for indx, val in enumerate(json_data):
        # Extract image ID and image path
        image_id = val["id"]
        image_path = os.path.join(image_dir, val["image"])

        # Generate JSON ID
        json_id = generate_jsonID(indx, data_size)

        ### STEP 03(a): Convert JPG to PNG format and store in output directory
        convert_jpg2png(image_id, image_path, output_subdirs[0])
        
        ### STEP 03(b): Read Trajectory and process the trajectory points as tuples
        trajectory = process_trajectory(val["trajectory"])

        ### STEP 03(c): Create Trajectory Overlay and Mask, and save
        draw_trajectory(image_path, image_id, trajectory, output_subdirs)

        # Get Image shape
        image_shape = get_image_shape(image_path)

        ### STEP 03(d): Normalize the trajectory points
        norm_trajectory = normalize_coords(trajectory, image_shape)

        # Create drivable path JSON file
        meta_dict = {
            "drivable_path": norm_trajectory,
            "image_width": image_shape[0],
            "image_height": image_shape[1]
        }

        traj_list.append({json_id: meta_dict})
        break

    ### STEP 03(e): Create drivable path JSON file
    create_drivable_path_json(json_dir, traj_list, output_dir)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = "Process ROADWork dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--image-dir",
        "-i", 
        type = str,
        required=True,
        help = "ROADWork Image Datasets directory"
    )
    parser.add_argument(
        "--annotation-dir",
        "-a", 
        type = str,
        required=True,
        help = """
        ROADWork Trajectory Annotations Parent directory. 
        Do not include subdirectories or files
        """
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type = str,
        default="output",
        help = "Output directory"
    )
    args = parser.parse_args()

    main(args)