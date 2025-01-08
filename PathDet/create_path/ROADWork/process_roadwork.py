"""
@Author: Sohel Mahmud
@Date: 12/16/2024
@Description: Process ROADWork dataset for PathDet groundtruth generation

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
log_filename = 'logs/roadwork_date_processing.log'
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

    for json_file in glob.glob(f"{json_dir}/*.json"):
        with open(json_file, "r") as fh:
            merged_data += json.load(fh)
    
    return merged_data

def draw_trajectory_line(image_path, image_id, trajectory, output_subdirs):
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
        #     mask_draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red', outline='blue')  # Draw original points

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
        
def create_drivable_path_json(output_dir, json_data):
    pass

def create_trajectory_mask(image_shape, mask_name, trajectory, output_dir,  lane_width = 5):
    """
    Create binary mask for the drivable path using the trajectory points
    """
    # Extract the image width and height
    img_width, img_height = image_shape

    # Create a new image with black background
    mask = Image.new("L", (img_width, img_height), 0)
    
    # Create a draw object
    mask_draw = ImageDraw.Draw(mask)
    
    # Draw the drivable path
    mask_draw.line(trajectory, fill = 255, width = lane_width)
    
    # Save the mask in output directory
    mask.save(os.path.join(output_dir, mask_name))

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

def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    draw_mode = args.draw_mode

    #### STEP 01: Create subdirectories for the output directory
    output_subdirs = create_output_subdirs(output_dir)
    print(output_subdirs)

    #### STEP 02: Read all JSON files and create JSON data (list of dictionaries)
    json_data = merge_json_files(json_dir)
    print(len(json_data))

    ### STEP 04: Parse JSON data and create drivable path JSON file
    ### STEP 04(a): Convert JPG to PNG format
    ### STEP 04(b): Read Trajectory and process the trajectory points
    ### STEP 04(c): Draw Trajectory line on the image and save
    ### STEP 04(d): Create Trajectory Mask and save
    ### STEP 04(e): Normalize the trajectory points
    ### STEP 04(f): Create drivable path JSON file

    for i in json_data:
        image_id = i["id"]
        image_path = os.path.join(image_dir, i["image"])
        print(image_path)

        # Convert JPG to PNG and store in output directory
        convert_jpg2png(image_id, image_path, output_subdirs[0])
        
        # Fetch and process the trajectory points as tuples
        trajectory = process_trajectory(i["trajectory"])

        # Draw Trajectory line on the image and save
        draw_trajectory_line(image_path, image_id, trajectory, output_subdirs)

        # Get Image shape
        image_shape = get_image_shape(image_path)

        # Normalize the trajectory points
        norm_trajectory = normalize_coords(trajectory, image_shape)

        # Create drivable path JSON file

        break



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
        help = "ROADWork Trajectory File directory"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type = str,
        default="output",
        help = "Output directory"
    )
    parser.add_argument(
        "--draw-mode",
        "-m",
        type = str,
        default="line",
        help = "Draw mode: line, point, both"
    )
    args = parser.parse_args()

    main(args)