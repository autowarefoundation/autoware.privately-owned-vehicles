"""
@Author: Sohel Mahmud
@Date: 10/10/2020
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
    subdirs_list = ["image", "segmentation", "visualization"]
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
        logger.info(f"Directory created: {directory_path} !")
    else:
        print(f"Directory '{directory_path}' already exists.")

def normalize_coords(lane, width, height):
    """
    Normalize the coordinates of trajectory points.
    """
    return [(x / width, y / height) for x, y in lane]

def get_trajectory(json_data, id):
    """
    Returns list of trajectory points as tuple
    """
    for j in json_data:
        if j['id'] == id:
            break
    trajectory = [(i['x'], i['y']) for i in j['trajectory']]
    print(trajectory)
    
    return trajectory

def draw_trajectory_line(image_path, trajectory, output_path):
    """
    Draw Trajectory on the image
    """
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
    
    # Draw the drivable line
    for i in range(len(trajectory) - 1):
        # Set the start and end point
        start_point = (trajectory[i]['x'], trajectory[i]['y'])
        end_point = (trajectory[i + 1]['x'], trajectory[i + 1]['y'])
        
        # Draw the line
        draw.line([start_point, end_point], fill="yellow", width=10)
        draw.line([start_point, end_point], fill="yellow", width=10)
        draw.line([start_point, end_point], fill="yellow", width=10)
    
    img.save(output_path)

def draw_trajectory_points(image_path, trajectory, output_path):
    """
    Draw Trajectory on the image
    """
    radius = 5
    point_color = (255, 0, 0)
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
    
    for point in trajectory:
        x, y = point["x"], point["y"]
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=point_color)
    
    img.save(output_path)

def create_drivable_path_json(output_dir, json_data):
    pass

def create_trajectory_mask(image_path, trajectory, output_dir,  lane_width = 5):
    """
    Create binary mask for the drivable path using the trajectory points
    """
    mask = Image.new("L", (img_width, img_height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.line(drivable_renormed, fill = 255, width = lane_w)
    mask.save(os.path.join(mask_dir, save_name))

def convert_jpg2png(image_path, output_dir):
    """
    Convert JPG image to PNG image
    """
    with Image.open(image_path) as img:
        # Get the Image ID and concat with PNG extention
        new_img = f"{get_id(image_path)}.png"
        
        # Copy raw img and put it in raw dir.
        img.save(os.path.join(output_dir, new_img))
        
        # Log the result
        logger.info(f"Converted JPG to PNG image: {new_img}")

def get_id(image_path):
    """
    Return the image name without extention as ID
    """
    # Extract only the image name
    tmp = os.path.basename(image_path)
    # Remove the image file extention and return only the name as ID
    tmp = tmp.split('.')[0]
    return tmp

def main(args):

    json_file = args.annotation
    image_dir = args.images_dir
    output_dir = args.output_dir
    draw_mode = args.draw_mode

    #### STEP 01: Check and Create Output Directories
    # Check Output directory exists
    create_output_subdirs(output_dir)


    # Read JSON file and create JSON data (list of dictionaries)
    with open(json_file, "r") as fh:
        json_data = json.load(fh)
    
    # print(type(json_data))
    # print(len(json_data))
    # print(type(json_data[0]))
    # print(json_data[0]['id'])
    
    for image in glob.glob(f"{image_dir}/*.jpg"):
        id = get_id(image)
        trajectory = get_trajectory(json_data, id)
        # output_path = os.path.join(output_dir, f"{id}.jpg")
        # draw_line_trajectory(image, trajectory, output_path)



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
        help = "Draw mode: line or point"
    )
    args = parser.parse_args()

    main(args)