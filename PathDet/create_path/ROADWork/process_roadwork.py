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

def draw_trajectory_line_points(image_path, trajectory, output_dir):
    # Split the trajectory into x and y coordinates
    x_coords = np.array([point[0] for point in trajectory])
    y_coords = np.array([point[1] for point in trajectory])

    # Perform cubic spline interpolation
    cs_x = CubicSpline(np.arange(len(x_coords)), x_coords, bc_type="natural")
    cs_y = CubicSpline(np.arange(len(y_coords)), y_coords, bc_type="natural")

    # Generate new interpolated points
    new_x = np.linspace(0, len(x_coords)-1, num=500)
    new_y = cs_y(new_x)  # Get y coordinates using the cubic spline

    # Load your image where the trajectory should be drawn
    with Image.open(image_path) as img:
        # Create a Draw object to draw on the image
        draw = ImageDraw.Draw(img)
   

    # Draw the interpolated trajectory on the image
    for i in range(len(new_x) - 1):
        start_point = (int(cs_x(new_x[i])), int(new_y[i]))
        end_point = (int(cs_x(new_x[i + 1])), int(new_y[i + 1]))
        draw.line([start_point, end_point], fill='red', width=3)  # Draw the line

    # Optional: Draw original points in blue
    for x, y in zip(x_coords, y_coords):
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='blue', outline='blue')  # Draw original points

    # Save the modified image with the overlay
    img.save()

def draw_trajectory_line(image_path, trajectory, output_path):
    """
    Draw Trajectory on the image
    """
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
    
    # Draw the drivable line
    for i in range(len(trajectory) - 1):
        # Set the start and end point
        start_point = trajectory[i]
        end_point = trajectory[i + 1]
        
        # Draw the line
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


def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    draw_mode = args.draw_mode

    #### STEP 01: Check and Create Output Directories
    # Check Output directory exists
    create_output_subdirs(output_dir)

    ### STEP 02: List all the images in the image directory
    # List all the images in the image directory
    img_list = os.listdir(image_dir)

    #### STEP 03: Read all JSON files and create JSON data (list of dictionaries)
    json_data = merge_json_files(json_dir)
    print(len(json_data))

    #### STEP 04: Parse JSON data and create drivable path JSON file
    #### STEP 04(a): Read Image
    #### STEP 04(b): Read Trajectory and process the trajectory points
    #### STEP 04(c): Normalize the trajectory points
    #### STEP 04(d): Create drivable path JSON file
    #### STEP 04(e): Draw Trajectory on the image
    #### STEP 04(f): Create Trajectory Mask
    for i in json_data:
        id = i["id"]
        img_path = i["image"]
        
        # Fetche the trajectories and process them as tuples
        trajectory = process_trajectory(i["trajectory"])

        # Normalize the trajectory points
        normalized_trajectory = normalize_coords(trajectory, Image.open(img_path).size)




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