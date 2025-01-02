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
import json
import logging
import glob
from PIL import Image, ImageDraw
from datetime import datetime

def check_directory_exists(directory_path: str):
    """Check if a directory exists; if not, create it."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
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

def draw_line_trajectory(image_path, trajectory, output_path):
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

def draw_point_trajectory(image_path, trajectory, output_path):
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

    # Check Output directory exists
    check_directory_exists(output_dir)


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
        "--images-dir",
        "-i", 
        type = str,
        required=True,
        help = "ROADWork Image Datasets directory"
    )
    parser.add_argument(
        "--annotation",
        "-a", 
        type = str,
        required=True,
        help = "ROADWork Annotation File Path"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type = str,
        default="output_dir",
        help = "Output directory"
    )
    args = parser.parse_args()

    main(args)