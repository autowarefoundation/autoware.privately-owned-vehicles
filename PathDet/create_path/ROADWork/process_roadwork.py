"""
@Author: Sohel Mahmud
@Date: 12/16/2024
@Description: Process ROADWork dataset for PathDet groundtruth generation

* STEP 01: Create subdirectories for the output directory
* STEP 02: Read all JSON files and create a combined JSON data (list of dictionaries)
* STEP 03: Parse JSON data and create drivable path JSON file and Trajecory Images (RGB and Binary)
    * STEP 03(a): Read Trajectory and process the trajectory points as tuples and integer
    * STEP 03(b): Create Trajectory Overlay
    * STEP 03(c): Crop the image to aspect ratio 2:1 and convert from JPG to PNG format and store in output directory
    * STEP 03(d): Create Cropped Image Mask using STEP 03(b) - 03(c)    
    * STEP 03(e): `Normalize` the trajectory points
    * STEP 03(f): Build `Data Structure` for final `JSON` file
* STEP 04: Create drivable path JSON file

Generate output structure
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json

"""

import os
import glob
import json
import logging
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Create Log files directory
log_filename = "/tmp/logs/roadwork_data_processing.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

# Creating and configuring the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Creating Logging format
formatter = logging.Formatter("[%(asctime)s: %(name)s] %(levelname)s\t%(message)s")

# Creating file handler and setting the logging formats
file_handler = logging.FileHandler(log_filename, mode="a")
file_handler.setFormatter(formatter)

# Creating console handler with logging format
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Adding handlers into the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


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
        logger.info(f"Directory created: {directory_path}")
    else:
        print(f"Directory '{directory_path}' already exists.")


def normalize_coords(trajectory, img_shape):
    """
    Normalize the coordinates of trajectory points.
    """
    img_width, img_height = img_shape[0], img_shape[1]
    return [(x / img_width, y / img_height) for x, y in trajectory]


def process_trajectory(trajectory):
    """
    Returns list of trajectory points as tuple
    """

    return [(i["x"], i["y"]) for i in trajectory]


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


def draw_trajectory_line(image, trajectory):
    """Draw the trajectory line"""

    # Convert trajectory to a NumPy array
    trajectory_array = np.array(trajectory)
    x_coords = trajectory_array[:, 0]
    y_coords = trajectory_array[:, 1]

    # Create a parameter t for interpolation, ranging from 0 to 1
    t = np.linspace(0, 1, len(x_coords))
    t_fine = np.linspace(0, 1, 500)  # More points for smooth interpolation

    # Interpolate x and y coordinates using cubic interpolation
    x_smooth = interp1d(t, x_coords, kind="cubic")(t_fine)
    y_smooth = interp1d(t, y_coords, kind="cubic")(t_fine)

    # Convert the smoothed points to the required format for polylines
    points = np.vstack((x_smooth, y_smooth)).T.astype(np.int32).reshape((-1, 1, 2))

    # Draw the polylines on the image
    cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=2)

    return image


def create_drivable_path_json(json_dir, traj_data, output_dir):
    """
    Generate JSON file for the drivable path trajectory
    """

    # The file name is `Hard Coded` as the name is fixed
    # Output file name
    out_file_name = "drivable_path.json"
    out_file_path = os.path.join(output_dir, out_file_name)

    # Extract the annotation files name and their parent directories
    json_files = [
        "/".join(i.split("/")[-3:]) for i in glob.glob(f"{json_dir}/**/*.json")
    ]

    # Process the trajectory data - traj_data is a list of dictionaries
    traj_dict = {k: v for i in traj_data for k, v in i.items()}

    # Create JSON Data Structure
    json_data = {"files": json_files, "data": traj_dict}

    with open(out_file_path, "w") as fh:
        json.dump(json_data, fh, indent=4)
        logger.info(f"{out_file_name} successfully generated!")


def get_top_crop_points(image_height, trajectory):

    base_point = max(trajectory, key=lambda item: item[1])
    y_bottom = int(base_point[1])
    y_top = image_height - y_bottom

    return (y_top, y_bottom)


def crop_to_aspect_ratio(image, trajectory):
    """Draw crop lines on image"""

    # Get the image dimensions
    img_height, img_width = image.shape[0], image.shape[1]

    # New y coordinates
    y_top, y_bottom = get_top_crop_points(img_height, trajectory)

    ### Pixel Cropping for 2:1 Aspect Ratio
    # Cropping pixels from left and right for aspect ratio 2:1
    corrected_height = y_bottom - y_top
    corrected_width = corrected_height * 2
    corrected_pixels = (img_width - corrected_width) // 2

    # New x coordinates
    x_left = corrected_pixels
    x_right = img_width - corrected_pixels

    cropped_image = image[y_top:y_bottom, x_left:x_right]

    # Log the result
    logger.info(
        f"Successfully Converted to Aspect Ratio 2:1 with shape: {cropped_image.shape}"
    )

    return cropped_image


def show_image(image, title="Image"):
    """Display the image"""

    # Display the image
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_image(image_id, image, output_subdir):
    """Save the image in PNG format"""

    # Create new image file path
    new_img = f"{image_id}.png"
    new_img_path = os.path.join(output_subdir, new_img)

    # Save the image in PNG format
    cv2.imwrite(new_img_path, image)

    # Log the result
    logger.info(f"Converted JPG to PNG image: {new_img}")


def create_mask(image_shape):
    # Set the width and height
    width, height = image_shape[0], image_shape[1]

    # Create a binary mask
    mask = np.zeros((width, height), dtype=np.uint8)

    return mask


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


def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    display = args.display

    #### STEP 01: Create subdirectories for the output directory

    subdirs_name = ["image", "visualization", "segmentation"]
    output_subdirs = create_output_subdirs(subdirs_name, output_dir)
    print(output_subdirs)

    #### STEP 02: Read all JSON files and create JSON data (list of dictionaries)
    json_data = merge_json_files(json_dir)

    # Get the size of the Dataset
    data_size = len(json_data)
    print(data_size)

    ## STEP 03: Parse JSON data and create drivable path JSON file
    # List of all trajectory ponts
    traj_list = []

    for indx, val in enumerate(json_data):
        # Extract image ID and image path
        image_id = val["id"]
        image_path = os.path.join(image_dir, val["image"])

        # Read Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        logger.info("")

        ### STEP 03(a): Read Trajectory and process the trajectory points
        ### as tuples and integer
        trajectory = process_trajectory(val["trajectory"])

        cropped_png_image = crop_to_aspect_ratio(image, trajectory)
        create_image(image_id, cropped_png_image, output_subdirs["image"])

        ### STEP 03(b): Create Trajectory Overlay
        image = draw_trajectory_line(image, trajectory)

        ### STEP 03(c): Crop the image to aspect ratio 2:1 and convert from
        ### JPG to PNG format and store in output directory
        cropped_traj_image = crop_to_aspect_ratio(image, trajectory)

        # Save the trajectory overlay image in PNG format
        create_image(image_id, cropped_traj_image, output_subdirs["visualization"])

        # Display the cropped image
        if display == "yes":
            show_image(cropped_traj_image, title="Cropped Image")

        ### STEP 03(d): Create Cropped Image Mask using STEP 03(b) - 03(c)
        # Create Mask
        mask = create_mask(image.shape)
        logger.info(f"Mask Created with shape: {mask.shape}")

        # Create Trajectory Mask
        mask = draw_trajectory_line(mask, trajectory)

        # Crop Trajectory mask
        cropped_mask = crop_to_aspect_ratio(mask, trajectory)
        create_image(image_id, cropped_mask, output_subdirs["segmentation"])

        ### STEP 03(e): Normalize the trajectory points
        norm_trajectory = normalize_coords(trajectory, cropped_png_image.shape)

        ### STEP 03(f): Build `Data Structure` for final `JSON` file
        # Generate JSON ID
        json_id = generate_jsonID(indx, data_size)

        # Create drivable path JSON file
        meta_dict = {
            "drivable_path": norm_trajectory,
            "image_width": image.shape[0],
            "image_height": image.shape[1],
        }

        traj_list.append({json_id: meta_dict})
        break

    ### STEP 04: Create drivable path JSON file
    create_drivable_path_json(json_dir, traj_list, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process ROADWork dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--image-dir",
        "-i",
        type=str,
        required=True,
        help="""
        ROADWork Image Datasets directory. 
        DO NOT include subdirectories or files.
        """,
    )
    parser.add_argument(
        "--annotation-dir",
        "-a",
        type=str,
        required=True,
        help="""
        ROADWork Trajectory Annotations Parent directory.
        Do not include subdirectories or files.
        """,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for image, segmentation, and visualization",
    )
    parser.add_argument(
        "--display",
        "-d",
        type=str,
        default="yes",
        help="Display the cropped image. Enter: [yes/no]",
    )
    args = parser.parse_args()

    main(args)
