"""
@Author: Sohel Mahmud
@Date: 12/16/2024
@Description: Process ROADWork dataset for PathDet groundtruth generation

* STEP 01: Create subdirectories for the output directory
* STEP 02: Read all `JSON` files and create a combined `JSON` data (list of dictionaries)
* STEP 03: Parse `JSON` data and create drivable path `JSON` file and `Trajecory Images` (`RGB` and `Binary`)
    * STEP 03(a): Process the `Trajectory Points` as tuples
    * STEP 03(b): Crop the original image to aspect ratio `2:1` and convert from `JPG` to `PNG` format and store in output directory
    * STEP 03(c): Create `Trajectory Overlay` and crop it to aspect ratio `2:1` and save the cropped image in `PNG` format
    * STEP 03(d): Create `Cropped Trajectory Binary Mask` with aspect ratio `2:1` and save the cropped mask in `PNG` format
    * STEP 03(e): Normalize the `Trajectory Points`
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
import math
import logging
import argparse
from tqdm import tqdm

import cv2
import numpy as np
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
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)

# Adding handlers into the logger
logger.addHandler(file_handler)
# logger.addHandler(console_handler)


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
        logger.info(f"Directory '{directory_path}' already exists.")


#### JSON FILE HELPER FUNCTIONS ####


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


#### TRAJECTORY CALCULATION HELPER FUNCTIONS ####


def opt_round(x):
    """
    Optimized Round up Numbers like 45.4999 to 46 as
    45.499 is rounded up to 45.5 which can be converted
    to integer value 46
    """
    # Round up to 1 decimal point, ex
    # 4.49 to 4.5, 4.445 to 4.5, etc
    y = round(x, 1)

    return math.ceil(y) if y - int(y) > 0.5 else math.floor(y)


def process_trajectory(trajectory):
    """
    Returns list of trajectory points as tuple
    """

    return [(opt_round(i["x"]), opt_round(i["y"])) for i in trajectory]


def get_traj_peak_point(trajectory):
    """Get the peak point of the trajectory"""
    return min(trajectory, key=lambda point: point[1])


def get_traj_base_point(trajectory, img_height):

    # Minimum pixels to crop from the bottom
    # As the car bonnet is within the 90-pixel window
    crop_pixels = 90

    # Filter out the trajectory points which are below the crop pixels
    trajectory = [point for point in trajectory if img_height - point[1] >= crop_pixels]

    return max(trajectory, key=lambda point: point[1])


def get_vertical_crop_points(image_height, trajectory):
    """Get Vertical crop points"""

    # Get the base point
    _, y_bottom = get_traj_base_point(trajectory, image_height)

    # Calculate  y-offset
    y_offset = image_height - y_bottom

    return (y_offset, y_bottom)


def get_horizontal_crop_points(image_width, y_top, y_bottom):
    """Get Horizontal crop points. It depends on the vertical crop points"""

    # Calculate the cropped width
    cropped_height = y_bottom - y_top

    # Calculate the cropped width with aspect ratio 2:1
    cropped_width = cropped_height * 2

    # Calculate the x offset for each side (left and right)
    x_offset = (image_width - cropped_width) // 2

    # New x coordinate
    x_right = image_width - x_offset

    return (x_offset, x_right)


def get_offset_values(image_shape, trajectory):
    """Calculate the offset values for the image"""
    img_height, img_width = image_shape[0], image_shape[1]

    # Get the vertical crop points
    y_offset, y_bottom = get_vertical_crop_points(img_height, trajectory)

    # Get the horizontal crop points
    x_offset, _ = get_horizontal_crop_points(img_width, y_offset, y_bottom)

    return (x_offset, y_offset)


def crop_to_aspect_ratio(image, trajectory):
    """Crop the image to aspect ratio 2:1"""

    # Get the image dimensions
    img_height, img_width = image.shape[0], image.shape[1]

    # New y coordinates
    y_top, y_bottom = get_vertical_crop_points(img_height, trajectory)

    ### Pixel Cropping for 2:1 Aspect Ratio
    # Cropping pixels from left and right for aspect ratio 2:1
    x_left, x_right = get_horizontal_crop_points(img_width, y_top, y_bottom)

    # Crop the image to aspect ratio 2:1
    cropped_image = image[y_top:y_bottom, x_left:x_right]

    # Log the result
    logger.info(
        f"Successfully Converted to Aspect Ratio 2:1 with shape: {cropped_image.shape}"
    )

    return cropped_image


def normalize_coords(trajectory, image_shape, crop_shape):
    """Normalize the Trajectory coordinates"""

    # Calculate Vertical and horizontal offset (pixels crops)
    x_offset, y_offset = get_offset_values(image_shape, trajectory)
    logger.info(f"x_offset: {x_offset} y_offset: {y_offset}")

    # Get the cropped width and height
    crop_height, crop_width, _ = crop_shape

    # Normalize the trajectory points
    tmp = [
        ((x - x_offset) / crop_width, (y - y_offset) / crop_height)
        for x, y in trajectory
    ]

    # Filter out the points which are outside the range [0, 1]
    norm_traj = [(x, y) for x, y in tmp if (0 <= x <= 1) and (0 <= y <= 1)]

    return norm_traj


#### IMAGE CREATION & VISUALIZATION HELPER FUNCTIONS ####


def create_mask(image_shape):
    # Set the width and height
    width, height = image_shape[0], image_shape[1]

    # Create a binary mask
    mask = np.zeros((width, height), dtype=np.uint8)

    logger.info(f"Mask Created with shape: {mask.shape}")

    return mask


def save_image(image_id, image, output_subdir):
    """Save the image in PNG format"""

    # Create new image file path
    new_img = f"{image_id}.png"
    new_img_path = os.path.join(output_subdir, new_img)

    # Save the image in PNG format
    cv2.imwrite(new_img_path, image)

    # Log the result
    logger.info(f"Converted JPG to PNG image: {new_img}")


def show_image(image, title="Image"):
    """Display the image"""

    # Display the image
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_trajectory_line(image, trajectory, color="yellow"):
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

    # Setup Line parameters
    line_color = (0, 255, 255) if color == "yellow" else (255, 255, 255)
    line_thickness = 2

    cv2.polylines(
        image, [points], isClosed=False, color=line_color, thickness=line_thickness
    )

    return image


def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir

    #### STEP 01: Create subdirectories for the output directory

    subdirs_name = ["image", "visualization", "segmentation"]
    output_subdirs = create_output_subdirs(subdirs_name, output_dir)

    #### STEP 02: Read and Merge all JSON files and create JSON data
    json_data = merge_json_files(json_dir)

    # Get the size of the Dataset
    data_size = len(json_data)
    logger.info(f"Dataset Size: {data_size}")
    logger.info(f"Output subdirectories: {subdirs_name}")

    ## STEP 03: Parse JSON data and create drivable path JSON file
    # List of all trajectory ponts
    traj_list = []

    # Counter for JSON ID
    indx = 0

    for val in tqdm(json_data, total=len(json_data)):
        # Extract image ID and image path
        image_id = val["id"]
        image_path = os.path.join(image_dir, val["image"])

        # Read Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        logger.info(f"Image Name: {image_id}.jpg")
        logger.info(f"Image Shape: {image.shape}")

        ### STEP 03(a): Process the Trajectory points as tuples
        trajectory = process_trajectory(val["trajectory"])

        # Check Empty Trajectory Path
        if not trajectory:
            logger.info(f"Invalid Trajectory path: {indx} {image_id}")
            continue

        ### STEP 03(b): Crop the original image to aspect ratio 2:1

        # Crop Image to aspect ratio 2:1
        cropped_png_image = crop_to_aspect_ratio(image, trajectory)

        ### ASSERTIONS ###

        # Assertion: Check the cropped image
        assert cropped_png_image is not None, "cropped_png_image should not be None"

        # Assertion: Validate the cropped image dimensions
        assert cropped_png_image.shape[0] < image.shape[0], (
            f"Cropped Height should not greater than Original one. "
            f"Original image height: {image.shape[0]}, "
            f"Cropped image height: {cropped_png_image.shape[0]}."
        )

        assert cropped_png_image.shape[1] < image.shape[1], (
            f"Cropped Width should not greater than Original one. "
            f"Original image width: {image.shape[1]}, "
            f"Cropped image width: {cropped_png_image.shape[1]}."
        )

        ### STEP 03(c): Create Trajectory Overlay and crop it to aspect ratio 2:1

        # Create Trajectory Overlay
        image = draw_trajectory_line(image, trajectory, color="yellow")

        # Crop the Trajectory Overlay to aspect ratio 2:1
        crop_traj_image = crop_to_aspect_ratio(image, trajectory)

        ### STEP 03(d): Create Cropped Trajectory Binary Mask with aspect ratio 2:1

        # Create Binary Mask with the shape (width & height) of original image
        mask = create_mask(image.shape)

        # Create Trajectory Mask
        mask = draw_trajectory_line(mask, trajectory, color="white")

        # Crop Trajectory Mask
        cropped_mask = crop_to_aspect_ratio(mask, trajectory)

        ### ASSERTIONS ###

        # Assertion: Check the cropped mask
        assert cropped_mask is not None, "cropped_mask should not be None"

        # Assertion: Check if the dimensions match
        assert cropped_png_image.shape[:2] == cropped_mask.shape, (
            f"Dimension mismatch: cropped_png_image has shape {cropped_png_image.shape[:2]} "
            f"while cropped_mask has shape {cropped_mask.shape}."
        )

        ### STEP 03(e): Normalize the trajectory points
        crop_shape = crop_traj_image.shape
        norm_trajectory = normalize_coords(
            trajectory,
            image.shape,
            crop_shape,
        )

        # Check Empty Trajectory paths
        if not norm_trajectory:
            logger.info(f"INVALID Trajectory path: {indx} {image_id}")
            continue

        ### STEP 03(F): Save all images (original, cropped, and overlay) in the output directory

        # Save the Cropped Image in PNG format
        save_image(image_id, cropped_png_image, output_subdirs["image"])

        # Save the cropped trajectory overlay image in PNG format (visualization)
        save_image(image_id, crop_traj_image, output_subdirs["visualization"])

        # Save the cropped trajectory binary mask in PNG format (segmentation) - binary mask
        save_image(image_id, cropped_mask, output_subdirs["segmentation"])

        ### STEP 03(f): Build `Data Structure` for final `JSON` file
        # Generate JSON ID
        json_id = generate_jsonID(indx, data_size)
        logger.info(f"Generated JSON ID: {json_id}")

        # Create drivable path JSON file
        meta_dict = {
            "image_id": image_id,
            "drivable_path": norm_trajectory,
            "image_width": image.shape[0],
            "image_height": image.shape[1],
        }

        traj_list.append({json_id: meta_dict})

        # Increment the index for JSON ID
        indx += 1

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
        DO NOT include subdirectories or files.""",
    )
    parser.add_argument(
        "--annotation-dir",
        "-a",
        type=str,
        required=True,
        help="""
        ROADWork Trajectory Annotations Parent directory.
        Do not include subdirectories or files.""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Output directory for image, segmentation, and visualization",
    )
    args = parser.parse_args()

    main(args)
