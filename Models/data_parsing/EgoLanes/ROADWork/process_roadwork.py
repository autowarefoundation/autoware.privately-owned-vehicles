"""
@Author: Sohel Mahmud
@Date: 12/16/2024
@Description: Process ROADWork dataset for PathDet groundtruth generation

* STEP 01: Create subdirectories for the output directory
* STEP 02: Read all `JSON` files and create a combined `JSON` data (list of dictionaries)
* STEP 03: Parse `JSON` data and create drivable path `JSON` file and `Trajecory Images` (`RGB` and `Binary`)
    * STEP 03(a): Process the `Trajectory Points` as tuples
    * STEP 03(b): Crop the original image to aspect ratio `2:1` and convert from `JPG` to `PNG` format
    * STEP 03(c): Normalize the `Trajectory Points` and filter out the points outside the range [0, 1]
    * STEP 03(d): Create `Trajectory Overlay` and crop it to aspect ratio `2:1`
    * STEP 03(e): Create `Cropped Trajectory Binary Mask` with aspect ratio `2:1`
    * STEP 03(f): Save all images (original, cropped, and overlay) in the output directory
    * STEP 03(g): Build `Data Structure` for final `JSON` file
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
log_filename = "/tmp/logs/process_roadwork_data.log"
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
    Generate JSON ID from 000000 to 999999. The number of digits is
    less or equal to 5 if the data size is less than 100000. Otherwise,
    the number of digits is equal to the number of digits in the data size.
    """

    # Get the number of digits in the data size
    digits = len(str(data_size))
    zfill_num = max(digits, 6)

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
    json_data = {k: v for i in traj_data for k, v in i.items()}

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


def get_traj_base_point(trajectory, img_height, crop_size=90):
    """
    Minimum pixels to crop from the bottom
    As the car bonnet is within the 90-pixel window
    except some images for Charlotte dataset
    """

    # Filter out the trajectory points which are below the crop pixels
    trajectory = [point for point in trajectory if img_height - point[1] >= crop_size]
    # print(trajectory)

    return max(trajectory, key=lambda point: point[1])


def get_vertical_crop_points(image_height, trajectory, crop_size=90):
    """Get Vertical crop points"""

    # Get the base point
    _, y_bottom = get_traj_base_point(trajectory, image_height, crop_size)
    # y_bottom = 1000

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


def get_offset_values(image_shape, trajectory, crop_size=90):
    """Calculate the offset values for the image"""
    img_height, img_width = image_shape[0], image_shape[1]

    # Get the vertical crop points
    y_offset, y_bottom = get_vertical_crop_points(img_height, trajectory, crop_size)

    # Get the horizontal crop points
    x_offset, _ = get_horizontal_crop_points(img_width, y_offset, y_bottom)

    return (x_offset, y_offset)


def crop_to_aspect_ratio(img, trajectory, crop_size=90):
    """Crop the image to aspect ratio 2:1"""

    # Get the image dimensions
    img_height, img_width = img.shape[0], img.shape[1]

    # New y coordinates
    y_top, y_bottom = get_vertical_crop_points(img_height, trajectory, crop_size)

    ### Pixel Cropping for 2:1 Aspect Ratio
    # Cropping pixels from left and right for aspect ratio 2:1
    x_left, x_right = get_horizontal_crop_points(img_width, y_top, y_bottom)

    # Crop the image to aspect ratio 2:1
    cropped_image = img[y_top:y_bottom, x_left:x_right]

    # Log the result
    logger.info(
        "Successfully Converted to Aspect Ratio 2:1 with shape: %s", cropped_image.shape
    )

    return cropped_image


def normalize_coords(trajectory, image_shape, crop_shape):
    """Normalize the Trajectory coordinates"""

    # Calculate Vertical and horizontal offset (pixels crops)
    x_offset, y_offset = get_offset_values(image_shape, trajectory)

    # Log the offset values
    logger.info("x_offset: %d y_offset: %d", x_offset, y_offset)

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


#### CHARLOTTE IMAGE HELPER FUNCTIONS ####
def check_charlotte_image(image_id):
    """Check if the image is from Charlotte dataset"""

    filter_list = [
        "charlotte_9dba2f64629f4296975300813cac6955_000000_16650_0070",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_21840_0060",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_21600_0080",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_20940_0050",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_20940_0060",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_20940_0070",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_08010_0050",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_08010_0060",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_08010_0070",
        "charlotte_9dba2f64629f4296975300813cac6955_000001_08010_0080",
    ]

    if image_id in filter_list:
        return True

    return False


#### IMAGE CREATION & VISUALIZATION HELPER FUNCTIONS ####


def show_image(image, window_size=(1600, 800), title="Image"):
    """Display the image"""

    width, height = window_size

    # Create a named window
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Show the image in the window
    cv2.imshow(title, image)

    # Resize the window after it has been shown
    cv2.resizeWindow(title, width, height)

    # Wait for a key press
    cv2.waitKey(0)

    # Destroy all windows
    cv2.destroyAllWindows()


def create_mask(image_shape):
    # Set the width and height
    width, height = image_shape[0], image_shape[1]

    # Create a binary mask
    mask = np.zeros((width, height), dtype=np.uint8)

    logger.info("Mask Created with shape: %s", mask.shape)

    return mask


def save_image(img_id, img, output_subdir):
    """Save the image in PNG format"""

    # Create new image file path
    new_img = f"{img_id}.png"
    new_img_path = os.path.join(output_subdir, new_img)

    # Save the image in PNG format
    cv2.imwrite(new_img_path, img)

    # Log the result
    logger.info("Converted JPG to PNG image: %s", new_img)


def draw_trajectory_line(img, trajectory, color="yellow"):
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
        img, [points], isClosed=False, color=line_color, thickness=line_thickness
    )

    return img


def main(args):

    json_dir = args.annotation_dir
    image_dir = args.image_dir
    crop_size = int(args.crop_size)
    output_dir = args.output_dir

    #### STEP 01: Create subdirectories for the output directory

    subdirs_name = ["image", "visualization", "segmentation"]
    output_subdirs = create_output_subdirs(subdirs_name, output_dir)

    #### STEP 02: Read and Merge all JSON files and create JSON data
    json_data = merge_json_files(json_dir)

    # Get the size of the Dataset
    data_size = len(json_data)

    # Log the result
    logger.info("Dataset Size: %d", data_size)
    logger.info("Output subdirectories: %s", subdirs_name)

    ## STEP 03: Parse JSON data and create drivable path JSON file
    # List of all trajectory ponts
    traj_list = []

    # List to store a map of JSON ID to Image ID
    image_map = []

    # Counter for JSON ID
    indx = 0

    for val in tqdm(
        json_data,
        total=len(json_data),
        position=0,
        ncols=100,
        desc="Processing ROADWork Dataset",
    ):
        # Extract image ID and image path
        image_id = val["id"]
        image_path = os.path.join(image_dir, val["image"])

        # Read Image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # log the result
        logger.info("Image Name: %s", image_id)
        logger.info("Image Shape: %s", image.shape)

        # Check Charlotte Image
        if check_charlotte_image(image_id):
            crop_size = 180
            logger.info("%s with crop size %d", image_id, crop_size)
        else:
            crop_size = 90

        ### STEP 03(a): Process the Trajectory points as tuples
        trajectory = process_trajectory(val["trajectory"])

        # VALIDATION: Check Empty Trajectory Path
        if not trajectory:
            logger.info("Invalid Trajectory path: %d %s", indx, image_id)
            continue

        ### STEP 03(b): Crop the original image to aspect ratio 2:1

        # Crop Image to aspect ratio 2:1
        cropped_png_image = crop_to_aspect_ratio(image, trajectory, crop_size)
        # show_image(cropped_png_image, title="Cropped Image")

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

        ### STEP 03(c): Normalize the trajectory points
        crop_shape = cropped_png_image.shape
        norm_trajectory = normalize_coords(
            trajectory,
            image.shape,
            crop_shape,
        )

        # VALIDATION: Check Empty Trajectory paths
        if not norm_trajectory:
            logger.info("INVALID Trajectory path: %d %s", indx, image_id)
            continue

        ### STEP 03(d): Create Trajectory Overlay and crop it to aspect ratio 2:1

        # Create Trajectory Overlay
        # Copy the original image
        traj_image = np.copy(image)
        traj_image = draw_trajectory_line(traj_image, trajectory, color="yellow")

        # Crop the Trajectory Overlay to aspect ratio 2:1
        cropped_traj_image = crop_to_aspect_ratio(traj_image, trajectory, crop_size)

        ### STEP 03(e): Create Cropped Trajectory Binary Mask with aspect ratio 2:1

        # Create Binary Mask with the shape (width & height) of original image
        mask = create_mask(image.shape)

        # Create Trajectory Mask
        mask = draw_trajectory_line(mask, trajectory, color="white")

        # Crop Trajectory Mask
        cropped_mask = crop_to_aspect_ratio(mask, trajectory, crop_size)

        ### ASSERTIONS ###

        # Assertion: Check the cropped mask
        assert cropped_mask is not None, "cropped_mask should not be None"

        # Assertion: Check if the dimensions match
        assert cropped_png_image.shape[:2] == cropped_mask.shape, (
            f"Dimension mismatch: cropped_png_image has shape {cropped_png_image.shape[:2]} "
            f"while cropped_mask has shape {cropped_mask.shape}."
        )

        ### STEP 03(f): Save all images (original, cropped, and overlay) in the output directory
        # Generate JSON ID for the image
        json_id = generate_jsonID(indx, data_size)

        # Log the result
        logger.info("Generated JSON ID: %s", json_id)

        # Save the Cropped Image in PNG format
        save_image(json_id, cropped_png_image, output_subdirs["image"])

        # Save the cropped trajectory overlay image in PNG format (visualization)
        save_image(json_id, cropped_traj_image, output_subdirs["visualization"])

        # Save the cropped trajectory binary mask in PNG format (segmentation) - binary mask
        save_image(json_id, cropped_mask, output_subdirs["segmentation"])

        ### STEP 03(g): Build `Data Structure` for final `JSON` file

        # Create drivable path JSON file
        meta_dict = {
            "drivable_path": norm_trajectory,
            "img_width": crop_shape[1],
            "img_height": crop_shape[0],
        }

        # Append the dictionary to the list
        traj_list.append({json_id: meta_dict})

        # Append the JSON ID and Image ID to the list
        image_map.append({image_id: json_id})
        tqdm.write(f"Processed {json_id} {image_id}")

        # Increment the index for JSON ID
        indx += 1

    ### STEP 04: Create drivable path JSON file
    create_drivable_path_json(traj_list, output_dir)

    ### Optional: Create JSON file for mapping image ID to JSON ID
    map_imageID_to_jsonID(image_map, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process ROADWork dataset - Egopapth groundtruth generation"
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
        "--crop-size",
        "-c",
        type=int,
        default=90,
        help="Minimum pixels to crop from the bottom",
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
