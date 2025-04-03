import argparse
import os
import json
import cv2
import warnings
from PIL import Image, ImageDraw
import copy
import sys
import math
from utils import *


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"


warnings.formatwarning = custom_warning_format


def annotateGT(classified_lanes, gt_images_path, output_dir="visualization", crop=None):
    """
    Annotates ground truth images with lane lines and saves them to the specified output directory.

    :param classified_lanes: Dictionary containing lane keypoints classified by lane type.
    :param gt_images_path: Path to the directory containing ground truth images.
    :param output_dir: Directory to save annotated images, defaults to "visualization".
    :param crop: Optional dictionary with TOP, RIGHT, BOTTOM, LEFT values for image cropping.
    :return: None
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)

    # Define lane colors
    lane_colors = {
        "egoleft_lane": (0, 255, 0),  # Green
        "egoright_lane": (0, 0, 255),  # Blue
        "other_lanes": (255, 255, 0),  # Yellow
    }
    lane_width = 5

    img_id_counter = 0

    for image_id, lanes in classified_lanes.items():
        image_path = os.path.join(gt_images_path, image_id + ".png")
        img_id_counter += 1

        if img_id_counter == 200:
            break

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_id} not found in {gt_images_path}. Skipping.")
            continue

        # Load image
        img = Image.open(image_path).convert("RGB")

        # Apply cropping if crop is provided
        if crop:
            CROP_TOP = crop["TOP"]
            CROP_RIGHT = crop["RIGHT"]
            CROP_BOTTOM = crop["BOTTOM"]
            CROP_LEFT = crop["LEFT"]

            img = img.crop(
                (
                    CROP_LEFT,
                    CROP_TOP,
                    ORIGINAL_IMG_WIDTH - CROP_RIGHT,
                    ORIGINAL_IMG_HEIGHT - CROP_BOTTOM,
                )
            )

        draw = ImageDraw.Draw(img)
        # Draw each lane
        for lane_type, lane_data in lanes.items():
            if lane_type in ["img_height", "img_width"]:
                continue  # Skip metadata keys

            color = lane_colors.get(
                lane_type, (255, 255, 255)
            )  # Default white if unknown

            for lane_group in lane_data:
                lane_points = []  # Convert keypoints
                for x, y in lane_group:
                    new_x = x * IMG_WIDTH
                    new_y = y * IMG_HEIGHT
                    lane_points.append((new_x, new_y))
                if image_id == "000022":

                    print("Image found")
                if len(lane_points) > 1:
                    # Ensure we have enough points to draw
                    draw.line(lane_points, fill=color, width=lane_width)

        # Ensure we save as PNG by changing the file extension
        image_name_base = image_id
        # image_name_base = os.path.splitext(image_name)[0]  # Remove the extension
        image_name_png = f"{image_name_base}.png"  # Add .png extension

        save_path = os.path.join(output_dir, "visualization", image_name_png)
        img.save(save_path, format="PNG")
        print(f"Saved annotated image: {save_path} with dimensions {img.size}")

    print("Annotation complete. All images saved in", output_dir)


def classify_lanes(data, gt_images_path, output_dir):
    """
    Classify lanes from BDD100K dataset, filtering out vertical lanes and processing
    the remaining lanes to identify their positions relative to the ego vehicle.
    Lanes with similar slopes and proximity are merged to reduce redundancy.

    Args:
        data: List of dictionaries containing lane annotations from BDD100K
        gt_images_path: Path to the ground truth images
        output_dir: Directory to save the processed results

    Returns:
        Dictionary mapping image IDs to classified lane data
    """
    result = {}
    # Define threshold for slope comparison
    SLOPE_THRESHOLD = 0.6
    DISTANCE_THRESHOLD = (
        0.15 * ORIGINAL_IMG_WIDTH
    )  # Required so that parallel lanes which are much distance apart are not merged.

    for entry in data:
        image_id = entry["name"]

        # Initialize result structure
        result[image_id] = {
            "egoleft_lane": [],
            "egoright_lane": [],
            "other_lanes": [],
            "img_height": IMG_HEIGHT,
            "img_width": IMG_WIDTH,
        }

        if "labels" not in entry or not entry["labels"]:
            KeyError(f"labels not present for Image {image_id}")
            continue

        anchor_points = []

        # Collect valid lanes and their anchor points
        valid_lanes = []
        for lane in entry["labels"]:
            if "poly2d" not in lane:
                KeyError(f"poly2d not present for Image {image_id}")
                continue

            # Skip lanes with vertical direction
            if (
                "attributes" in lane
                and "laneDirection" in lane["attributes"]
                and lane["attributes"]["laneDirection"] == "vertical"
            ):
                continue

            vertices = lane["poly2d"][0]["vertices"]
            # vertices -> [[x1, y1], [x2, y2]]
            # Make a copy of vertices to avoid modifying the original
            vertices_copy = vertices.copy()
            anchor = getLaneAnchor(vertices_copy)

            # if image_id == "000022":
            #    print("Found Image")

            # if image_id == "000022":
            #    draw_keypoints_on_image(image_id, gt_images_path, vertices, output_dir)

            if (
                anchor[0] is not None and anchor[1] is not None
            ):  # Avoid lanes with undefined anchors and horizontal and vertical lanes.
                anchor_points.append((anchor[0], lane["id"], anchor[1]))
                valid_lanes.append(lane)
        # if image_id == "000022":
        #    print(f"Image is {image_id}")

        # Sort lanes by anchor x position
        anchor_points.sort()

        if image_id == "000010":
            print("Found Image")

        # Determine ego indexes
        ego_indexes = getEgoIndexes(anchor_points)

        if isinstance(ego_indexes, str):
            continue

        left_idx, right_idx = ego_indexes

        if image_id == "000153":
            print("Found Image")

        # Create a list to track which lanes have been merged
        merged_lanes = [False] * len(anchor_points)

        # For each lane, check if it can be merged with the next lane
        i = 0
        while i < len(anchor_points) - 1:
            if merged_lanes[i]:
                i += 1
                continue

            current_slope = anchor_points[i][2]
            next_slope = anchor_points[i + 1][2]

            if image_id == "000153" and (i == left_idx or i == left_idx - 1):
                print("Image found")

            # Check if both slopes are valid (not None) and not perpendicular
            slopes_valid = (
                current_slope is not None
                and next_slope is not None
                and (current_slope * next_slope != -1)
            )
            anchor_distance_valid = (
                abs(anchor_points[i][0] - anchor_points[i + 1][0]) <= DISTANCE_THRESHOLD
            )

            # Check if slopes are within threshold
            slopes_similar = False
            if slopes_valid:
                # Calculate the angle between the lines
                angle_radians = math.atan(
                    abs((current_slope - next_slope) / (1 + current_slope * next_slope))
                )
                slopes_similar = angle_radians <= SLOPE_THRESHOLD

            if slopes_valid and slopes_similar and anchor_distance_valid:
                # Get the lane IDs and objects
                current_lane_id = anchor_points[i][1]
                next_lane_id = anchor_points[i + 1][1]

                current_lane = next(
                    lane for lane in valid_lanes if lane["id"] == current_lane_id
                )
                next_lane = next(
                    lane for lane in valid_lanes if lane["id"] == next_lane_id
                )

                # Merge the lanes
                merged_lane = merge_lane_lines(
                    current_lane["poly2d"][0]["vertices"],
                    next_lane["poly2d"][0]["vertices"],
                )

                # Add to the appropriate category
                if i + 1 == left_idx:
                    result[image_id]["egoleft_lane"].append(merged_lane)
                elif i == right_idx:
                    result[image_id]["egoright_lane"].append(merged_lane)
                else:
                    result[image_id]["other_lanes"].append(merged_lane)

                # Mark both lanes as merged
                merged_lanes[i] = True
                merged_lanes[i + 1] = True

                # Skip the next lane since it's been merged
                i += 2
            else:
                # This lane wasn't merged, add it as is
                lane_id = anchor_points[i][1]
                lane = next(lane for lane in valid_lanes if lane["id"] == lane_id)

                if i == left_idx:
                    result[image_id]["egoleft_lane"].append(
                        lane["poly2d"][0]["vertices"]
                    )
                elif i == right_idx:
                    result[image_id]["egoright_lane"].append(
                        lane["poly2d"][0]["vertices"]
                    )
                else:
                    result[image_id]["other_lanes"].append(
                        lane["poly2d"][0]["vertices"]
                    )

                i += 1

        # Handle the last lane if it wasn't merged
        if i < len(anchor_points) and not merged_lanes[i]:
            lane_id = anchor_points[i][1]
            lane = next(lane for lane in valid_lanes if lane["id"] == lane_id)

            if left_idx is not None and i == left_idx:
                result[image_id]["egoleft_lane"].append(lane["poly2d"][0]["vertices"])
            elif right_idx is not None and i == right_idx:
                result[image_id]["egoright_lane"].append(lane["poly2d"][0]["vertices"])
            else:
                result[image_id]["other_lanes"].append(lane["poly2d"][0]["vertices"])

        # if image_id == "000022":
        #    print(f"Image is {image_id}")
    return result


def format_data(data, crop):
    """
    Normalize all keypoints in the data by dividing x-coordinates by img_width
    and y-coordinates by img_height to scale them between 0 and 1.

    Args:
        data (dict): Dictionary containing image data with keypoints and dimensions
        crop (dict): Dictionary specifying cropping boundaries (TOP, RIGHT, BOTTOM, LEFT)

    Returns:
        dict: Dictionary with the same structure but normalized keypoints and sequential image keys
    """
    formatted_data = {}

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]

    for idx, (image_key, image_data) in enumerate(data.items()):
        # Create a copy of the image data
        formatted_image = image_data.copy()

        # Normalize lane keypoints
        for lane_type in ["egoleft_lane", "egoright_lane", "other_lanes"]:
            if lane_type in image_data:
                formatted_lanes = []

                for lane in image_data[lane_type]:
                    formatted_lane = []

                    if image_key == "000022":
                        print(f"Image is {image_key}")
                    # Interpolate for very few points.
                    if len(lane) < 10:
                        lane = interpolated_list(
                            lane, 5
                        )  # 5 points between each pair of points.

                    for point in lane:
                        # Normalize x by width and y by height
                        if (
                            CROP_LEFT <= point[0] <= (ORIGINAL_IMG_WIDTH - CROP_RIGHT)
                        ) and (
                            CROP_TOP <= point[1] <= (ORIGINAL_IMG_HEIGHT - CROP_BOTTOM)
                        ):
                            new_x = point[0] - CROP_LEFT
                            new_y = point[1] - CROP_TOP
                            # Normalize and crop
                            formatted_lane.append(
                                [new_x / IMG_WIDTH, new_y / IMG_HEIGHT]
                            )
                        else:
                            # write complex logic to find projected point on cropped image.
                            pass  # Placeholder for the complex logic

                    formatted_lanes.append(formatted_lane)

                # Update the list in place
                formatted_image[lane_type] = formatted_lanes
        formatted_data[image_key] = formatted_image
        if image_key == "000022":
            print(f"Image is {image_key}")
    return formatted_data


def getLaneAnchor(lane):
    """
    Determine "anchor" point of a lane.

    """
    # Sort lane keypoints in decreasing order of y coordinates.
    lane.sort(key=lambda point: point[1], reverse=True)

    (x2, y2) = lane[0]
    (x1, y1) = lane[1]

    num_vertical = 0
    num_horizontal = 0

    for i in range(1, len(lane) - 1, 1):
        if lane[i][0] != x2:
            (x1, y1) = lane[i]
            break
    if x1 == x2:
        num_vertical += 1
        return (x1, None, None)
    # Get slope and intercept
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    # Get projected point on the image's bottom
    if a == 0:
        num_horizontal += 1
        return (x1, None, None)
    x0 = (ORIGINAL_IMG_HEIGHT - b) / a

    return (x0, a, b)


def getEgoIndexes(anchors):
    """
    Identifies the two ego lanes (left and right) from sorted anchor points.

    This function determines which lanes are the ego-vehicle lanes by finding
    the lanes closest to the center of the image (ORIGINAL_IMG_WIDTH/2).

    Args:
        anchors: List of tuples (x_position, lane_id, slope), sorted by x_position.
                Each tuple represents a lane with its anchor point x-coordinate,
                unique ID, and slope.

    Returns:
        tuple: (left_idx, right_idx) - Indices of the left and right ego lanes
               in the anchors list.
        str: Error message if proper ego lanes cannot be determined.

    Logic:
        1. Iterate through sorted anchors to find the first lane with x-position
           greater than or equal to the image center.
        2. This lane and the one before it are considered the ego lanes.
        3. Special cases:
           - If the first lane is already past center (i=0), use it and the next lane
           - If no lanes are past center, use the last two lanes
           - If there's only one lane, return an error message
    """
    tolerance = 0.05  # 5% tolerance
    for i in range(len(anchors)):
        if anchors[i][0] >= (ORIGINAL_IMG_WIDTH - (tolerance * ORIGINAL_IMG_WIDTH)) / 2:
            if i == 0:
                # First lane is already past the center
                if len(anchors) >= 2:
                    # Use first and second lanes as ego lanes
                    return (i, i + 1)
                return (None, 0)
            # Normal case: use the lane before and at the center
            return (i - 1, i)

    # If we get here, no lanes are past the center
    if len(anchors) >= 2:
        # Use the last two lanes as ego lanes
        return (len(anchors) - 2, len(anchors) - 1)
    elif len(anchors) == 1:
        return (0, None)
    return "Unhandled Edge Case"


def process_binary_mask(args):
    """
    Processes binary lane mask images from BDD100K dataset.

    This function reads binary lane mask images, inverts them (assuming lanes are black
    and background is white in the original), crops them according to specified dimensions,
    and saves them with serialized filenames in the output directory.

    Args:
        args: Command-line arguments containing:
            - labels_dir: Directory containing the BDD100K lane mask images
            - output_dir: Directory where processed images will be saved
            - crop: Tuple of (top, right, bottom, left) crop values in pixels

    Output:
        Processed binary masks saved to {args.output_dir}/segmentation/ with
        serialized filenames (000000.png, 000001.png, etc.)

    Note:
        The function inverts the binary masks to make lanes white (255) and
        background black (0) for consistency with the pipeline's requirements.
    """

    CROP_TOP, CROP_RIGHT, CROP_BOTTOM, CROP_LEFT = args.crop

    # Check output directory exists or not
    os.makedirs(os.path.join(args.output_dir, "segmentation"), exist_ok=True)

    # Get list of images from input directory
    image_files = [
        f
        for f in os.listdir(os.path.join(args.labels_dir, "lane", "masks", "train"))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    binary_lane_mask_path = os.path.join(args.labels_dir, "lane", "masks", "train")
    # print(binary_lane_mask_path)

    img_id_counter = 0
    # Process each image
    for image_name in image_files:
        input_path = os.path.join(
            binary_lane_mask_path, image_name
        )  # Original image path
        image_id = str(str(img_id_counter).zfill(6))
        img_id_counter += 1
        if img_id_counter == 200:
            break
        output_path = os.path.join(args.output_dir, "segmentation", f"{image_id}.png")

        # Read the binary mask image in grayscale mode
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            # **Invert the binary mask** (assumes lane is black and background is white)
            img = 255 - img  # Invert colors

            # Crop the image
            height, width = img.shape
            cropped_img = img[
                CROP_TOP : height - CROP_BOTTOM, CROP_LEFT : width - CROP_RIGHT
            ]

            # Save the processed image
            cv2.imwrite(output_path, cropped_img)
        else:
            print(f"Skipped: {image_name} (Invalid image)")


def saveGT(json_data_path, gt_images_path, args):
    """
    Copies images from gt_images_path to the output directory with serialized filenames.
    Uses image names from lane_train.json instead of reading directory contents.

    Args:
        gt_images_path (str): Path to the directory containing ground truth images.
        args: Command-line arguments containing output_dir and labels_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    with open(json_data_path, "r") as f:
        data = json.load(f)

    # Check first data.
    print(json.dumps(data[:2], indent=4))

    print(f"Found {len(data)} entries in lane_train.json")

    img_id_counter = 0
    skipped_img_counter = 0

    # Process each image entry from the JSON file
    for entry in data:
        image_id = str(str(img_id_counter).zfill(6))  # Format as 000000, 000001, etc.
        img_id_counter += 1
        if "name" not in entry:
            skipped_img_counter += 1
            continue

        # if image_id == "000022":
        #    print("Found Image")

        input_path = os.path.join(gt_images_path, entry["name"])  # Original image path
        entry["name"] = image_id

        # Skip if file doesn't exist
        if not os.path.exists(input_path):
            print(
                f"Warning: Ground truth image {image_id} not found in {gt_images_path}. Skipping."
            )
            continue

        output_path = os.path.join(args.output_dir, "images", f"{image_id}.png")

        # Read the image
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)  # OpenCV loads as BGR

        if img is not None:
            # Save the image
            cv2.imwrite(output_path, img)
        else:
            skipped_img_counter += 1
            continue
        if img_id_counter == 200:
            break

    print(
        f"Copied {img_id_counter} images to {os.path.join(args.output_dir, 'images')} and skipped {skipped_img_counter} images."
    )

    return data


# Create interpolation functions for x-coordinates
def interpolate_x(y, vertices):
    """Get x value at given y using linear interpolation"""
    for i in range(len(vertices) - 1):
        y1, y2 = vertices[i][1], vertices[i + 1][1]
        if y1 <= y <= y2:
            # use current best ground truth values for interpolation.
            x1, x2 = vertices[i][0], vertices[i + 1][0]
            # Linear interpolation
            if y2 - y1 == 0:
                return x1
            return x1 + (x2 - x1) * (y - y1) / (y2 - y1)
    return None


def interpolated_list(point_list, req_points):
    """
    Takes original point list and inserts 3 interpolated points between
    each consecutive pair of points in the original list.
    Uses the linear equation between each specific pair of points.

    Args:
        point_list: List of [x, y] coordinates
        req_points: Number of interpolated points to insert between each pair of points
    Returns:
        List of original points with interpolated points inserted
    """
    if not point_list or len(point_list) < 2:
        return point_list.copy()

    result = []

    # Process each pair of consecutive points in original ordering
    for i in range(len(point_list) - 1):
        p1 = point_list[i]
        p2 = point_list[i + 1]

        # Add the first point of this segment
        result.append(p1.copy())

        # Get the linear equation between p1 and p2
        x1, y1 = p1
        x2, y2 = p2

        # Add interpolated points between p1 and p2
        for j in range(1, req_points + 1):
            t = j / (
                req_points + 1
            )  # Choose t to be 1/req_points, 2/req_points, ..., req_points/req_points + 1

            # Calculate y using linear interpolation
            y = y1 + t * (y2 - y1)

            # Calculate x using the linear equation between p1 and p2
            if y2 - y1 == 0:  # Horizontal line
                x = x1 + t * (x2 - x1)
            else:
                # Linear interpolation for x based on y
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)

            result.append([x, y])

    # Add the last point
    result.append(point_list[-1].copy())

    return result


def merge_lane_lines(vertices1, vertices2):
    """
    Merge two lane lines into a single lane by interpolating and averaging points.

    Args:
        vertices1: List of [x, y] coordinates for first lane
        vertices2: List of [x, y] coordinates for second lane
    Returns:
        List of merged [x, y] coordinates
    """
    # Sort vertices by y-coordinate
    v1_sorted = sorted(vertices1, key=lambda p: p[1])
    v2_sorted = sorted(vertices2, key=lambda p: p[1])

    # Get y-coordinate range
    y_min = max(v1_sorted[0][1], v2_sorted[0][1])
    y_max = min(v1_sorted[-1][1], v2_sorted[-1][1])

    # Generate merged points
    merged = []
    num_points = 50  # Number of points in merged lane

    for i in range(num_points):
        y = y_min + (y_max - y_min) * i / (num_points - 1)
        x1 = interpolate_x(y, v1_sorted)
        x2 = interpolate_x(y, v2_sorted)

        if x1 is not None and x2 is not None:
            x = (x1 + x2) / 2  # Average x coordinates
            merged.append([x, y])

    return merged


if __name__ == "__main__":
    # ============================== Constants ============================== #

    # Define Dimension
    ORIGINAL_IMG_WIDTH = 1280
    ORIGINAL_IMG_HEIGHT = 720

    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description="Process BDD100k dataset - EgoLanes groundtruth generation"
    )

    # bdd100k/images/100k/
    parser.add_argument(
        "--image_dir",
        type=str,
        help="BDD100k ground truth image parent directory (right after extraction)",
    )

    # bdd100k/labels/
    parser.add_argument(
        "--labels_dir",
        type=str,
        help="BDD100k labels directory (right after extraction)",
        required=True,
    )

    # ./output
    parser.add_argument(
        "--output_dir", type=str, help="Desired output directory", required=True
    )

    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        help="Crop image: [TOP, RIGHT, BOTTOM, LEFT]. Must always be 4 ints. Non-cropped sizes are 0.",
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        default=[0, 140, 220, 140],
        required=False,
    )

    args = parser.parse_args()

    json_data_path = os.path.join(
        args.labels_dir, "lane", "polygons", "lane_train.json"
    )

    # Save ground truth images.
    gt_images_path = os.path.join(args.image_dir, "train")
    data = saveGT(
        json_data_path, gt_images_path, args
    )  # Do not enable if not required.

    # ============================== Get crop image params ============================== #

    # Updated dimensions
    IMG_WIDTH = ORIGINAL_IMG_WIDTH - args.crop[1] - args.crop[3]
    IMG_HEIGHT = ORIGINAL_IMG_HEIGHT - args.crop[0] - args.crop[2]

    crop = {
        "TOP": args.crop[0],
        "RIGHT": args.crop[1],
        "BOTTOM": args.crop[2],
        "LEFT": args.crop[3],
    }

    # ============================== Save binary mask ============================== #
    # process_binary_mask(args)

    # ============================== Identify EgoLanes ============================== #
    gt_images_path = os.path.join(args.output_dir, "images")
    classified_lanes = classify_lanes(data, gt_images_path, output_dir=args.output_dir)

    # # # ============================== Normalise, crop Image data  ============================== #
    formatted_data = format_data(classified_lanes, crop)

    # # # ============================== AnnotateGT ================================ # # #

    annotateGT(formatted_data, gt_images_path, output_dir=args.output_dir, crop=crop)

    # # # ============================== Save result JSON ============================== #

    # Save classified lanes as new JSON file
    # output_file = os.path.join(args.output_dir, "bdd100k_egolanes_train.json")

    # with open(output_file, "w") as f:
    #     json.dump(formatted_data, f, indent=4)
