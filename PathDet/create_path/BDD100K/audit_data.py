import os
import json
import numpy as np
from PIL import Image
import argparse


def loadJsonFile(file_name):
    """Load a JSON file."""
    with open(file_name, 'r') as f:
        return json.load(f)


def filterFileNames(mask_dir, json_file, max_files):
    """
    Filter filenames from a JSON file based on specific conditions.

    :param mask_dir: Directory containing mask images.
    :param json_file: JSON data to filter.
    :param max_files: Maximum number of files to process.
    :return: List of filtered filenames.
    """
    name_list = []
    for file in json_file:
        # Check if 'labels' key exists and is valid
        if 'labels' not in file or not file['labels']:
            continue

        labels = file['labels']
        conditions = (
            all(label['category'] != 'crosswalk' for label in labels) and
            all(label['attributes']['laneDirection'] != 'vertical' for label in labels)
        )
        if conditions:
            name = file['name']
            if np.any(extractRedMask(mask_dir, name)):
                name_list.append(name)
        if max_files != None:
            if len(name_list) >= max_files:
                break

    return name_list


def writeFilteredJson(input_json, filtered_names, output_file):
    """Write a new JSON file with entries matching filtered names."""
    filtered_data = [
        entry for entry in input_json if entry['name'] in filtered_names
    ]
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered JSON data saved to {output_file}")


def saveSelectedFiles(mask_dir, input_json, filtered_names, output_dir):
    """Save selected mask images and write the corresponding filtered JSON."""
    output_image_dir = os.path.join(output_dir, 'colormaps')
    os.makedirs(output_image_dir, exist_ok=True)

    for name in filtered_names:
        name_png = name.replace('jpg', 'png')
        image_path = os.path.join(mask_dir, name_png)
        image = Image.open(image_path).convert("RGB")
        image.save(os.path.join(output_image_dir, name_png))

    writeFilteredJson(input_json, filtered_names, os.path.join(output_dir, 'drivable_path_audited.json'))


def extractRedMask(mask_dir, file_name):
    """Extract the red mask from a given PNG file."""
    file_name = file_name.replace('jpg', 'png')
    mask_path = os.path.join(mask_dir, file_name)

    red_min = np.array([150, 0, 0])     # Lower bound for red
    red_max = np.array([255, 100, 100]) # Upper bound for red

    img = Image.open(mask_path).convert("RGB")
    img_np = np.array(img)

    red_mask = np.all((img_np >= red_min) & (img_np <= red_max), axis=-1).astype(np.uint8) * 255
    return red_mask


def main():
    parser = argparse.ArgumentParser(description="Process drivable path data.")
    parser.add_argument('--lane_json', type=str, required=True, help="Path to the lane JSON file.")
    parser.add_argument('--mask_dir', type=str, required=True, help="Directory containing drivable path masks.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save filtered data and masks.")
    parser.add_argument('--max_files', type=int, required=False, help="Maximum number of files to process.")
    args = parser.parse_args()

    # Load the lane JSON file
    lane_json_data = loadJsonFile(args.lane_json)

    if args.max_files:
        # Filter filenames based on conditions
        filtered_names = filterFileNames(args.mask_dir, lane_json_data, args.max_files)
    else:
        filtered_names = filterFileNames(args.mask_dir, lane_json_data, None)
        

    # Save filtered files and JSON
    saveSelectedFiles(args.mask_dir, lane_json_data, filtered_names, args.output_dir)


if __name__ == '__main__':
    main()
