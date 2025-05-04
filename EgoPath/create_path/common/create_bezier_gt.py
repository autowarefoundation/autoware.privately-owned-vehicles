#! /usr/bin/env python3

import argparse
from PIL import Image, ImageDraw
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../../../"
)))
import shutil
import json

from Models.data_utils.load_data_ego_path import (
    LoadDataEgoPath, 
    VALID_DATASET_LIST
)

# Evaluate the x, y coords of a bezier point list at a given t-param value
def evaluate_bezier(bezier, t):

        # Throw an error if parameter t is out of bounds
        if not (0 < t < 1):
            raise ValueError("Please ensure t parameter is in the range [0, 1]")
        
        # Evaluate cubic bezier curve for value of t in range [0, 1]
        x = ((1-t)**3)*bezier[0] + 3*((1-t)**2)*t*bezier[2] \
            + 3*(1-t)*(t**2)*bezier[4] + (t**3)*bezier[6]
        
        y = ((1-t)**3)*bezier[1] + 3*((1-t)**2)*t*bezier[3] \
            + 3*(1-t)*(t**2)*bezier[5] + (t**3)*bezier[7]
        
        return x, y


if (__name__ == "__main__"):

    # ================ Input args ================

    parser = argparse.ArgumentParser(
        description = "Process CurveLanes dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "-d", "--dataset_dir",
        dest = "dataset_dir", 
        type = str, 
        help = f"Master dataset directory, should contain all 6 datasets: {VALID_DATASET_LIST}",
        required = True
    )
    args = parser.parse_args()

    # Parse dirs
    dataset_dir = args.dataset_dir

    IMAGES_DIRNAME = "image"
    JSON_PATHNAME = "drivable_path.json"
    BEZIER_DIRNAME = "bezier-visualization"
    BEZSON_PATHNAME = "bezier_path.json"

    # ================ Main process through all 6 ================

    for dataset in ["TUSIMPLE"]:

        this_dataloader = LoadDataEgoPath(
            labels_filepath = os.path.join(dataset_dir, dataset, JSON_PATHNAME),
            images_filepath = os.path.join(dataset_dir, dataset, IMAGES_DIRNAME),
            dataset = dataset
        )

        # Prep JSON
        bezson_dict = {}

        # Prep bezier_visualization
        bezier_dir = os.path.join(dataset_dir, dataset, BEZIER_DIRNAME)
        if (os.path.exists(bezier_dir)):
            shutil.rmtree(bezier_dir)
        os.makedirs(bezier_dir)

        invalid_count = 0

        # Bezier gen
        for is_train in [True, False]:

            if (is_train):
                split = "train"
                images = this_dataloader.train_images
            else:
                split = "val"
                images = this_dataloader.val_images
            print(f"\tCurrently processing {split} split...")

            # Thru each sample
            for i, img_path in enumerate(images):

                # Image ID
                img_id = img_path.split("/")[-1].replace(".png", "")

                img, bezier_points, is_valid = this_dataloader.getItem(i, is_train)
                h, w, _ = img.shape

                if (type(bezier_points) == int):
                    invalid_count += 1
                else:
                    # Acquire bezier curve
                    bezier_curve = []
                    for step in range(2, 10, 1):
                        t = step / 10
                        x, y = evaluate_bezier(bezier_points, t)
                        bezier_curve.append((x, y))
                    # Draw
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    pallete = {
                        "point_red": (255, 0, 0), 
                        "line_green": (0, 255, 0)
                    }
                    lane_w = 4

                    # Curve
                    draw.line(
                        [(int(x * w), int(y * h)) for (x, y) in bezier_curve], 
                        fill = pallete["line_green"], 
                        width = lane_w // 2
                    )

                    # Points
                    bezier_points = list(zip(
                        bezier_points[0 : : 2],
                        bezier_points[1 : : 2]
                    ))
                    for (x, y) in bezier_points:
                        x_deno = int(x * w)
                        y_deno = int(y * h)
                        draw.ellipse(
                            (
                                x_deno - lane_w, y_deno - lane_w, 
                                x_deno + lane_w, y_deno + lane_w
                            ),
                            fill = pallete["point_red"]
                        )
                    
                    # Save
                    img_pil.save(os.path.join(bezier_dir, f"{img_id}.jpg"))

                    # Add to JSON, but first convert all to float64 
                    # each 4-digit rounded to reduce JSON filesize
                    bezier_points = [
                        (round(float(p[0]), 4), round(float(p[1]), 4))
                        for p in bezier_points
                    ]
                    bezier_curve = [
                        (round(float(p[0]), 4), round(float(p[1]), 4))
                        for p in bezier_curve
                    ]
                    bezson_dict[img_id] = {
                        "points" : bezier_points,
                        "curve"  : bezier_curve
                    }
                    
            print(f"\tFinished processing {len(images)} samples of {split} split.")

        # Dump JSON
        bezson_path = os.path.join(dataset_dir, dataset, BEZSON_PATHNAME)
        with open(bezson_path, "w") as f:
            json.dump(
                bezson_dict, f, 
                indent = 4
            )
        print(f"\tFinished dumping JSON.")

        print(f"Finish bezier visualization for dataset {dataset}.")
        print(f"Number of invalid bezier fits: {invalid_count}\n")