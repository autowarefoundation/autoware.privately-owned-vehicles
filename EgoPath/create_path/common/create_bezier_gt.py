import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    "../../../"
)))
from Models.data_utils.load_data_ego_path import (
    LoadDataEgoPath, 
    VALID_DATASET_LIST, 
    VALID_DATASET_LITERALS
)

# Evaluate the x, y coords of a bezier point list at a given t-param value
def evaluate_bezier(self, bezier, t):

        # Throw an error if parameter t is out of boudns
        if not (0 < t < 1):
            raise ValueError("Please ensure t parameter is in the range [0, 1]")
        
        # Evaluate cubic bezier curve for value of t in range [0, 1]
        x = ((1-t)**3)*bezier[0][0] + 3*((1-t)**2)*t*bezier[0][2] \
            + 3*(1-t)*(t**2)*bezier[0][4] + (t**3)*bezier[0][6]
        
        y = ((1-t)**3)*bezier[0][1] + 3*((1-t)**2)*t*bezier[0][3] \
            + 3*(1-t)*(t**2)*bezier[0][5] + (t**3)*bezier[0][7]
        
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

    IMAGES_DIR = "image"
    JSON_PATH = "drivable_path.json"
    BEZIER_DIR = "bezier-visualization"
    BEZSON_PATH = "bezier_path.json"

    # ================ Main process through all 6 ================

    for dataset in VALID_DATASET_LIST:

        this_dataloader = LoadDataEgoPath(
            labels_filepath = os.path.join(dataset_dir, dataset, JSON_PATH),
            images_filepath = os.path.join(dataset_dir, dataset, IMAGES_DIR),
            dataset = dataset
        )

        # Merge back tran and val splits
        # this_all_images = (this_dataloader.train_images + this_dataloader.val_images).sort()
        # this_all_labels = (this_dataloader.train_labels + this_dataloader.val_labels).sort()

        # Bezier gen
        for is_train in [True, False]:

            if (is_train):
                split = "train"
                N_samples = this_dataloader.N_trains
            else:
                split = "val"
                N_samples = this_dataloader.N_vals
            print(f"\tCurrent processing {split} split")

            for i in range(N_samples):

                img, bezier_points, is_valid = this_dataloader.getItem(i, is_train)
                
                for step in range(2, 11):
                    t = step / 10
