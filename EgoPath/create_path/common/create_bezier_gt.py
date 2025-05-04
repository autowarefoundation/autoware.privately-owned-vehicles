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

                img, bezier_curve, is_valid = this_dataloader.getItem(i, is_train)
                bezier_curve = list(zip(
                    bezier_curve[0 : : 2],
                    bezier_curve[1 : : 2]
                ))
                print(bezier_curve)