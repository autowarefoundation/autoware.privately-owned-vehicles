import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
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

    # == Input args ==

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

    # == Preprocess ==

    list_subdirs = ["image", "segmentation", "visualization"]
    json_path = "drivable_path.json"