import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
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
    