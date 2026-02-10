import cv2
import sys
import os
import numpy as np
from PIL import Image
import cmapy



def mask_segmentation(prediction):
    """
    Generate RGB visualization from prediction mask.
    Background = orange, class 1 = purple, class 2 = green.
    """
    shape = prediction.shape
    vis_predict_object = np.zeros((shape[0], shape[1], 3), dtype="uint8")

    # Default background → orange
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    # Class 1 (object) → purple
    fg = np.where(prediction == 1)
    vis_predict_object[fg[0], fg[1], :] = (145, 28, 255)

    # Class 2 (road/drivable surface) → green
    road = np.where(prediction == 2)
    vis_predict_object[road[0], road[1], :] = (0, 255, 0)

    return vis_predict_object


def add_mask_segmentation(input_frame, prediction, alpha):
    """
    Overlay segmentation mask on input frame with given alpha transparency.
    Target size = prediction size
    """
    mask = mask_segmentation(prediction)

    # Resize input frame to prediction shape
    input_frame_resized = cv2.resize(input_frame, (mask.shape[1], mask.shape[0]))

    # Blend
    output_frame = cv2.addWeighted(mask, alpha, input_frame_resized, 1 - alpha, 0)
    return output_frame


def visualize_scene3d(input_frame, prediction, alpha):
    """
    Create output image for scene 3d depth estimation
    Target size = prediction size
    """
    # Normalize prediction to [0, 255]
    prediction_image = 255.0 * (
        (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction) + 1e-8)
    )
    prediction_image = prediction_image.astype(np.uint8)

    # Apply colormap → (H, W, 3)
    prediction_image = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))

    # Resize input frame to match prediction
    if input_frame.shape[:2] != prediction_image.shape[:2]:
        input_frame = cv2.resize(input_frame, (prediction_image.shape[1], prediction_image.shape[0]))

    # Ensure input is 3 channels
    if len(input_frame.shape) == 2:
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)

    # Blend
    output_frame = cv2.addWeighted(prediction_image, alpha, input_frame, 1 - alpha, 0)
    return output_frame
