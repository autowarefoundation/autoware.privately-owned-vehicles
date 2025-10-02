import cv2
import sys
import math
import numpy as np
from PIL import ImageDraw
from argparse import ArgumentParser
sys.path.append('../..')
from inference.auto_steer_infer import AutoSteerNetworkInfer

    
def make_visualization(
        image: np.ndarray,
        prediction: np.ndarray
):

    # Fetch predictions + calculations
    left_lane_offset = prediction[0] * 640
    right_left_offset = prediction[1] * 640
    ego_path_offset = prediction[2] * 640
    start_angle = prediction[3]
    start_delta_x = ego_path_offset + 100 * math.sin(start_angle)
    start_delta_y = 319 - (100 * math.cos(start_angle))
    end_angle = prediction[4]
    end_point_x = prediction[5] * 640
    end_point_y = prediction[6] * 320
    end_delta_x = end_point_x - 30 * math.sin(end_angle)
    end_delta_y = end_point_y + 30 * math.cos(end_angle)

    # Start drawing
    draw = ImageDraw.Draw(image)
    POINT_R = 3
    LINE_W = 2
    DOWN_MARGIN = 310
    COLOR_OFFSET = (255, 0, 0)      # Blue
    COLOR_EGOPATH = (255, 255, 0)   # Yellow
    COLOR_END = (255, 0, 255)       # Red

    # Offsets
    draw.ellipse(
        (
            left_lane_offset - POINT_R, 
            DOWN_MARGIN - POINT_R, 
            left_lane_offset + POINT_R, 
            DOWN_MARGIN + POINT_R
        ), 
        fill = COLOR_OFFSET
    )
    draw.ellipse(
        (
            right_left_offset - POINT_R, 
            DOWN_MARGIN - POINT_R, 
            right_left_offset + POINT_R, 
            DOWN_MARGIN + POINT_R
        ), 
        fill = COLOR_OFFSET
    )
    draw.line(
        (
            left_lane_offset, DOWN_MARGIN, 
            right_left_offset, DOWN_MARGIN
        ),
        fill = COLOR_OFFSET,
        width = LINE_W
    )

    # Ego path
    draw.ellipse(
        (
            ego_path_offset - POINT_R, 
            DOWN_MARGIN - POINT_R, 
            ego_path_offset + POINT_R, 
            DOWN_MARGIN + POINT_R
        ), 
        fill = COLOR_EGOPATH
    )
    draw.line(
        (
            ego_path_offset, DOWN_MARGIN, 
            start_delta_x, start_delta_y
        ),
        fill = COLOR_EGOPATH,
        width = LINE_W
    )

    # End point
    draw.ellipse(
        (
            end_point_x - POINT_R, 
            end_point_y - POINT_R, 
            end_point_x + POINT_R, 
            end_point_y + POINT_R
        ), 
        fill = COLOR_END
    )
    draw.line(
        (
            end_point_x, end_point_y, 
            end_delta_x, end_delta_y
        ),
        fill = COLOR_END,
        width = LINE_W
    )

    # Return visualized image
    vis_image = np.array(image)
    return vis_image