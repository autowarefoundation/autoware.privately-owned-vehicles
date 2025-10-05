import os
import sys
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw
from argparse import ArgumentParser
sys.path.append('../..')
from inference.auto_steer_infer import AutoSteerNetworkInfer

    
def make_visualization_data(
        image: Image,
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
    COLOR_OFFSET = (0, 255, 255)    # Cyan
    COLOR_EGOPATH = (255, 255, 0)   # Yellow
    COLOR_END = (255, 0, 0)         # Red

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
    return image


def make_visualization_seg(
        image: Image,
        prediction: np.ndarray
):
    
    # Creating visualization object
    shape = prediction.shape
    row = shape[0]
    col = shape[1]
    seg_pred_object = np.array(
        np.zeros(
            (row, col, 3), 
            dtype = "uint8"
        )
    )

    # Foreground object labels
    pred_labels = np.where(prediction > 0)

    # Visualization
    COLOR_SEG = (0, 255, 145)       # Lil lime green
    for i in range(3):
        seg_pred_object[
            pred_labels[0],
            pred_labels[1], 
            i
        ] = COLOR_SEG[i]
    
    # Alpha blending
    alpha = 0.5
    pred_vis = cv2.addWeighted(
        seg_pred_object, 
        alpha, 
        np.array(image), 
        1,
        0
    )

    vis_image = Image.fromarray(pred_vis)
    return vis_image


def main(): 

    parser = ArgumentParser()
    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path", 
        help = "Path to Pytorch checkpoint file to load model dict",
        required = False
    )
    parser.add_argument(
        "-i", 
        "--input_image_dirpath", 
        dest = "input_image_dirpath", 
        help = "Path to input image directory which will be processed by AutoSteer",
        required = True
    )
    parser.add_argument(
        "-o",
        "--output_image_dirpath",
        dest = "output_image_dirpath",
        help = "Path to output image directory where visualizations will be saved",
        required = True
    )
    args = parser.parse_args()

    input_image_dirpath = args.input_image_dirpath
    output_image_dirpath = args.output_image_dirpath
    if (not os.path.exists(output_image_dirpath)):
        os.makedirs(output_image_dirpath)

    # Saved model checkpoint path
    model_checkpoint_path = (
        args.model_checkpoint_path 
        if args.model_checkpoint_path is not None 
        else ""
    )
    model = AutoSteerNetworkInfer(
        checkpoint_path = model_checkpoint_path
    )
    print("AutoSteer model successfully loaded!")

    # Process through input image dir
    for filename in sorted(os.listdir(input_image_dirpath)):
        if (filename.endswith((".png", ".jpg", ".jpeg"))):

            # Fetch image
            input_image_filepath = os.path.join(
                input_image_dirpath, filename
            )
            img_id = filename.split(".")[0].zfill(3)

            print(f"Reading Image: {input_image_filepath}")
            image = Image.open(input_image_filepath).convert("RGB")
            image = image.resize((640, 320))

            # Inference
            binary_segg_pred, path_data_pred = model.inference(image)

            # Data visualization
            vis_image = make_visualization_data(
                image.copy(), 
                path_data_pred
            )
            
            output_image_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_data.png"
            )
            vis_image.save(output_image_filepath)

            # Segmentation visualization
            vis_seg = make_visualization_seg(
                image.copy(), 
                binary_segg_pred
            )

            output_seg_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_seg.png"
            )
            vis_seg.save(output_seg_filepath)

        else:
            print(f"Skipping non-image file: {filename}")
            continue


if __name__ == "__main__":
    main()