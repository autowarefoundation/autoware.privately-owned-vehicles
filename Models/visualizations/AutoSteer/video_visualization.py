# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import cv2
import sys
import time
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime, timedelta

from torch import get_file_path

sys.path.append('../..')
from Models.inference.auto_steer_infer import AutoSpeedNetworkInfer


def rotate_wheel(wheel_img, angle_deg):
    """Rotate a PNG with alpha channel."""
    h, w = wheel_img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        wheel_img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return rotated


def overlay_on_top(base_img, rotated_wheel_img, frame_time, steering_angle, rotated_gt_wheel_img=None):
    """Put wheel image at the top center of the base image."""
    H, W = base_img.shape[:2]
    oh, ow = rotated_wheel_img.shape[:2]
    x = (W - ow - 60)
    y = 20  # small top margin

    # Copy base
    image = base_img.copy()

    # If overlay has alpha
    if rotated_wheel_img.shape[2] == 4:
        alpha = rotated_wheel_img[:, :, 3] / 255.0
        overlay_rgb = rotated_wheel_img[:, :, :3]

        # Blend
        for c in range(3):
            image[y:y + oh, x:x + ow, c] = (
                    overlay_rgb[:, :, c] * alpha +
                    image[y:y + oh, x:x + ow, c] * (1 - alpha)
            )

    else:
        # No alpha, hard paste
        image[y:y + oh, x:x + ow] = rotated_wheel_img

    if rotated_gt_wheel_img is not None:
        gt_oh, gt_ow = rotated_gt_wheel_img.shape[:2]
        gt_x = (W - ow - 208)
        gt_y = 20  # small top margin

        # If overlay has alpha
        if rotated_gt_wheel_img.shape[2] == 4:
            alpha = rotated_gt_wheel_img[:, :, 3] / 255.0
            overlay_rgb = rotated_gt_wheel_img[:, :, :3]

            # Blend
            for c in range(3):
                image[gt_y:gt_y + gt_oh, gt_x:gt_x + gt_ow, c] = (
                        overlay_rgb[:, :, c] * alpha +
                        image[gt_y:gt_y + gt_oh, gt_x:gt_x + gt_ow, c] * (1 - alpha)
                )

        else:
            # No alpha, hard paste
            image[gt_y:gt_y + gt_oh, gt_x:gt_x + gt_ow] = rotated_gt_wheel_img

    # -------- ADD TEXT HERE --------
    cv2.putText(
        image,
        frame_time,
        (x - 60, y + oh + 30),  # position (x,y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        0.6,  # scale
        (255, 255, 255),  # color (white)
        2,  # thickness
        cv2.LINE_AA  # anti-aliased
    )

    cv2.putText(
        image,
        f"{steering_angle:.2f} deg",
        (x - 60, y + oh + 60),  # position (x,y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        0.6,  # scale
        (255, 255, 255),  # color (white)
        2,  # thickness
        cv2.LINE_AA  # anti-aliased
    )

    return image


def load_ground_truth(gt_file_path):
    with open(gt_file_path, 'r') as f:
        data = json.load(f)
    return data


def make_visualization(frame, prediction):
    # Convert prediction to string
    text = f"Pred: {prediction:.2f}"

    # Put text on the frame
    cv2.putText(
        frame,  # image
        text,  # text to display
        (20, 40),  # position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        1,  # font scale
        (0, 255, 0),  # color (BGR)
        2,  # thickness
        cv2.LINE_AA  # line type
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--egolanes_checkpoint_path", dest="egolanes_checkpoint_path",
                        help="path to pytorch EgoLane scheckpoint file to load model dict")
    parser.add_argument("-a", "--autosteer_checkpoint_path", dest="autosteer_checkpoint_path",
                        help="path to pytorch AutoSteer checkpoint file to load model dict")
    parser.add_argument("-i", "--video_filepath", dest="video_filepath",
                        help="path to input video which will be processed by AutoSteer")
    parser.add_argument("-o", "--output_file", dest="output_file",
                        help="path to output video visualization file, must include output file name")
    parser.add_argument('-v', "--vis", action='store_true', default=False,
                        help="flag for whether to show frame by frame visualization while processing is occuring")
    parser.add_argument('-g', "--ground_truth",
                        help="json file containing ground truth steering angles for each frame")
    args = parser.parse_args()

    # Saved model checkpoint path
    egolanes_checkpoint_path = args.egolanes_checkpoint_path
    autosteer_checkpoint_path = args.autosteer_checkpoint_path
    model = AutoSpeedNetworkInfer(egolanes_checkpoint_path=egolanes_checkpoint_path,
                                  autosteer_checkpoint_path=autosteer_checkpoint_path)
    print('AutoSteer Model Loaded')

    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = args.video_filepath
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_datetime = datetime.now()  # or read metadata if available

    # Wheel image
    wheel_img_path = "../../../Media/wheel.png"
    gt_wheel_img_path = "../../../Media/wheel_green.png"
    # Load wheel image (use PNG with transparency)
    wheel_raw = cv2.imread(wheel_img_path, cv2.IMREAD_UNCHANGED)
    scaled_wheel = cv2.resize(wheel_raw, None, fx=0.8, fy=0.8)

    gt_wheel_raw = cv2.imread(gt_wheel_img_path, cv2.IMREAD_UNCHANGED)
    scaled_gt_wheel = cv2.resize(gt_wheel_raw, None, fx=0.8, fy=0.8)

    # Output filepath
    output_filepath_obj = args.output_file + '.avi'
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Video writer object
    writer_obj = cv2.VideoWriter(output_filepath_obj,
                                 cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))

    # Ground Truth file path
    gt_file_path = args.ground_truth
    gt = None
    if gt_file_path is not None:
        gt = load_ground_truth(gt_file_path)

    # Check if video catpure opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    else:
        print('Reading video frames')

    # Transparency factor
    alpha = 0.5

    # Read until video is completed
    print('Processing started')
    frame_index = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((640, 320))

            # Running inference
            steering_angle = model.inference(image_pil)

            # --- Rotate wheel image ---
            rotated_wheel_img = rotate_wheel(scaled_wheel, steering_angle)
            rotated_gt_wheel_img = None

            # --- GT wheel image ---
            if gt is not None:
                gt_angle = gt["frames"][frame_index]["steering_angle_corrected"]
                rotated_gt_wheel_img = rotate_wheel(scaled_gt_wheel, gt_angle)

            # Resizing to match the size of the output video
            # which is set to standard HD resolution
            frame = cv2.resize(frame, (1280, 720))
            # make_visualization(frame, prediction)
            frame_time = start_datetime + timedelta(seconds=frame_index / fps)
            date_time = frame_time.strftime("%m/%d/%Y %I:%M:%S")
            frame = overlay_on_top(frame, rotated_wheel_img, date_time, steering_angle, rotated_gt_wheel_img)

            if (args.vis):
                cv2.imshow('Prediction Objects', frame)
                cv2.waitKey(10)

            # Writing to video frame
            writer_obj.write(frame)

        else:
            print('Frame not read - ending processing')
            break
        frame_index += 1

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
# %%
