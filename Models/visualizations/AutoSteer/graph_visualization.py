# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import cv2
import sys
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

sys.path.append('../..')
from Models.inference.auto_steer_infer import AutoSpeedNetworkInfer

np.Inf = np.inf


def load_ground_truth(gt_file_path):
    with open(gt_file_path, 'r') as f:
        data = json.load(f)
    return data


def visualize_graph(gt_angles, prediction_angles, output_file=None, vis=False):
    x = np.arange(len(gt_angles))  # sample indices, 0..n-1

    plt.figure(figsize=(12, 4))
    # Plot GT
    plt.plot(x, gt_angles, label="Ground Truth", color="green", linewidth=2)

    # Plot Prediction
    plt.plot(x, prediction_angles, label="Prediction", color="blue", linewidth=2, linestyle="--")

    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title("Ground Truth/Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if save_path is provided
    if output_file is not None:
        plt.savefig(output_file, dpi=300)  # high-quality PNG
        print(f"Plot saved to: {output_file}")
    if vis:
        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--egolanes_checkpoint_path", dest="egolanes_checkpoint_path",
                        help="path to pytorch EgoLane scheckpoint file to load model dict")
    parser.add_argument("-a", "--autosteer_checkpoint_path", dest="autosteer_checkpoint_path",
                        help="path to pytorch AutoSteer checkpoint file to load model dict")
    parser.add_argument("-i", "--input_path", dest="input_path",
                        help="path to input video which will be processed by AutoSteer")
    parser.add_argument("-o", "--output_file", dest="output_file",
                        help="path to output graph visualization file, must include output file name and extension")
    parser.add_argument('-v', "--vis", action='store_true', default=False,
                        help="flag for whether to show grap")
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
    input_path = Path(args.input_path)
    output_file = args.output_file
    vis = args.vis

    # Ground Truth file path
    gt_file_path = args.ground_truth
    gt = None
    if gt_file_path is not None:
        gt = load_ground_truth(gt_file_path)

    # Read until video is completed
    print('Processing started')

    gt_angles = []
    prediction_angles = []
    for i, img_path in enumerate(sorted(input_path.iterdir())):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            frame = Image.open(img_path)
            w, h = frame.size  # (2880, 1860)
            crop_top = h - 1440  # 420
            frame = frame.crop((0, crop_top, w, h))
            frame = frame.resize((640, 320), Image.LANCZOS)
            # frame.show()
            gt_angle = gt["frames"][i]["steering_angle_corrected"]
            gt_angles.append(gt_angle)

            steering_angle = model.inference(frame)
            prediction_angles.append(steering_angle)

    visualize_graph(gt_angles, prediction_angles, output_file, vis)

    print('Completed')


if __name__ == '__main__':
    main()
# %%
