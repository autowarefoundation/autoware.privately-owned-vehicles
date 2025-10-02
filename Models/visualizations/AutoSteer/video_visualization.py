import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from argparse import ArgumentParser
sys.path.append('../..')
from inference.auto_steer_infer import AutoSteerNetworkInfer
from image_visualization import make_visualization

FRAME_INF_SIZE = (640, 320)
FRAME_ORI_SIZE = (1280, 720)


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path", 
        help = "Path to Pytorch checkpoint file to load model dict"
    )
    parser.add_argument(
        "-i", 
        "--input_video_filepath", 
        dest = "input_video_filepath", 
        help = "Path to input video which will be processed by AutoSteer"
    )
    parser.add_argument(
        "-o", 
        "--output_video_dir", 
        dest = "output_video_dir", 
        help = "Path to output video directory where the output videos will be saved"
    )
    args = parser.parse_args()

    if (not os.path.exists(args.output_video_dir)):
        os.makedirs(args.output_video_dir)

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = AutoSteerNetworkInfer(
        checkpoint_path = model_checkpoint_path
    )
    print("AutoSteer model successfully loaded!")

    # Fetch video file
    print("Reading video")
    cap = cv2.VideoCapture(args.input_video_filepath)

    # Pre-prep output
    if (not cap.isOpened()):
        print("Error opening video stream or file")
        return
    output_filepath = os.path.join(
        args.output_video_dir,
        "output_video_data.avi"
    )
    writer = cv2.VideoWriter(
        output_filepath,
        cv2.VideoWriter_fourcc(*"MJPG"), 
        cap.get(cv2.CAP_PROP_FPS),
        FRAME_ORI_SIZE
    )
    
    # Read video frame-by-frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(
        range(frame_count),
        desc = "Processing video frames: ",
        unit = "frames",
        colour = "green"
    ):

        ret, frame = cap.read()
        if (not ret):
            print(f"Frame {i} could not be read, skipped.")
            continue

        # Fetch frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize(FRAME_INF_SIZE)

        # Inference
        prediction = model.inference(image_pil)
        vis_image = make_visualization(image, prediction)

        # Resize
        vis_image = cv2.resize(vis_image, FRAME_ORI_SIZE)

        # Write to output
        writer.write(vis_image)

    # Release resources
    cap.release()
    writer.release()
    print(f"Output video saved to: {output_filepath}")


if (__name__ == "__main__"):
    main()