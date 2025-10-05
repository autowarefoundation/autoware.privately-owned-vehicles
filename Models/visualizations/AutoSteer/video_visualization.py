import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from argparse import ArgumentParser
sys.path.append('../..')
from inference.auto_steer_infer import AutoSteerNetworkInfer
from image_visualization import make_visualization_data, make_visualization_seg

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
    
    # For data visualization
    output_filepath_data = os.path.join(
        args.output_video_dir,
        "output_video_data.avi"
    )
    writer_data = cv2.VideoWriter(
        output_filepath_data,
        cv2.VideoWriter_fourcc(*"MJPG"), 
        cap.get(cv2.CAP_PROP_FPS),
        FRAME_ORI_SIZE
    )
    
    # For segmentation visualization
    output_filepath_seg = os.path.join(
        args.output_video_dir,
        "output_video_seg.avi"
    )
    writer_seg = cv2.VideoWriter(
        output_filepath_seg,
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
        image = Image.fromarray(frame)
        image = image.resize(FRAME_INF_SIZE)

        # Inference
        seg_pred, data_pred = model.inference(image)

        # Frame preprocessing
        vis_image_data = make_visualization_data(image.copy(), data_pred)
        vis_image_data = np.array(vis_image_data)
        vis_image_data = cv2.resize(vis_image_data, FRAME_ORI_SIZE)

        vis_image_seg = make_visualization_seg(image.copy(), seg_pred)
        vis_image_seg = np.array(vis_image_seg)
        vis_image_seg = cv2.resize(vis_image_seg, FRAME_ORI_SIZE)

        # Write to outputs
        writer_data.write(vis_image_data)
        writer_seg.write(vis_image_seg)

    # Release resources
    cap.release()
    writer_data.release()
    writer_seg.release()
    print(f"Data         visualization video saved to: {output_filepath_data}")
    print(f"Segmentation visualization video saved to: {output_filepath_seg}")


if (__name__ == "__main__"):
    main()