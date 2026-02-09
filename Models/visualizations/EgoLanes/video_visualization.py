import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from argparse import ArgumentParser
sys.path.append('../..')
from inference.ego_lanes_infer import EgoLanesNetworkInfer
from image_visualization import make_visualization

FRAME_INF_SIZE = (640, 320)
FRAME_ORI_SIZE = (720, 360)


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
        "--output_video_path", 
        dest = "output_video_path", 
        help = "Path to output video where the output video will be saved"
    )
    parser.add_argument(
        '-v', "--vis",
        action='store_true',
        default=False,
        help="flag for whether to show frame by frame visualization while processing is occurring"
    )
    args = parser.parse_args()

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = EgoLanesNetworkInfer(
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
    
    # Visualization preparation
    output_filepath_data = args.output_video_path
    writer_data = cv2.VideoWriter(
        output_filepath_data,
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
        image = np.array(image)

        # Inference
        prediction = model.inference(image)

        # Frame preprocessing
        vis_image_data = make_visualization(image.copy(), prediction)
        vis_image_data = np.array(vis_image_data)
        vis_image_data = cv2.resize(vis_image_data, FRAME_ORI_SIZE)

        # Displaying the visualization
        if args.vis:
            cv2.imshow('EgoLanes Prediction', vis_image_data)
            cv2.waitKey(10)

        # Writing to video frame
        writer_data.write(vis_image_data)

    # Release resources
    cap.release()
    writer_data.release()
    cv2.destroyAllWindows()
    print(f"Visualization video saved to: {output_filepath_data}")


if (__name__ == "__main__"):
    main()