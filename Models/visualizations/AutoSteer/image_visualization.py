import os
import sys
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.auto_steer_infer import AutoSteerNetworkInfer

    
def make_visualization(
        image: np.ndarray,
        prediction: np.ndarray
):
    
    # Prepping canvas
    vis_predict_object = np.zeros((320, 640, 3), dtype = "uint8")

    # Fetch predictions and groundtruth labels
    pred_egoleft_lanes  = np.where(prediction[0,:,:] > 0)
    pred_egoright_lanes = np.where(prediction[1,:,:] > 0)
    pred_other_lanes    = np.where(prediction[2,:,:] > 0)

    # Color codes
    egoleft_color   = [0, 255, 255]
    egoright_color  = [255, 0, 200]
    others_color    = [0, 255, 145]

    # Visualize egoleft
    for i in range(3):
        vis_predict_object[
            pred_egoleft_lanes[0], 
            pred_egoleft_lanes[1], 
            i
        ] = egoleft_color[i]

    # Visualize egoright
    for i in range(3):
        vis_predict_object[
            pred_egoright_lanes[0],
            pred_egoright_lanes[1],
            i
        ] = egoright_color[i]

    # Visualize other lanes
    for i in range(3):
        vis_predict_object[
            pred_other_lanes[0],
            pred_other_lanes[1],
            i
        ] = others_color[i]

    # Fuse image with mask
    alpha = 0.5
    fused_image = cv2.addWeighted(
        vis_predict_object, alpha,
        image, 1,
        0
    )
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB)
    fused_image = Image.fromarray(fused_image)

    return fused_image


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
            image = np.array(image)

            # Inference
            prediction = model.inference(image)

            # Visualization
            vis_image = make_visualization(
                image.copy(), 
                prediction
            )
            
            output_image_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_data.png"
            )
            vis_image.save(output_image_filepath)

        else:
            print(f"Skipping non-image file: {filename}")
            continue


if __name__ == "__main__":
    main()