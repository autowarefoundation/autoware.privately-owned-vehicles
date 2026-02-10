import os
import sys
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('../..')
from inference.ego_lanes_infer import EgoLanesNetworkInfer

    
def make_visualization(
        image: np.ndarray,
        prediction: np.ndarray,
        alpha: float = 0.5
):
    
    # Compute scale from prediction to image
    img_h, img_w = image.shape[:2]
    _, ph, pw = prediction.shape
    scale_x = img_w / pw
    scale_y = img_h / ph

    # Prep img for drawing (convert to BGR for OpenCV)
    img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # Fetch coords (y, x) for each channel
    pred_coords = [
        np.where(prediction[c] > 0)
        for c in range(prediction.shape[0])
    ]

    # Choose radius based on scale
    base_scale = min(scale_x, scale_y)
    radius = max(1, int(round(base_scale * 0.5)))

    # Color codes (RGB)
    colors = [
        (0, 72, 255),       # Blue      (ego left)
        (200, 0, 255),      # Magenta   (ego right)
        (0, 153, 0),        # Green     (other lanes)
    ]

    overlay = img_bgr.copy()

    # Draw
    for i, (ys, xs) in enumerate(pred_coords):
        if ys.size == 0:
            continue

        # Scale and clip to image bounds
        xs_scaled = (xs.astype(np.float32) * scale_x).astype(np.int32)
        ys_scaled = (ys.astype(np.float32) * scale_y).astype(np.int32)
        xs_scaled = np.clip(xs_scaled, 0, img_w - 1)
        ys_scaled = np.clip(ys_scaled, 0, img_h - 1)

        # Draw dots
        color = colors[i] if i < len(colors) else (255,255,255)
        for x, y in zip(xs_scaled, ys_scaled):
            cv2.circle(
                img_bgr, 
                (int(x), int(y)), 
                radius, color, thickness = -1, 
                lineType=cv2.LINE_AA
            )

    # Back to PIL Image
    vis_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

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
    model = EgoLanesNetworkInfer(
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