import cv2
import sys
import os
import numpy as np
from PIL import Image


def load_image(input_folder, image_file, resize=(640, 320)):
    """
        Preprocessing input folder + resize image to desired size.
    """
    image_path = os.path.join(input_folder, image_file)
    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[WARNING] Skipping unreadable image: {image_file}")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize(resize)

    return image_pil, frame