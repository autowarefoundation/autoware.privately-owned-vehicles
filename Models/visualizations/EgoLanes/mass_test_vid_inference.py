import os
import subprocess
from tqdm import tqdm

BEST_WEIGHT_PATH = "<path to best model weights file>"
TEST_VID_DIR = "<test video directory path, containing various video files>"
OUTPUT_VID_DIR = "<output directory for multiple output videos>"
if (not os.path.exists(OUTPUT_VID_DIR)):
    os.makedirs(OUTPUT_VID_DIR)

for vid_file in tqdm(
    sorted(os.listdir(TEST_VID_DIR)),
    colour = "green"
):
    if vid_file.endswith(".mp4"):
        
        vid_id = vid_file.split(".")[0]
        input_vid_path = os.path.join(TEST_VID_DIR, vid_file)
        output_vid_dir = os.path.join(OUTPUT_VID_DIR, vid_id + ".avi")

        command = [
            "uv", "run", "python3",
            "video_visualization.py",
            "-i", input_vid_path,
            "-o", output_vid_dir,
            "-p", BEST_WEIGHT_PATH
        ]

        result = subprocess.run(
            command,
            check = True
        )