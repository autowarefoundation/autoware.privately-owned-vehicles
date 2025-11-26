import os
import subprocess
from tqdm import tqdm

BEST_WEIGHT_PATH = "/home/tranhuunhathuy/Documents/Autoware/autoware.privately-owned-vehicles/Models/saves/AutoSteer/models/iter_301692_epoch_3_step_15000.pth"
TEST_VID_DIR = "/mnt/Storage/pov_datasets/TEST_VIDEOS/READY/"
OUTPUT_VID_DIR = "/mnt/Storage/pov_datasets/TEST_VIDEOS/RESULTS/"
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