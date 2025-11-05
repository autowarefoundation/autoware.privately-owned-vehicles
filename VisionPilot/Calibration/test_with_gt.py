import os
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import itertools
import yaml
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

# Enable eager execution for TensorFlow
tf.enable_eager_execution()

def get_manual_bbox(camera_image, points_all, cp_points_all):
    """Displays an image and allows the user to manually select a bounding box."""
    
    print("\n--- Manual Bounding Box Selection ---")
    print("Please click on the image to select a bounding box around a target vehicle.")
    print("1. Click the TOP-LEFT corner of the vehicle.")
    print("2. Click the BOTTOM-RIGHT corner of the vehicle.")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    img = tf.image.decode_jpeg(camera_image.image)
    img_np = img.numpy()  # Convert to NumPy array if it's a Tensor

    height, width = img_np.shape[:2]

    # print(f"max u (width - 1): {width - 1}")
    # print(f"max v (height - 1): {height - 1}")
    ax.imshow(img)
    ax.set_title("Click TOP-LEFT, then BOTTOM-RIGHT to define BBox", fontsize=16)
    plt.grid(False)
    
    # Wait for the user to make two clicks
    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(points) < 2:
        print("BBox selection cancelled or timed out. Exiting.")
        return None

    (u1, v1), (u2, v2) = points
    print(f"u1: {u1}, v1: {v1}")
    print(f"u2: {u2}, v2: {v2}")
    
    # Top-left is always smaller than bottom-right when clicked correctly
    # x1, y1 = top-left, x2, y2 = bottom-right
    BBOX = [int(u1), int(v1), int(u2), int(v2)]
    print(f"Manual BBox selected: {BBOX}")
    
    return points, BBOX

def load_homography_from_yaml(filename="homography.yaml"):
    """Loads a homography matrix from a YAML file."""
    try:
        with open(filename, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
            rows = yaml_data['H']['rows']
            cols = yaml_data['H']['cols']
            data = yaml_data['H']['data']
            
            H = np.array(data).reshape((rows, cols))
            print(f"Successfully loaded {filename}")
            return H
            
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or parse {filename}. Please run compute_homography.py first.")
        print(e)
        return None

def main(args):
    # --- 1. Setup ---
    # Load the pre-computed homography matrix from YAML
    H = load_homography_from_yaml()
    if H is None:
        return

    # Use the same data file as the homography calculation for a fair test
    FILENAME = args.filename
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    
    # --- Loop through the specified number of frames ---
    for frame_index, data in enumerate(itertools.islice(dataset, args.num_frames)):
        print(f"\n--- Processing Frame {frame_index + 1}/{args.num_frames} ---")

        # --- 2. Load Data ---
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame=frame, range_images=range_images, camera_projections=camera_projections, range_image_top_pose=range_image_top_pose
        )
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)
        images = sorted(frame.images, key=lambda i: i.name)
        front_camera_image = images[0]

        # --- 3. Interactive Bounding Box for Each Frame ---
        points, bbox = get_manual_bbox(front_camera_image, points_all, cp_points_all)
        if points is None:
            # Allow skipping frames or exiting
            user_input = input("BBox selection skipped. Type 'exit' to stop or press Enter to continue to the next frame: ")
            if user_input.lower() == 'exit':
                break
            continue

        # --- 4. Estimate Distance with Homography ---
        # Get the bottom-center point of the bounding box (where car touches ground)
        (u1, v1), (u2, v2) = points
        u = (u1 + u2) / 2 # Horizontal center
        v = v2 # Bottom edge
        print(f"Bottom-center UV: u={u}, v={v}")
        
        # Transform this 2D point to 3D world coordinates using the homography
        uv_point = np.float32([u, v]).reshape(-1, 1, 2)
        xy_predicted = cv2.perspectiveTransform(uv_point, H).reshape(2)
        
        print(f"Predicted world XY: {xy_predicted}")
        estimated_distance = xy_predicted[0]
        print(f"-> Homography Estimated Distance: {estimated_distance:.4f} meters")

        # --- 5. Find Ground Truth Distance with LiDAR ---
        # Find all LiDAR points that project into the front camera's view
        mask_cam = (cp_points_all[:, 0] == front_camera_image.name)
        cp_points_front = cp_points_all[mask_cam]
        points_front = points_all[mask_cam]

        # Find all of those points that fall inside the bounding box
        # cp_points_front[:, 1] is u (x in image), cp_points_front[:, 2] is v (y in image)
        mask_bbox = (cp_points_front[:, 1] >= bbox[0]) & \
                    (cp_points_front[:, 1] <= bbox[2]) & \
                    (cp_points_front[:, 2] >= bbox[1]) & \
                    (cp_points_front[:, 2] <= bbox[3])
        
        points_in_bbox = points_front[mask_bbox]

        if len(points_in_bbox) == 0:
            print("Could not find any LiDAR points inside the selected bounding box.")
            continue

        print(f"Found {len(points_in_bbox)} LiDAR points in bbox")
        
        # Drop all points to ground plane (z=0) and calculate XY distance only
        points_xy = points_in_bbox[:, :2]  # Take only X and Y, drop Z
        distances_in_bbox = np.linalg.norm(points_xy, axis=1)
        
        # The ground truth is the closest point on the car to us (in XY plane)
        gt_distance = np.min(distances_in_bbox)
        mean_distance = np.mean(distances_in_bbox)
        print(f"-> LiDAR Ground Truth Distance (min): {gt_distance:.4f} meters")
        print(f"-> LiDAR Ground Truth Distance (mean): {mean_distance:.4f} meters")

        # --- 6. Compare ---
        error = abs(estimated_distance - gt_distance)
        print(f"   Absolute Error: {error:.4f} meters")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test homography matrix accuracy on a sequence of frames.")
    parser.add_argument('--filename', type=str, 
                        default='/home/pranavdoma/Downloads/waymo/segment-10153695247769592104_787_000_807_000_with_camera_labels.tfrecord',
                        help='Path to the Waymo TFRecord file.')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames to process for the test.')
    args = parser.parse_args()
    main(args)
