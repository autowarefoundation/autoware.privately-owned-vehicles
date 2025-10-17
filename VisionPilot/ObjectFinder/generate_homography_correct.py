import numpy as np
import json
import cv2
import os
import glob
from pathlib import Path
import yaml

def transform_points_to_road_coord(lanes_xyz, extrinsic):
    # This function replicates the exact coordinate transformation from eval_3D_lane.py
    # to convert points from the Waymo camera coordinate system to the final road coordinate system.

    # Transformation matrices taken directly from the evaluation script
    R_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    R_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    cam_representation = np.linalg.inv(
                                np.array([[0, 0, 1, 0],
                                          [-1, 0, 0, 0],
                                          [0, -1, 0, 0],
                                          [0, 0, 0, 1]], dtype=float))

    # Apply the series of matrix multiplications from the script (lines 315-321 and 345)
    E = extrinsic.copy()
    E[:3, :3] = np.matmul(np.matmul(
                            np.matmul(np.linalg.inv(R_vg), E[:3, :3]),
                                R_vg), R_gc)
    E[0:2, 3] = 0.0

    transformed_lanes_xyz = []
    for lane_xyz in lanes_xyz:
        lane_xyz = np.array(lane_xyz)
        # Add a row of 1s to make the points homogeneous
        lane_xyz_hom = np.vstack((lane_xyz, np.ones((1, lane_xyz.shape[1]))))
        
        # Apply the full transformation
        lane_road_coord_hom = np.matmul(E, np.matmul(cam_representation, lane_xyz_hom))
        
        # Convert back to 3D and transpose for easier handling
        lane_road_coord = lane_road_coord_hom[0:3, :].T
        transformed_lanes_xyz.append(lane_road_coord)
        
    return transformed_lanes_xyz

def main(dataset_dir, output_dir):
    print(f"Searching for annotation files in: {dataset_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    json_files = glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True)
    print(f"Found {len(json_files)} annotation files.")

    if not json_files:
        print("Error: No JSON files found. Please check the dataset_dir path.")
        return

    all_homographies = []

    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)

        lane_lines = data.get('lane_lines', [])
        if not lane_lines:
            print(f"  -> Skipping, no lane_lines found.")
            continue

        # 1. Extract original data
        extrinsic = np.array(data['extrinsic'])
        lanes_uv_original = [np.array(lane['uv']).T for lane in lane_lines]
        lanes_xyz_original = [lane['xyz'] for lane in lane_lines]
        lanes_visibility = [lane['visibility'] for lane in lane_lines]
        
        # 2. Perform the coordinate transformation on 3D points
        lanes_xyz_road = transform_points_to_road_coord(lanes_xyz_original, extrinsic)
        
        # 3. Filter and create correspondence pairs
        image_points = []
        world_points = []
        
        for lane_uv, lane_xyz, visibility in zip(lanes_uv_original, lanes_xyz_road, lanes_visibility):
            visibility = np.array(visibility)
            
            # The visibility array might be longer than the uv/xyz arrays. Trim it.
            min_len = min(len(visibility), len(lane_uv), len(lane_xyz))
            visibility = visibility[:min_len]
            lane_uv = lane_uv[:min_len]
            lane_xyz = lane_xyz[:min_len]

            visible_indices = visibility > 0
            
            visible_uv = lane_uv[visible_indices]
            visible_xyz = lane_xyz[visible_indices]
            
            image_points.extend(visible_uv)
            # Use only (x, y) for the ground plane homography
            world_points.extend(visible_xyz[:, :2])
            
        if len(image_points) < 4:
            print(f"  -> Skipping, not enough visible points ({len(image_points)}) for homography.")
            continue
            
        # 4. Calculate and save homography
        H, mask = cv2.findHomography(np.array(image_points), np.array(world_points), cv2.RANSAC, 5.0)
        
        if H is None:
            print("  -> Homography calculation failed.")
            continue

        output_filename = os.path.join(output_dir, Path(json_file).stem + '.yaml')
        homography_data = {'H': H.tolist(), 'source_file': json_file}
        
        with open(output_filename, 'w') as f:
            yaml.dump(homography_data, f, default_flow_style=False)
            
        print(f"  -> Homography saved to {output_filename}")

        all_homographies.append(H)

    # After processing all files, analyze the variance
    if len(all_homographies) > 1:
        print("\n--- Homography Variance Analysis ---")
        
        # Use the first homography as the baseline
        h_base = all_homographies[0]
        differences = []
        
        for i in range(1, len(all_homographies)):
            # Calculate the Frobenius norm of the difference
            diff = np.linalg.norm(h_base - all_homographies[i], 'fro')
            differences.append(diff)
            print(f"Difference between H_0 and H_{i}: {diff:.6f}")
            
        avg_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        print("\nSummary:")
        print(f"Processed {len(all_homographies)} frames in the segment.")
        print(f"Average difference (Frobenius norm) from the first matrix: {avg_diff:.6f}")
        print(f"Maximum difference (Frobenius norm) from the first matrix: {max_diff:.6f}")
        print("--- End of Analysis ---")


    print("\nProcessing complete.")

if __name__ == '__main__':
    # --- Configuration ---
    DATASET_BASE_DIR = '/home/pranavdoma/Downloads/OpenLane/'
    # --- Focus on just ONE segment for this analysis ---
    SEGMENT_DIR = 'lane3d_1000_validation_test-001/validation/segment-17065833287841703_2980_000_3000_000_with_camera_labels'
    ANNOTATION_DIR = os.path.join(DATASET_BASE_DIR, SEGMENT_DIR)
    OUTPUT_DIR = os.path.join(DATASET_BASE_DIR, 'homography_output_segment_test')
    # ---------------------
    
    main(ANNOTATION_DIR, OUTPUT_DIR)
