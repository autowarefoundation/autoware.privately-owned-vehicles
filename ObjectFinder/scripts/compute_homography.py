import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import json

tf.enable_eager_execution()

# Function to extract lane points and compute homography
def extract_lane_points_and_compute_homography(lane_labels):
    real_x = []
    real_y = []
    u = []
    v = []

    for lane in lane_labels["lane_lines"]:
        u_list, v_list = lane['uv']
        x, y, z = lane["xyz"]
        xy = np.array(list(zip(x, y)))

        if len(u_list) == len(v_list):
            u.append(u_list[0])
            v.append(v_list[0])
            real_x.append(x[0])
            real_y.append(y[0])

            u.append(u_list[-1])
            v.append(v_list[-1])
            real_x.append(x[-1])
            real_y.append(y[-1])

        else:
            print("Lengths unequal")

    uv = np.stack([u, v], axis=1).astype(np.float32)
    xy = np.stack([real_x, real_y], axis=1).astype(np.float32)

    print("UV points:", uv)
    print("XY points:", xy)

    # Compute homography from ground plane (x, y) to image (u, v)
    H, mask = cv2.findHomography(xy, uv, method=cv2.RANSAC)

    return H, mask


def main(args):
    # Load OpenLane JSON
    with open(args.json_file, 'r') as f:
        lane_labels = json.load(f)

    # Extract lane points and compute homography
    H, _ = extract_lane_points_and_compute_homography(lane_labels)

    # Print and save the homography matrix
    print("Computed Homography Matrix:")
    print(H)

    H_inv = np.linalg.inv(H)
    print("Inverse Homography Matrix:")
    print(H_inv)

    # Save the inverse homography matrix to the specified output file
    np.save(args.output_file, H_inv)
    print(f"Saved inverse homography matrix to {args.output_file}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compute homography from lane labels.")
    parser.add_argument("json_file", type=str, help="Path to the input JSON file with lane labels.")
    parser.add_argument("output_file", type=str, help="Name of the output file to save the homography matrix.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
