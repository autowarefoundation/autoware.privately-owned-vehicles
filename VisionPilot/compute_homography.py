import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

# Enable eager execution for TensorFlow
tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


def get_manual_roi(projected_points, camera_image):
    """Displays an image and allows the user to manually select an ROI with two clicks."""
    
    print("\n--- Manual ROI Selection ---")
    print("Please click on the image to select the ROI.")
    print("1. Click the TOP-LEFT corner of your desired region.")
    print("2. Click the BOTTOM-RIGHT corner of your desired region.")
    
    # Create a plot to display the image for selection
    fig, ax = plt.subplots(figsize=(20, 12))
    img = tf.image.decode_jpeg(camera_image.image)
    ax.imshow(img)
    
    # Plot all the points to give context for the selection
    xs = projected_points[:, 0]
    ys = projected_points[:, 1]
    ranges = projected_points[:, 2]
    cmap = plt.get_cmap('jet')
    colors = cmap((ranges % 20.0) / 20.0)
    ax.scatter(xs, ys, c=colors, s=5.0, edgecolors="none", alpha=0.5)
    
    ax.set_title("Click TOP-LEFT, then BOTTOM-RIGHT to define ROI", fontsize=16)
    plt.grid(False)
    plt.axis('on')
    
    # Wait for the user to make two clicks
    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(points) < 2:
        print("ROI selection cancelled or timed out. Exiting.")
        return None

    (x1, y1), (x2, y2) = points
    
    # Ensure the coordinates are ordered correctly as min and max
    roi_x_min = min(x1, x2)
    roi_y_min = min(y1, y2)
    roi_x_max = max(x1, x2)
    roi_y_max = max(y1, y2)

    ROI = [int(roi_x_min), int(roi_y_min), int(roi_x_max), int(roi_y_max)]
    print(f"Manual ROI selected: {ROI}")
    
    return ROI


def plot_points_on_image(projected_points, camera_image, point_size=5.0, filename="projection_output.png"):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    point_size: the point size.
    filename: The name of the file to save the plot to.
  """
  # Decode the JPEG image
  img = tf.image.decode_jpeg(camera_image.image)
  
  # Create a plot
  plt.figure(figsize=(20, 12))
  plt.imshow(img)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

  # CRITICAL FIX: Ensure we are using the 'projected_points' argument passed to this function,
  # and not a variable from the parent scope.
  xs = projected_points[:, 0]
  ys = projected_points[:, 1]
  ranges = projected_points[:, 2]

  # Generate colors based on range
  cmap = plt.get_cmap('jet')
  colors = cmap((ranges % 20.0) / 20.0)
  
  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none", alpha=0.5)
  
  # Save the figure to a file
  plt.savefig(filename)
  print(f"Saved projection image to {filename}")
  plt.close() # Close the plot to free up memory

def define_and_filter_roi(projected_points, three_d_points, image, roi_coords):
    """Defines the ROI and filters points to be within it."""
    
    # Get the dynamic dimensions of the front camera image
    image_shape = tf.image.decode_jpeg(image.image).shape
    image_height = image_shape[0]
    image_width = image_shape[1]

    # If no specific ROI is provided via command line, use a smart default.
    # This default targets the center of the road, avoiding the hood and the horizon.
    if roi_coords is None:
        roi_x_min = image_width // 4
        roi_y_min = image_height // 3
        roi_x_max = image_width * 3 // 4
        roi_y_max = image_height * 2 // 3
        ROI = [roi_x_min, roi_y_min, roi_x_max, roi_y_max]
    else:
        ROI = roi_coords

    print(f"Using ROI for an image of size ({image_width}x{image_height}): {ROI}")

    # Create a boolean mask to select points that are INSIDE the ROI.
    roi_mask = (projected_points[:, 0] >= ROI[0]) & \
               (projected_points[:, 0] < ROI[2]) & \
               (projected_points[:, 1] >= ROI[1]) & \
               (projected_points[:, 1] < ROI[3])

    # Apply the mask to get ONLY the points inside the ROI
    projected_points_in_roi = projected_points[roi_mask]
    
    # We must also filter the corresponding 3D points so they stay in sync.
    points_in_roi = three_d_points.numpy()[roi_mask]
    
    print(f"Found {projected_points_in_roi.shape[0]} points within the ROI.")
    
    return projected_points_in_roi, points_in_roi

def compute_and_save_homography(projected_points_in_roi, points_in_roi):
    """Computes and saves the homography matrix from corresponding 2D and 3D points."""

    # These are the corresponding points we will use to calculate the homography.
    # Source points are the 2D pixel coordinates (u, v) from the camera image.
    src_points = projected_points_in_roi[:, :2]

    # Destination points are the 3D world coordinates (x, y) on the ground plane.
    # We ignore the z-coordinate as per the homography definition for a planar surface.
    dst_points = points_in_roi[:, :2]

    print(f"Extracted {len(src_points)} corresponding points for homography calculation.")

    # ---- Step 5: Compute the Homography Matrix ----
    print("Calculating homography matrix...")

    # Use cv2.findHomography to compute the matrix.
    # RANSAC is a robust method that can handle outliers in the data.
    homography_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    if homography_matrix is not None:
        print("Successfully calculated homography matrix:")
        print(homography_matrix)
        
        # Save the homography matrix to a file for later use
        homography_filename = "homography.npy"
        np.save(homography_filename, homography_matrix)
        print(f"Homography matrix saved to {homography_filename}")
    else:
        print("Could not compute homography matrix. Not enough inliers found.")

def main(args):
    # TODO: Replace with the actual path to your TFRecord file
    FILENAME = args.filename
    
    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    
    # We will iterate through the dataset to find the first frame with valid pose information
    frame = None
    for data in dataset:
        # Parse the frame data
        frame_candidate = open_dataset.Frame()
        frame_candidate.ParseFromString(bytearray(data.numpy()))
        # Check if the frame has the required 4x4 pose transform matrix (16 elements)
        if len(frame_candidate.pose.transform) == 16:
            frame = frame_candidate
            print("Found a valid frame with pose information.")
            break

    if frame is None:
        print(f"Error: Could not find a valid frame with pose information in {FILENAME}")
        return

    # ---- Step 1: Extract LiDAR points and Camera Projections ----

    # Parse range images and camera projections from the frame
    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert range images to point clouds
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose
    )

    # Concatenate points from all LiDAR sensors
    # points_all contains the 3D coordinates (x, y, z) of each LiDAR point in the vehicle frame
    points_all = np.concatenate(points, axis=0)
    
    # Concatenate camera projections from all LiDAR sensors
    # cp_points_all contains the corresponding 2D projection for each 3D point:
    # [camera_name, u, v] where u,v are pixel coordinates
    cp_points_all = np.concatenate(cp_points, axis=0)

    # Get a list of all camera images in the frame, sorted by name
    images = sorted(frame.images, key=lambda i: i.name)

    print(f"Successfully processed frame.")
    print(f"Total LiDAR points: {points_all.shape[0]}")
    print(f"Front camera name: {open_dataset.CameraName.Name.Name(images[0].name)}")

    # ---- Step 2: Filter and Visualize Projections for the Front Camera ----

    # Get the front camera image
    front_camera_image = images[0]
    
    # Create a tensor of all camera projection points
    cp_points_all_tensor = tf.constant(cp_points_all)
    
    # Create a mask to select points that project to the front camera
    mask = tf.equal(cp_points_all_tensor[..., 0], front_camera_image.name)
    
    # Filter the camera projection points using the mask
    cp_points_front_camera = tf.boolean_mask(cp_points_all_tensor, mask)

    # Filter the corresponding 3D points
    points_front_camera = tf.boolean_mask(points_all, mask)

    # Cast the projection points tensor to float32 to ensure type consistency
    cp_points_front_camera = tf.cast(cp_points_front_camera, dtype=tf.float32)

    # Calculate the range (distance) of each 3D point from the vehicle origin
    points_range = tf.norm(points_front_camera, axis=-1)

    # Combine the projected pixel coordinates (u, v) and the range
    projected_points_front_camera = tf.concat(
        [cp_points_front_camera[..., 1:3], tf.expand_dims(points_range, axis=-1)], axis=-1
    ).numpy()

    print(f"Found {projected_points_front_camera.shape[0]} points projecting to the front camera.")
    
    # Visualize the points on the image
    print("Visualizing all projected points on the front camera image...")
    plot_points_on_image(projected_points_front_camera, front_camera_image, filename="full_projection_output.png")

    # ---- Step 3 & 4: Define ROI, Filter, and Extract Points ----
    
    # Determine the ROI
    if args.manual_roi:
        # Enter interactive mode for ROI selection
        roi_coords = get_manual_roi(projected_points_front_camera, front_camera_image)
        if roi_coords is None:
            return # Exit if manual selection is cancelled
    else:
        # Use command-line argument or the default
        roi_coords = args.roi

    projected_points_in_roi, points_in_roi = define_and_filter_roi(
        projected_points_front_camera, 
        points_front_camera, 
        front_camera_image, 
        roi_coords
    )
    
    # Visualize ONLY the points inside the ROI to confirm our selection.
    print("Visualizing ONLY the points within the defined ROI...")
    plot_points_on_image(projected_points_in_roi, front_camera_image, filename="roi_projection_output.png")

    # ---- Step 5: Compute the Homography Matrix ----
    compute_and_save_homography(projected_points_in_roi, points_in_roi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute a homography matrix from Waymo Open Dataset data.")
    parser.add_argument('--filename', type=str, 
                        default='/home/pranavdoma/Downloads/waymo/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord',
                        help='Path to the Waymo TFRecord file.')
    parser.add_argument('--roi', type=int, nargs=4, 
                        help='Specify the ROI as a rectangle with four integer values: x_min y_min x_max y_max. '
                             'If not provided, a default ROI targeting the main road will be used.')
    parser.add_argument('--manual_roi', action='store_true',
                        help='Enable interactive, manual ROI selection by clicking on the image.')
    args = parser.parse_args()
    main(args)
