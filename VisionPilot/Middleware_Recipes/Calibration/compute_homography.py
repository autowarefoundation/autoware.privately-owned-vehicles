import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import yaml
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import itertools

# Enable eager execution for TensorFlow
tf.enable_eager_execution()


# ============================================================================
# ROI Selection
# ============================================================================

def get_manual_roi(projected_points, camera_image):
    """
    Interactive ROI selection via matplotlib clicks.
    User clicks top-left, then bottom-right corners.
    """
    print("\n=== Manual ROI Selection ===")
    print("1. Click TOP-LEFT corner of desired region")
    print("2. Click BOTTOM-RIGHT corner of desired region")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    img = tf.image.decode_jpeg(camera_image.image)
    ax.imshow(img)
    
    # Show LiDAR projection for context
    xs, ys, ranges = projected_points[:, 0], projected_points[:, 1], projected_points[:, 2]
    cmap = plt.get_cmap('jet')
    colors = cmap((ranges % 20.0) / 20.0)
    ax.scatter(xs, ys, c=colors, s=5.0, edgecolors="none", alpha=0.5)
    
    ax.set_title("Click TOP-LEFT, then BOTTOM-RIGHT to define ROI", fontsize=16)
    plt.grid(False)
    
    # Wait for two clicks
    points = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(points) < 2:
        print("ROI selection cancelled. Exiting.")
        return None

    (x1, y1), (x2, y2) = points
    roi = [int(min(x1, x2)), int(min(y1, y2)), 
           int(max(x1, x2)), int(max(y1, y2))]
    
    print(f"Selected ROI: {roi}")
    return roi


def get_default_roi(image):
    """
    Returns a smart default ROI covering the center of the road.
    Avoids car hood (bottom) and horizon (top).
    """
    img_shape = tf.image.decode_jpeg(image.image).shape
    h, w = img_shape[0], img_shape[1]
    
    roi = [w // 4, h // 3, w * 3 // 4, h * 2 // 3]
    print(f"Using default ROI for {w}x{h} image: {roi}")
    return roi


def filter_points_in_roi(projected_points, three_d_points, roi):
    """
    Filters projected 2D points and corresponding 3D points to those within ROI.
    """
    roi_mask = (
        (projected_points[:, 0] >= roi[0]) &
        (projected_points[:, 0] < roi[2]) &
        (projected_points[:, 1] >= roi[1]) &
        (projected_points[:, 1] < roi[3])
    )
    
    projected_roi = projected_points[roi_mask]
    points_roi = three_d_points.numpy()[roi_mask]
    
    print(f"Points in ROI: {projected_roi.shape[0]}")
    return projected_roi, points_roi


# ============================================================================
# Homography Computation
# ============================================================================

def compute_homography(projected_points_2d, world_points_3d):
    """
    Computes homography from 2D image points to 3D world points (ground plane).
    Uses RANSAC for robustness.
    """
    src_points = projected_points_2d[:, :2]  # (u, v) in image
    dst_points = world_points_3d[:, :2]      # (x, y) in world (z ignored)
    
    print(f"Computing homography from {len(src_points)} point correspondences...")
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is not None:
        print("Homography matrix computed successfully:")
        print(H)
    else:
        print("Failed to compute homography!")
    
    return H


def test_homography_consistency(H, projected_points_2d, ground_truth_3d):
    """
    Tests homography accuracy by transforming 2D points to world coords
    and comparing against ground truth LiDAR 3D points.
    Returns average error in meters.
    """
    src_points = np.float32(projected_points_2d[:, :2]).reshape(-1, 1, 2)
    predicted_world = cv2.perspectiveTransform(src_points, H).reshape(-1, 2)
    gt_world = ground_truth_3d[:, :2]
    
    errors = np.linalg.norm(predicted_world - gt_world, axis=1)
    return np.mean(errors)


# ============================================================================
# Visualization
# ============================================================================

def plot_points_on_image(projected_points, camera_image, filename="projection_output.png"):
    """
    Plots LiDAR points projected onto camera image.
    Color-coded by range.
    """
    img = tf.image.decode_jpeg(camera_image.image)
    
    plt.figure(figsize=(20, 12))
    plt.imshow(img)
    plt.title(f"LiDAR Projection - {open_dataset.CameraName.Name.Name(camera_image.name)}")
    plt.grid(False)
    plt.axis('off')

    xs, ys, ranges = projected_points[:, 0], projected_points[:, 1], projected_points[:, 2]
    cmap = plt.get_cmap('jet')
    colors = cmap((ranges % 20.0) / 20.0)
    plt.scatter(xs, ys, c=colors, s=5.0, edgecolors="none", alpha=0.5)
    
    plt.savefig(filename)
    print(f"Saved visualization: {filename}")
    plt.close()


# ============================================================================
# I/O
# ============================================================================

def save_homography_to_yaml(H, filename="homography.yaml"):
    """
    Saves homography matrix to YAML in ObjectFinder-compatible format.
    Uses 'H' key with rows, cols, data structure.
    """
    if H is None:
        print("Cannot save: homography is None")
        return
    
    yaml_data = {
        'H': {
            'rows': H.shape[0],
            'cols': H.shape[1],
            'data': H.flatten().tolist()
        }
    }
    
    with open(filename, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"Homography saved to: {filename} (as 'H' field)")


# ============================================================================
# Frame Processing
# ============================================================================

def extract_front_camera_data(frame):
    """
    Extracts LiDAR points projected onto front camera from a Waymo frame.
    Returns: (projected_points_2d, world_points_3d, front_camera_image)
    """
    # Parse LiDAR and camera data
    (range_images, camera_projections, _, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose
    )
    
    points_all = np.concatenate(points, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)
    
    # Get front camera image
    images = sorted(frame.images, key=lambda i: i.name)
    front_camera_image = images[0]
    
    # Filter points for front camera only
    cp_points_tensor = tf.constant(cp_points_all)
    mask = tf.equal(cp_points_tensor[..., 0], front_camera_image.name)
    
    cp_points_front = tf.boolean_mask(cp_points_tensor, mask)
    points_front = tf.boolean_mask(points_all, mask)
    
    # Cast to float32 and compute range
    cp_points_front = tf.cast(cp_points_front, dtype=tf.float32)
    points_range = tf.norm(points_front, axis=-1)
    
    # Format as [u, v, range]
    projected_points = tf.concat(
        [cp_points_front[..., 1:3], tf.expand_dims(points_range, axis=-1)],
        axis=-1
    ).numpy()
    
    return projected_points, points_front, front_camera_image


# ============================================================================
# Main Pipeline
# ============================================================================

def main(args):
    """
    Main homography calibration pipeline:
    1. Load Waymo TFRecord
    2. Extract LiDAR projected onto front camera
    3. Define ROI (manual or default)
    4. Compute homography from 2D→3D correspondences
    5. Save to YAML
    6. Test consistency on additional frames
    """
    print(f"Loading TFRecord: {args.filename}")
    dataset = tf.data.TFRecordDataset(args.filename, compression_type='')
    
    reference_roi = None
    H_ref = None
    
    for frame_idx, data in enumerate(itertools.islice(dataset, args.num_frames)):
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx + 1}/{args.num_frames}")
        print('='*60)
        
        # Parse frame
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        if len(frame.pose.transform) != 16:
            print(f"Skipping frame {frame_idx + 1}: missing pose information")
            continue
        
        # Extract front camera data
        projected_points, points_3d, front_camera = extract_front_camera_data(frame)
        
        if frame_idx == 0:
            # === REFERENCE FRAME: Compute homography ===
            print("\n>>> REFERENCE FRAME: Computing homography <<<")
            
            # Select ROI
            if args.manual_roi:
                reference_roi = get_manual_roi(projected_points, front_camera)
                if reference_roi is None:
                    return
            elif args.roi:
                reference_roi = args.roi
                print(f"Using CLI-provided ROI: {reference_roi}")
            else:
                reference_roi = get_default_roi(front_camera)
            
            # Filter points in ROI
            proj_roi, points_roi = filter_points_in_roi(
                projected_points, points_3d, reference_roi
            )
            
            if len(proj_roi) < 4:
                print("ERROR: Not enough points in ROI for homography computation")
                return
            
            # Visualize ROI points
            plot_points_on_image(proj_roi, front_camera, filename="roi_ref_frame.png")
            
            # Compute homography
            H_ref = compute_homography(proj_roi, points_roi)
            
            if H_ref is None:
                print("ERROR: Failed to compute homography. Exiting.")
                return
            
            # Save to YAML
            save_homography_to_yaml(H_ref, filename=args.output)
        
        else:
            # === VALIDATION FRAMES: Test consistency ===
            print(f"\n>>> VALIDATION FRAME {frame_idx + 1}: Testing consistency <<<")
            
            if H_ref is None:
                print("No reference homography available. Skipping.")
                continue
            
            # Use same ROI as reference
            proj_roi, points_roi = filter_points_in_roi(
                projected_points, points_3d, reference_roi
            )
            
            if len(proj_roi) < 4:
                print("Not enough points in ROI for validation. Skipping.")
                continue
            
            # Test homography accuracy
            avg_error = test_homography_consistency(H_ref, proj_roi, points_roi)
            print(f"Average reprojection error: {avg_error:.4f} meters")
    
    print("\n" + "="*60)
    print("Homography calibration complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute homography calibration from Waymo LiDAR→Camera projection"
    )
    parser.add_argument(
        '--filename', 
        type=str,
        required=True,
        help='Path to Waymo TFRecord file'
    )
    parser.add_argument(
        '--num_frames', 
        type=int, 
        default=3,
        help='Number of frames to process (1st = calibration, rest = validation)'
    )
    parser.add_argument(
        '--roi', 
        type=int, 
        nargs=4,
        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
        help='ROI as [x_min y_min x_max y_max]. If not set, uses default center region.'
    )
    parser.add_argument(
        '--manual_roi', 
        action='store_true',
        help='Enable interactive ROI selection (click on image)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='homography.yaml',
        help='Output YAML path (default: homography.yaml)'
    )
    
    args = parser.parse_args()
    main(args)
