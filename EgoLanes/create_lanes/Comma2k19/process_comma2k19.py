import os
import sys
import numpy as np
import utils.orientation as orient
from utils.camera import img_from_device, denormalize, view_frame_from_device_frame
import cv2
import glob
import json
import argparse

# Define the region of interest (ROI)
x_off, y_off = 62, 84  # Top-left corner coordinates
img_w, img_h = 1048, 524  # Width and height of the ROI
future_frames = 100
distance_thres = 100
def load_frame_pos(seg_path):
    """
    Load frame positions and orientations from the given segment path.

    Parameters:
    seg_path (str): The path to the segment directory containing the global_pose data.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - frame_positions: The positions of the frames.
        - frame_orientations: The orientations of the frames.
    """
    frame_positions = np.load(seg_path + 'global_pose/frame_positions')
    frame_orientations = np.load(seg_path + 'global_pose/frame_orientations')
    return frame_positions, frame_orientations

def polygon_area(x,y):
    if len(x) < 3:
        return 10000
    x = np.append(x, x[-1])
    y = np.append(y, img_h)
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    return 0.5*np.abs(main_area + correction)

def get_frame_positions_local(frame_count, frame_positions, frame_orientations):
    """
    Get local frame positions relative to a specific frame.

    Parameters:
    frame_count (int): The index of the frame to use as the reference.
    seg_path (str): The path to the segment directory containing the global_pose data.

    Returns:
    numpy.ndarray: An array of local frame positions.
    """
    ecef_from_local = orient.rot_from_quat(frame_orientations[frame_count])
    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions - frame_positions[frame_count])
    frame_positions_local = frame_positions_local[frame_count:]
    frame_positions = frame_positions[frame_count:]
    dist = np.linalg.norm(frame_positions - frame_positions[0], axis=1)
    if np.max(dist) < distance_thres:
        end = len(dist)
    else:
        end = int(np.where(dist>distance_thres)[0][0])
    start = int(np.argmin(np.where(dist == 0)))
    frame_positions_local = frame_positions_local[start:end]
    # print(np.linalg.norm(frame_positions[frame_count+200] - frame_positions[frame_count]))
    return frame_positions_local

def extrapolate_to_bottom(img_pts):
    """
    Extrapolate the line defined by the first two points to the bottom of the image.

    Parameters:
    points (numpy.ndarray): An array of points as tuples (x, y).

    Returns:
    numpy.ndarray: An array of points with the line extrapolated to the bottom.
    """
    try:
        a = np.where(img_pts[:,1] > img_h)[0]
        idx = int(np.argmax(a))
        x1, y1 = img_pts[idx -1]
        x2, y2 = img_pts[idx]
        y = img_h 
        if y2-y1 != 0:
            x = int(x1 + (x2 - x1) * (y - y1) / (y2 - y1))
            return np.vstack(([x,img_h],img_pts[idx:]))
    except Exception as e:
        print(e)
    x1, y1= img_pts[0]
    x2, y2= img_pts[1]
    
    if x2==x1 or y1==y2:
        x = x1
    else:
        slope = int(y2 - y1) / int(x2 - x1)
        intercept = y1 - slope * x1
        x = int((img_h-intercept)/slope)

    return np.vstack(([x,img_h],img_pts))

def check_multiple_x_for_y(points):
    # Sort by y values
    sorted_points = points[points[:, 1].argsort()]

    # Get unique y values and their counts
    unique_y, counts = np.unique(sorted_points[:, 1], return_counts=True)

    # Check if any y value has more than one corresponding x value
    return np.any(counts > 50) or np.any(points[:,1] < 200)

def generate_mask_vis(frame_count, out_path, processed_img_count, 
                      frame_positions, frame_orientations, org_img,
                      height=1.2, line_color=(0,255,0)):
    """
    Generate a mask visualization for a specific frame.

    Parameters:
    frame_count (int): The index of the frame to process.
    seg_path (str): The path to the segment directory containing the images.
    width (float): The width of the path to visualize.
    height (float): The height of the path to visualize.
    fill_color (tuple): The color to fill the mask with.
    line_color (tuple): The color to draw the lines with.

    Returns:
    numpy.ndarray: An array of image points.
    """
    device_path = get_frame_positions_local(frame_count, frame_positions, frame_orientations)
    device_path = device_path + np.array([0, 0, height])


    img_points_norm = img_from_device(device_path)
    img_pts = denormalize(img_points_norm)
    # filter out things rejected along the way
    valid = np.isfinite(img_pts).all(axis=1)
    img_pts = img_pts[valid].astype(int)
    img_pts = img_pts[np.sort(np.unique(img_pts, axis=0, return_index=True)[1])]
    img_pts = img_pts -[x_off,y_off]

    x_s = img_pts[:, 0]
    y_s = img_pts[:, 1]
    valid = (x_s >= 0) & (x_s < img_w) & \
           (y_s >= 0) 
    img_pts = img_pts[valid]
    if len(img_pts) < 5 or check_multiple_x_for_y(img_pts):
        return None
    img_pts = extrapolate_to_bottom(img_pts)
    if img_pts[0][0]<250 or img_pts[0][0]>750 or img_pts[-1][0]<150 or img_pts[-1][0]>950:
        return None
    auc = polygon_area(img_pts[:,0], img_pts[:,1])
    if auc > 17000:
        return None 
    # print(len(img_pts_l), len(img_pts_r))
    img = org_img.copy()
    mask = np.zeros((img_h, img_w), np.uint8)
    for i in range(len(img_pts) - 1):
        # if img_pts[i+1][1] > img_pts[i][1]:
        #     return None
        cv2.line(img, img_pts[i], img_pts[i + 1], line_color, 3)
        cv2.line(mask, img_pts[i], img_pts[i + 1], (255), 3)
    # print(processed_img_count, auc, img_pts[0], img_pts[-1])
    cv2.imwrite(f"{out_path}images/{processed_img_count:06d}.png", org_img)
    cv2.imwrite(f"{out_path}segmentation/{processed_img_count:06d}.png", mask)
    cv2.imwrite(f"{out_path}visualization/{processed_img_count:06d}.png", img)
    img_pts = [
        [point[0] / img_w, point[1] / img_h]
        for point in img_pts
    ]
    return img_pts

def extract_frames(seg_path, out_path, img_count, downsampling_factor=1):
    """    Extract frames from a video file and save them as images.

    Parameters:
    seg_path (str): The path to the segment directory containing the video file.
    downsampling_factor (int): The factor by which to downsample the frames.

    Returns:
    int: The total number of frames extracted.
    """
    video_path = seg_path +"video.hevc"

    frame_positions, frame_orientations = load_frame_pos(seg_path)
    total_frames = frame_positions.shape[0]

    vidcap = cv2.VideoCapture(video_path)

    if not vidcap.isOpened():
        print("Error opening video file")
        return

    frame_count = 0
    last_saved_frame = 0
    jdata = {}
    success, image = vidcap.read()
        
    x2, y2 = x_off+img_w, y_off+img_h # Bottom-right corner coordinates
    
    while success and frame_count < (total_frames-future_frames):
        if frame_count > last_saved_frame + downsampling_factor:
            cropped_image = image[y_off:y2, x_off:x2]
            drive_path = generate_mask_vis(frame_count, out_path, img_count, frame_positions, frame_orientations, cropped_image)
            if drive_path is not None:
                data = {f"{img_count:06d}":{"drivable_path": drive_path.tolist(),"img_width": img_w, "img_height": img_h}}
                jdata.update(data)
                img_count += 1
                last_saved_frame = frame_count
        success, image = vidcap.read()
        frame_count += 1
    vidcap.release()
    return jdata, img_count

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Process video segments to generate masks and visualizations.")
    parser.add_argument('--df', type=int, default=60, help='Factor by which to downsample the frames.')
    parser.add_argument('--dataset_root_path', type=str, default="", help='Root path of the dataset.')
    parser.add_argument('--out_path', type=str, default="", help='Path to the output directory.')

    args = parser.parse_args()
    downsampling_factor = args.df
    dataset_root_path = args.dataset_root_path
    out_path = args.out_path

    if not os.path.exists(out_path):
        os.makedirs(out_path)
        os.makedirs(out_path + "segmentation")
        os.makedirs(out_path + "visualization")
        os.makedirs(out_path + "images")

    segments = glob.glob(dataset_root_path+"Chunk_*/*/*/")
    json_data = {}
    processed_img_count, img_count = 0, 0
    for seg in segments:
        print(f"Start processing {seg}")
        jdata, img_count =  extract_frames(seg, out_path, img_count, downsampling_factor)
        json_data.update(jdata)
        # print(f"Finished processing {seg}")
    with open(f"{out_path}drivable_path.json", "w") as json_file:
        json.dump(json_data, json_file)
    print("All segments processed successfully", len(segments))
