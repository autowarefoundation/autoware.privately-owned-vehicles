#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.height_map import HeightMap
from SuperDepth.create_depth.common.depth_boundaries import DepthBoundaries
from SuperDepth.create_depth.common.depth_sparse_supervision import DepthSparseSupervision

def parseCalib(calib_filepath):
    
    focal_length = 0
    cy = 0
    lidar_to_rgb = 0
    K_rgb = 0

    # Reading calibration file JSON
    with open(calib_filepath, 'r') as file:
        calib_data = json.load(file)
        K_rgb = calib_data["intrinsics"]["rgb"]["K"]
        lidar_to_rgb = calib_data["extrinsics"]["lidar2rgb"]
        focal_length = K_rgb[1][1]
        cy = K_rgb[1][2]

    return focal_length, cy, K_rgb, lidar_to_rgb

def load_lidar_data(lidar_path):

    loaded_pcd = np.fromfile(lidar_path, dtype=np.float64)
    return loaded_pcd.reshape((-1, 6))

def project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb):

    pcnp_front_ones = np.concatenate((point_cloud_xyz, np.ones((point_cloud_xyz.shape[0], 1))), axis=1)
    point_cloud_xyz_front = sensor2rgb @ pcnp_front_ones.T
    uv_img_cords = K_rgb @ point_cloud_xyz_front[0:3, :] / point_cloud_xyz_front[2, :]
    return uv_img_cords

def filter_by_image_boundaries(pcd_lidar, uv_img_cords, h, w):

    valid_indices = (
            (uv_img_cords[0, :] > 0) &
            (uv_img_cords[0, :] < (w - 1)) &
            (uv_img_cords[1, :] > 0) &
            (uv_img_cords[1, :] < (h - 1))
    )
    pcd_lidar = pcd_lidar[valid_indices]
    uv_img_cords_filtered = uv_img_cords[:, valid_indices]
    return pcd_lidar, uv_img_cords_filtered

def filter_and_project_pcd_to_image(pcd, sensor2rgb, K_rgb, target_shape=(1920, 1080)):

    point_cloud_xyz = pcd[:, :3]
    w, h = target_shape

    uv_img_cords = project_pcd_to_image(K_rgb, point_cloud_xyz, sensor2rgb)

    pcd, uv_img_cords_filtered = filter_by_image_boundaries(pcd, uv_img_cords, h, w)

    return uv_img_cords_filtered, pcd

def load_points_in_image_lidar(lidar_path, K_rgb, lidar2rgb, target_shape=(1920, 1080)):

    # Load the lidar data from the .bin file
    pcd_points = load_lidar_data(lidar_path)

    # Project to image
    uv_img_cords_filtered, pcd_points_filtered = \
        filter_and_project_pcd_to_image(pcd_points, lidar2rgb, K_rgb, target_shape)
    
    return uv_img_cords_filtered, pcd_points_filtered

def create_image_from_point_cloud(uv_img_cords_filtered, filtered_pcd_points, target_shape=(1920, 1080),
                                        height_channel=True, dtype=np.float32):
    w, h = target_shape
    assert uv_img_cords_filtered.shape[1] == filtered_pcd_points.shape[0], "Dimensions mismatch."

    # Initialize an image with zeros
    image = np.zeros((h, w, 3), dtype=dtype)

    # Extract x, y, z, intensity, ring, and timestamp from filtered_pcd_points
    x = filtered_pcd_points[:, 0]  # front
    y = filtered_pcd_points[:, 1]  # left-right
    z = filtered_pcd_points[:, 2]  # height
    intensity = filtered_pcd_points[:, 3]

    # Calculate range as the Euclidean distance
    range_channel = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Height is the z coordinate
    height_channel_entry = z if height_channel else np.zeros(z.shape)

    # Map and store the channels in the image
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 0] = range_channel
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 1] = intensity
    image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), 2] = height_channel_entry

    return image

def cropData(image_left, depth_map, depth_boundaries, height_map, sparse_supervision):

    # Cropping out those parts of data for which depth is unavailable
    image_left = image_left.crop((500, 300, 1300, 700))
    depth_map = depth_map[300:700, 500:1300]
    depth_boundaries = depth_boundaries[300:700, 500:1300]
    height_map = height_map[300:700, 500:1300]
    sparse_supervision = sparse_supervision[300:700, 500:1300]

    return image_left, depth_map, depth_boundaries, height_map, sparse_supervision

def main():
    
    # Argument parser for data root path and save path
    parser = ArgumentParser()
    parser.add_argument("-r", "--root", dest="root_data_path", help="path to root folder with input ground truth labels and images")
    parser.add_argument("-s", "--save", dest="root_save_path", help="path to folder where processed data will be saved")
    args = parser.parse_args()

    # Filepaths for data loading and savind
    root_data_path = args.root_data_path
    root_save_path = args.root_save_path

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'lidar/'
    images_filepath = root_data_path + 'frame_camera/'
    calib_filepath =  root_data_path + 'calib.json'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.bin")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)
   
    checkData = CheckData(num_images, num_depth_maps)
    check_passed = checkData.getCheck()

    if(check_passed):
        print('Beginning processing of data')

        focal_length, cy, K_rgb, lidar_to_rgb = parseCalib(calib_filepath)
        target_shape=(1920, 1080)
        
        # Height map limits
        max_height = 7
        min_height = -5

        # Camera height above ground
        camera_height = 1.4

        # Looping through data 
        for index in range(0, num_depth_maps):
            
            print(f'Processing image {index} of {num_depth_maps-1}')

            # Image
            image = Image.open(str(images[index]))

            # Pointcloud projection
            uv_img_cords_filtered, pcd_points_filtered = \
                load_points_in_image_lidar(str(depth_maps[index]), 
                    K_rgb, lidar_to_rgb, target_shape = target_shape)

            pointcloud_projection = create_image_from_point_cloud(uv_img_cords_filtered, pcd_points_filtered, target_shape)
            sparse_depth_map = pointcloud_projection[:,:,0]

            # Filled in depth map
            lidar_depth_fill = LidarDepthFill(sparse_depth_map)
            depth_map = lidar_depth_fill.getDepthMap()
            depth_map_fill_only = lidar_depth_fill.getDepthMapFillOnly()
            
            # Calculating depth boundaries
            boundary_threshold = 8
            depthBoundaries = DepthBoundaries(depth_map_fill_only, boundary_threshold)
            depth_boundaries = depthBoundaries.getDepthBoundaries()

            # Height map
            heightMap = HeightMap(depth_map, max_height, min_height, 
                 camera_height, focal_length, cy)
            height_map = heightMap.getHeightMap()

            # Sparse Supervision
            supervision_threshold = 25
            depthSparseSupervision = DepthSparseSupervision(image, height_map, max_height, min_height, supervision_threshold)
            sparse_supervision = depthSparseSupervision.getSparseSupervision()

             # Crop side regions where depth data is missing
            image, depth_map, depth_boundaries, height_map, sparse_supervision= \
                cropData(image, depth_map, depth_boundaries, height_map, sparse_supervision)
                        
            # Save files
            # RGB Image as PNG
            image_save_path = root_save_path + '/image/' + str(index) + '.png'
            image.save(image_save_path, "PNG")

            # Depth map as binary file in .npy format
            depth_save_path = root_save_path + '/depth/' + str(index) + '.npy'
            np.save(depth_save_path, depth_map)

            # Height map as binary file in .npy format
            height_save_path = root_save_path + '/height/' + str(index) + '.npy'
            np.save(height_save_path, height_map)

            # Sparse supervision map as binary file in .npy format
            supervision_save_path = root_save_path + '/supervision/' + str(index) + '.npy'
            np.save(supervision_save_path, sparse_supervision)

            # Boundary mask as PNG
            boundary_save_path = root_save_path + '/boundary/' + str(index) + '.png'
            boundary_mask = Image.fromarray(depth_boundaries)
            boundary_mask.save(boundary_save_path, "PNG")

            # Height map plot for data auditing purposes
            height_plot_save_path = root_save_path + '/height_plot/' + str(index) + '.png'
            plt.imsave(height_plot_save_path, height_map, cmap='inferno_r')

        print('----- Processing complete -----')    

if __name__ == '__main__':
    main()
#%%