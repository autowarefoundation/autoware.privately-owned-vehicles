#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
import json
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.stereo_sparse_supervision import StereoSparseSupervision
from SuperDepth.create_depth.common.height_map import HeightMap

def parseCalib(calib_files):
    
    calib_logs = []
    focal_lengths = []
    centre_y_vals = []

    for i in range (0, len(calib_files)):
        
        # Get filepath to calibration file
        calib_filepath = str(calib_files[i])
        
        # Storing log associated with calibration file
        calib_log = calib_filepath[56:-37]
        calib_logs.append(calib_log)

        # Reading calibration file JSON
        with open(calib_filepath, 'r') as file:
            data = json.load(file)

            for p in range(0, len(data['camera_data_'])):
                camera = (data['camera_data_'][p]['key'])

                # Read focal length and principal point y value
                if(camera == 'image_raw_stereo_front_left_rect'):
                    focal_length = (data['camera_data_'][p]['value']['focal_length_y_px_'])
                    cy = (data['camera_data_'][p]['value']['focal_center_y_px_'])

                    focal_lengths.append(focal_length)
                    centre_y_vals.append(cy)

    return calib_logs, focal_lengths, centre_y_vals

def createDepthMap(depth_data, focal_length, baseline):

    assert(np.max(depth_data) > 255)
    depth_data = depth_data.astype('float32') / 256.

    valid_pixels = depth_data > 0

    # Using the stereo relationship, recover the depth map by:
    depth_map = np.float32((focal_length * baseline) / (depth_data + (1.0 - valid_pixels)))

    # Clamping max value
    depth_map[depth_map > 200] = 0

    return depth_map      

def findDepthBoundaries(depth_map):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Initializing depth boundary mask
    depth_boundaries = np.zeros_like(depth_map, dtype=np.uint8)

    # Fiding depth boundaries
    for i in range(1, height-1):
        for j in range(1, width-1):

            # Finding derivative
            x_grad = depth_map[i-1,j] - depth_map[i+1, j]
            y_grad = depth_map[i,j-1] - depth_map[i, j+1]
            grad = abs(x_grad) + abs(y_grad)
            
            # Derivative threshold accounting for gap in depth map
            if(grad > 8 and depth_map[i-1, j] != 0):
                depth_boundaries[i,j] = 255

    return depth_boundaries 

def cropData(image_left, depth_map, depth_boundaries, height_map, sparse_supervision):

    # Getting size of depth map
    size = depth_map.shape
    height = size[0]
    width = size[1]

    # Cropping out those parts of data for which depth is unavailable
    image_left = image_left.crop((256, 950, width, height-50))
    depth_map = depth_map[950:height-50, 256 : width]
    depth_boundaries = depth_boundaries[950:height-50, 256 : width]
    height_map = height_map[950:height-50, 256 : width]
    sparse_supervision = sparse_supervision[950:height-50, 256 : width]

    return image_left, depth_map, depth_boundaries, height_map, sparse_supervision

def main():
    
    # Filepaths for data loading and saving
    root_data_path = '/mnt/media/Argoverse/'
    root_save_path = '/mnt/media/SuperDepth/Argoverse/'

    # Paths to read ground truth depth and input images from training data
    depth_filepath = root_data_path + 'disparity_maps_v1.1/'
    images_filepath = root_data_path + 'rectified_stereo_images_v1.1/train/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/stereo_front_left_rect_disparity/*.png")])
    images_left = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_left_rect/*.jpg")])
    images_right = sorted([f for f in pathlib.Path(images_filepath).glob("*/stereo_front_right_rect/*.jpg")])
    calib_files = sorted([f for f in pathlib.Path(images_filepath).glob("*/*.json")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images_left = len(images_left)
    num_images_right = len(images_right)

    check_data_left = CheckData(num_images_left, num_depth_maps)
    check_data_right = CheckData(num_images_right, num_depth_maps)

    check_passed_left = check_data_left.getCheck()
    check_passed_right = check_data_right.getCheck()

    if(check_passed_left and check_passed_right):

        print('Beginning processing of data')

        calib_logs, focal_lengths, centre_y_vals = parseCalib(calib_files)
        
        # Stereo camera baseline distance
        baseline = 0.2986  

        # Camera height above road surface
        camera_height = 1.67

        # Height map limits
        max_height = 7
        min_height = -2

        # Looping through data with temporal downsampling to get frames every second
        counter = 0
        for index in range(0, num_depth_maps, 5):

            print(f'Processing image {index} of {num_depth_maps-1}')

            # Open images and pre-existing masks
            image_left = Image.open(str(images_left[index]))
            image_right = Image.open(str(images_right[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            # Getting index associated with log
            data_log = str(str(images_left[index])[56:-69])
            log_index = calib_logs.index(data_log)
            
            # Reading focal length and principal point y-offset based on log index
            focal_length = focal_lengths[log_index]
            cy = centre_y_vals[log_index]

            # Create depth map
            sparse_depth_map = createDepthMap(depth_data, focal_length, baseline)

            # Fill in sparse depth map
            lidar_depth_fill = LidarDepthFill(sparse_depth_map)
            depth_map = lidar_depth_fill.getDepthMap()
            depth_map_fill_only = lidar_depth_fill.getDepthMapFillOnly()
            
            # Calculating depth boundaries
            depth_boundaries = findDepthBoundaries(depth_map_fill_only)

            # Height map
            heightMap = HeightMap(depth_map, max_height, min_height, 
                 camera_height, focal_length, cy)
            height_map = heightMap.getHeightMap()

            # Sparse supervision
            stereoSparseSupervision = StereoSparseSupervision(image_left, image_right, max_height, min_height, 
                    baseline, camera_height, focal_length, cy)
            sparse_supervision = stereoSparseSupervision.getSparseHeightMap()

            # Crop side regions where depth data is missing
            image_left, depth_map, depth_boundaries, height_map, sparse_supervision= \
                cropData(image_left, depth_map, depth_boundaries, height_map, sparse_supervision)

            # Save files
            # RGB Image as PNG
            image_save_path = root_save_path + '/image/' + str(counter) + '.png'
            image_left.save(image_save_path, "PNG")

            # Depth map as binary file in .npy format
            depth_save_path = root_save_path + '/depth/' + str(counter) + '.npy'
            np.save(depth_save_path, depth_map)

            # Height map as binary file in .npy format
            height_save_path = root_save_path + '/height/' + str(counter) + '.npy'
            np.save(height_save_path, height_map)

            # Sparse supervision map as binary file in .npy format
            supervision_save_path = root_save_path + '/supervision/' + str(counter) + '.npy'
            np.save(supervision_save_path, sparse_supervision)

            # Boundary mask as PNG
            boundary_save_path = root_save_path + '/boundary/' + str(counter) + '.png'
            boundary_mask = Image.fromarray(depth_boundaries)
            boundary_mask.save(boundary_save_path, "PNG")
            
            counter += 1

        print('----- Processing complete -----')         
            

if __name__ == '__main__':
    main()
#%%