#! /usr/bin/env python3
#%%
import pathlib
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData
from Scene3D.create_metric_depth.common.lidar_depth_fill import LidarDepthFill

def removeExtraSamples(image_folders):
    
    filtered_images_left = []
    filtered_images_right = []

    for i in range(0, len(image_folders)):
        images_left = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_02/*data/*.png")])
        images_right = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_03/*data/*.png")])
        for j in range(5, len(images_left) - 5):
            filtered_images_left.append(images_left[j])
            filtered_images_right.append(images_right[j])

    return filtered_images_left, filtered_images_right

def createDepthMap(depth_data):

    assert(np.max(depth_data) > 255)
    depth_map = depth_data.astype('float32') / 256.
    return depth_map


def cropData(image_left, depth_map_fill_only, validity_mask):

    # Getting size of depth map
    size = depth_map_fill_only.shape
    height = size[0]
    width = size[1]

    # Cropping out those parts of data for which depth is unavailable
    image_left = image_left.crop((256, 0, width - 100, height))
    depth_map_fill_only = depth_map_fill_only[:, 256 : width - 100]
    validity_mask = validity_mask[:, 256 : width - 100]

    return image_left, depth_map_fill_only, validity_mask


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
    depth_filepath = root_data_path + 'train/'
    images_filepath = root_data_path + 'data/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*/proj_depth/*groundtruth/*image_02/*.png")])
    image_folders = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    # Remove extra samples
    images_left, images_right = removeExtraSamples(image_folders)

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


        for index in range(15815, num_depth_maps):

            print(f'Processing image {index} of {num_depth_maps-1}')
            
            # Open images and pre-existing masks
            image_left = Image.open(str(images_left[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            # Create depth map
            sparse_depth_map = createDepthMap(depth_data)
            lidarDepthFill = LidarDepthFill(sparse_depth_map)
            depth_map = lidarDepthFill.getDepthMap()
            
            # Validity mask
            validity_mask = np.zeros_like(depth_map)
            validity_mask[np.where(depth_map != 0)] = 1

            # Crop side regions where depth data is missing
            image_left, depth_map, validity_mask = \
                cropData(image_left, depth_map, validity_mask)

            # Save files
            # RGB Image as JPG
            image_save_path = root_save_path + 'image-full/' + str(index) + '.jpg'
            image_left.save(image_save_path)

            # Depth map as binary file in .npy format
            depth_save_path = root_save_path + 'depth-full/' + str(index) + '.npy'
            np.save(depth_save_path, depth_map)

            # Validity mask as black and white PNG
            validity_save_path = root_save_path + 'validity-full/' + str(index) + '.png'
            validity_mask = Image.fromarray(np.uint8(validity_mask*255))
            validity_mask.save(validity_save_path, "PNG")


        print('----- Processing complete -----') 
                

if __name__ == '__main__':
    main()
#%%