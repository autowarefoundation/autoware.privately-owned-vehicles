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
    image_left = image_left.crop((40, 0, width - 40, height))
    depth_map_fill_only = depth_map_fill_only[:, 40 : width - 40]
    validity_mask = validity_mask[:, 40 : width - 40]

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
    depth_filepath = root_data_path + 'depth-raw/'
    images_filepath = root_data_path + 'image-raw/'

    # Reading dataset labels and images and sorting returned list in alphabetical order
    depth_maps = sorted([f for f in pathlib.Path(depth_filepath).glob("*.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    # If all data checks have been passed
    num_depth_maps = len(depth_maps)
    num_images = len(images)

    check_data = CheckData(num_images, num_depth_maps)
    check_passed = check_data.getCheck()


    if(check_passed):

        print('Beginning processing of data')

        # Looping through data with temporal downsampling to get frames every second
        counter = 0
        
        for index in range(0, num_images):

            print(f'Processing image {index} of {num_depth_maps-1}')
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            # Create depth map
            sparse_depth_map = createDepthMap(depth_data)
            lidarDepthFill = LidarDepthFill(sparse_depth_map)
            depth_map = lidarDepthFill.getDepthMap()
        
            # Validity mask
            validity_mask = np.zeros_like(depth_map)
            validity_mask[np.where(depth_map != 0)] = 1

            # Crop side regions where depth data is missing
            image, depth_map, validity_mask = \
                cropData(image, depth_map, validity_mask)

            # Save files
            # RGB Image as PNG
            image_save_path = root_save_path + '/image/' + str(index) + '.jpg'
            image.save(image_save_path)

            # Depth map as binary file in .npy format
            depth_save_path = root_save_path + '/depth/' + str(index) + '.npy'
            np.save(depth_save_path, depth_map)

            # Validity mask as black and white PNG
            validity_save_path = root_save_path + '/validity/' + str(index) + '.png'
            validity_mask = Image.fromarray(np.uint8(validity_mask*255))
            validity_mask.save(validity_save_path, "PNG")
            
            counter += 1
        
        print('----- Processing complete -----') 
                

if __name__ == '__main__':
    main()
#%%