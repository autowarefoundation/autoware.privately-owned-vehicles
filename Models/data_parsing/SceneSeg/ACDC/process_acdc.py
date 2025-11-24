#! /usr/bin/env python3
import pathlib
from argparse import ArgumentParser
from PIL import Image
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData


# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    sky_colour = (61, 184, 255)
    background_objects_colour = (61, 93, 255)
    vulnerable_living_colour = (255, 61, 61)
    small_mobile_vehicle_colour = (255, 190, 61)
    large_mobile_vehicle_colour = (255, 116, 61)
    road_edge_delimiter_colour = (216, 255, 61)
    road_colour = (0, 255, 220)

    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            # SKY
            if px[x, y] == (70,130,180):
                cx[x,y] = sky_colour

            # BACKGROUND OBJECTS
            # Building    
            elif px[x,y] == (70, 70, 70):
                cx[x,y] = background_objects_colour
            # Pole  
            elif px[x,y] == (153, 153, 153):
                cx[x,y] = background_objects_colour
            # Traffic Light
            elif px[x,y] == (250, 170, 30):
                cx[x,y] = background_objects_colour 
            # Traffic Sign
            elif px[x,y] == (220, 220, 0):
                cx[x,y] = background_objects_colour 
            # Vegetation
            elif px[x,y] == (107, 142, 35):
                cx[x,y] = background_objects_colour
            # Terrain
            elif px[x,y] == (152, 251, 152):
                cx[x,y] = background_objects_colour

            # VULNERABLE LIVING
            # Person
            elif px[x,y] == (220, 20, 60):
                cx[x,y] = vulnerable_living_colour

            # SMALL MOBILE VEHICLE
            # Rider
            elif px[x,y] == (255, 0, 0):
                cx[x,y] = small_mobile_vehicle_colour
            # Motorcylce
            elif px[x,y] == (0, 0, 230):
                cx[x,y] = small_mobile_vehicle_colour
            # Bicycle
            elif px[x,y] == (119, 11, 32):
                cx[x,y] = small_mobile_vehicle_colour

            # LARGE MOBILE VEHICLE
            # Car
            elif px[x,y] == (0, 0, 142):
                cx[x,y] = large_mobile_vehicle_colour
            # Truck
            elif px[x,y] == (0, 0, 70):
                cx[x,y] = large_mobile_vehicle_colour
            # Bus
            elif px[x,y] == (0, 60, 100):
                cx[x,y] = large_mobile_vehicle_colour
            # Train
            elif px[x,y] == (0, 80, 100):
                cx[x,y] = large_mobile_vehicle_colour
            
            # ROAD EDGE DELIMITER
            # Wall
            elif px[x,y] == (102, 102, 156):
                cx[x,y] = road_edge_delimiter_colour
            # Fence
            elif px[x,y] == (190, 153, 153):
                cx[x,y] = road_edge_delimiter_colour

            # ROAD
            elif px[x,y] == (128, 64, 128):
                cx[x,y] = road_colour

    return coarseSegColorMap

def main():

    parser = ArgumentParser()
    parser.add_argument("-l", "--labels", dest="labels_filepath", help="path to folder with input ground truth labels")
    parser.add_argument("-i", "--images", dest="images_filepath", help="path to folder with input images")
    parser.add_argument("-ls", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-is", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    # Paths to read input images and ground truth label masks from training data
    labels_filepath = args.labels_filepath
    images_filepath = args.images_filepath

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.labels_save_path
    images_save_path = args.images_save_path

    # Reading dataset labels and images and sorting returned list in alphabetical order
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*labelColor.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

    # Getting number of labels and images
    num_labels = len(labels)
    num_images = len(images)

    # Check if sample numbers are correct
    check_data = CheckData(num_images, num_labels)
    check_passed = check_data.getCheck()
    
    # If all data checks have been passed
    if(check_passed):

        print('Beginning processing of data')

        # Looping through data
        for index in range(0, num_images):
            
            # Open images and pre-existing masks
            image = Image.open(str(images[index]))
            label = Image.open(str(labels[index]))

            # Create new Coarse Segmentation mask
            coarseSegColorMap = createMask(label)
            
            # Save images
            image.save(images_save_path + str(index) + ".png","PNG")
            coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            print(f'Processing image {index} of {num_images-1}')
    
        print('----- Processing complete -----') 

if __name__ == '__main__':
    main()