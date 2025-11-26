#! /usr/bin/env python3
import pathlib
from argparse import ArgumentParser
from PIL import Image
import sys
sys.path.append('../../../')
from Models.data_utils.check_data import CheckData


# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap, skyMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px = colorMap.load()

    # Loading and resizing sky pixel mask
    skyMap = skyMap.resize((row,col))
    ps = skyMap.load()

    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Colourmaps for classes
    background_objects_colour = (61, 93, 255)
    foreground_objects_colour = (255, 28, 145)
    road_colour = (0, 255, 220)
    sky_colour = (61, 184, 255)


    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            if type(px[x,y]) == int:
                # BACKGROUND OBJECTS
                if px[x,y] == 2:
                    cx[x,y] = background_objects_colour

                # EGO VEHICLE
                elif px[x,y] == 4:
                    cx[x,y] = background_objects_colour

                # MOVABLE OBJECTS
                elif px[x,y] == 1:
                    cx[x,y] = foreground_objects_colour

                # LANE MARKINGS
                elif px[x,y] == 0:
                    cx[x,y] = road_colour

                # ROAD
                elif px[x,y] == 3:
                    cx[x,y] = road_colour

                # SKY
                if ps[x,y] == sky_colour:
                    cx[x,y] = sky_colour

            else:
                # BACKGROUND OBJECTS   
                if px[x,y][0:3] == (128, 128, 96):
                    cx[x,y] = background_objects_colour

                # EGO VEHICLE
                elif px[x,y][0:3] == (204, 0, 255):
                    cx[x,y] = background_objects_colour

                # MOVABLE OBJECTS
                elif px[x,y][0:3] == (0, 255, 102):
                    cx[x,y] = foreground_objects_colour

                # LANE MARKINGS
                elif px[x,y][0:3] == (255, 0, 0):
                    cx[x,y] = road_colour

                # ROAD
                elif px[x,y][0:3] == (64, 32, 32):
                    cx[x,y] = road_colour

                # SKY
                if ps[x,y] == sky_colour:
                    cx[x,y] = sky_colour

    return coarseSegColorMap

def main():

    parser = ArgumentParser()
    parser.add_argument("-l", "--labels", dest="labels_filepath", help="path to folder with input ground truth labels")
    parser.add_argument("-sm", "--sky-masks", dest="sky_masks_filepath", help="path to folder with ground truth sky mask labels")
    parser.add_argument("-i", "--images", dest="images_filepath", help="path to folder with input images")
    parser.add_argument("-ls", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-is", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    # Paths to read input images and ground truth label masks from training data
    labels_filepath = args.labels_filepath
    images_filepath = args.images_filepath
    sky_masks_filepath = args.sky_masks_filepath

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.labels_save_path
    images_save_path = args.images_save_path

    # Reading dataset labels and images and sorting returned list in alphabetical order
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])
    sky_masks = sorted([f for f in pathlib.Path(sky_masks_filepath).glob("*.png")])

    # Getting number of labels and images
    num_labels = len(labels)
    num_images = len(images)
    num_sky_masks = len(sky_masks)

    # Check if sample numbers are correct
    check_data = CheckData(num_images, num_labels)
    check_data_sky = CheckData(num_images, num_sky_masks)
    check_passed = check_data.getCheck()
    check_sky_passed = check_data_sky.getCheck()

    # If all data checks have been passed
    if(check_passed and check_sky_passed):

        print('Beginning processing of data')

        # Looping through data
        for index in range(0, len(labels)):

            # Open images and pre-existing masks
            image = Image.open(str(images[index])).convert('RGB')
            label = Image.open(str(labels[index]))
            sky_mask = Image.open(str(sky_masks[index]))

            # Create new Coarse Segmentation mask
            coarseSegColorMap = createMask(label, sky_mask)
        
            # Save images
            image.save(images_save_path + str(index) + ".png","PNG")
            coarseSegColorMap.save(labels_save_path + str(index) + ".png","PNG")

            print(f'Processing image {index} of {num_images-1}')
    
        print('----- Processing complete -----') 

if __name__ == '__main__':
    main()