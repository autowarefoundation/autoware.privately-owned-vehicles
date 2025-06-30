#%%
#! /usr/bin/env python3
import pathlib
import numpy as np
import os
from argparse import ArgumentParser
from PIL import Image

# Simply function which makes a directory if it doesn't exist
def checkDir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    

# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap, image):

    # Getting colormap
    vals = np.array(colorMap)

    # Checking size of colormap
    shape = vals.shape
    row = shape[0]
    col = shape[1]

    # Initializing segmentation and visualization mask
    segMask = np.zeros((row, col), dtype='uint8')
    visMask = np.array(image)
    
    # Getting foreground object labels
    cone = np.where(vals == 13)
    drum = np.where(vals == 14)
    vertical_panel = np.where(vals == 15)
    tubular_marker = np.where(vals == 16)

    # Assign to binary mask
    segMask[cone[0], cone[1]] = 255
    segMask[drum[0], drum[1]] = 255
    segMask[vertical_panel[0], vertical_panel[1]] = 255
    segMask[tubular_marker[0], tubular_marker[1]] = 255   

    # Assign to visualization mask
    full_labels = np.where(segMask == 255)
    visMask[full_labels[0], full_labels[1], 0] = 255
    visMask[full_labels[0], full_labels[1], 1] = 200
    visMask[full_labels[0], full_labels[1], 2] = 0

    return segMask, visMask

def main():

    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_root_path", dest="data_root_path", help="path to folder with ground truth data")
    parser.add_argument("-s", "--save_root_path", dest="save_root_path", help="path to where processed data should be saved")
    args = parser.parse_args()
    
    # Paths to read input images and ground truth label masks from training data
    labels_filepath = args.data_root_path + 'gtFine/'
    images_filepath = args.data_root_path + 'images/'

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.save_root_path + 'label/'
    images_save_path = args.save_root_path + 'image/'
    visualization_save_path = args.save_root_path + 'visualization/'

    # Check output save directories - if they don't exist, create them
    checkDir(labels_save_path)
    checkDir(images_save_path)
    checkDir(visualization_save_path)

    # Reading dataset labels and images and sorting returned list in alphabetical order
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*")])
    labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*labelIds.png")])
    
    # Printing number of images and ground truth labels found
    print('Found ', len(images), ' images')
    print('Found ', len(labels), ' ground truth labels')

    # Process images
    for i in range(0, len(images)):
        
        # Print progress
        print('Processing ', i+1, ' of ' , len(images))

        # Read image and get ID
        image_file = str(images[i])
        image_file_id = image_file[len(images_filepath):] # image_file.replace(images_filepath, '')
        image_file_id = image_file_id[0:-4]

        # Get corresponding label ID from image ID
        label_file = labels_filepath + image_file_id + '_labelIds.png'

        try:
            # If we have a valid label, open it
            label = Image.open(label_file)

            # Open the Image and get its width and height
            image = Image.open(image_file)
            width, height = image.size

            # Crop to achieve a 2:1 width to height aspect ratio if the image is too tall
            if(height > width/2):

                image = image.crop((0, height/2 - width/4, width-1, height/2 + width/4))
                label = label.crop((0, height/2 - width/4, width-1, height/2 + width/4))
            
            # Create the label and visualiztion
            label_mask, vis_mask = createMask(label, image)

            # Apply alpha transparency factor of 0.5
            label_mask_composite = np.uint8(label_mask*0.5)

            # Save the image
            image.save(images_save_path + str(i).zfill(4) + ".jpg")

            # Save the ground truth segmentation mask
            mask = Image.fromarray(label_mask)
            mask.save(labels_save_path + str(i).zfill(4) + ".png", "PNG")

            # Create the visualization through image compositing
            vis = Image.fromarray(vis_mask)
            label_mask_composite = Image.fromarray(label_mask_composite)
            visualization = Image.composite(image, vis, label_mask_composite)

            # Save the visualization for data auditing
            visualization.save(visualization_save_path + str(i).zfill(4) + ".png", "PNG")

        except FileNotFoundError:
            print('Label not found')

    print('Finished processing')

if __name__ == '__main__':
    main()
#%%