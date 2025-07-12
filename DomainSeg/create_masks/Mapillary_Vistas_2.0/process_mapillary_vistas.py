#! /usr/bin/env python3
import pathlib
from argparse import ArgumentParser
from PIL import Image
import os
import numpy as np


# Create coarse semantic segmentation mask
# of combined classes
def createMask(colorMap):

    # Initializing and loading pixel data
    row, col = colorMap.size
    px =  np.asarray(colorMap.convert("RGB"), dtype='float32')  #colorMap.load() #np.array(colorMap) #
    coarseSegColorMap = Image.new(mode="RGB", size=(row, col))
    cx = coarseSegColorMap.load()

    # Initial class colours
    object_traffic_cone = np.array([210, 60, 60], dtype='float32')
    construction_barrier_temporary = np.array([250, 170, 35], dtype='float32')

    # new class Mask
    binary_one  = (255, 255, 255)
    binary_zero = (0,   0,   0  )

    # to filter out image/label without desired class
    is_class_present = False

    # Extracting classes and assigning to colourmap
    for x in range(row):
        for y in range(col):

            if (  px[y][x][0] == object_traffic_cone[0] and px[y][x][1] == object_traffic_cone[1] and px[y][x][2] == object_traffic_cone[2]  ):
                cx[x,y] = binary_one
                is_class_present = True
     
            elif (  px[y][x][0] == construction_barrier_temporary[0] and px[y][x][1] == construction_barrier_temporary[1] and px[y][x][2] == construction_barrier_temporary[2]  ):
                cx[x,y] = binary_one
                is_class_present = True
            
            else:
                cx[x,y] = binary_zero
            

    #coarseSegColorMap = Image.fromarray(coarseSegColorMap)
            
    return coarseSegColorMap, is_class_present
    

def main():
    
    parser = ArgumentParser()
    parser.add_argument("-trlb", "--trainlabels", dest="train_labels_filepath", help="path to folder with train ground truth labels")
    parser.add_argument("-trim", "--trainimages", dest="train_images_filepath", help="path to folder with train images")
    parser.add_argument("-valb", "--valilabels", dest="val_labels_filepath", help="path to folder with validation ground truth labels")
    parser.add_argument("-vaim", "--valiimages", dest="val_images_filepath", help="path to folder with validation images")
    parser.add_argument("-lbs", "--labels-save", dest="labels_save_path", help="path to folder where processed labels will be saved")
    parser.add_argument("-ims", "--images-save", dest="images_save_path", help="path to folder where corresponding images will be saved")
    args = parser.parse_args()

    # Paths to read input images and ground truth label masks from validation data
    val_labels_filepath = args.val_labels_filepath
    val_images_filepath = args.val_images_filepath

    # Paths to read input images and ground truth label masks from training data
    train_labels_filepath = args.train_labels_filepath
    train_images_filepath = args.train_images_filepath

    # Paths to save training data with new coarse segmentation masks
    labels_save_path = args.labels_save_path
    images_save_path = args.images_save_path
    
  

    ## create folders to save images and labels
    folder_path =  images_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    folder_path =  labels_save_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

    index_for_saving = 0
        

    # Reading validation dataset labels and images and sorting returned list in alphabetical order
    val_labels = sorted([f for f in pathlib.Path(val_labels_filepath).glob("*.png")])
    val_images = sorted([f for f in pathlib.Path(val_images_filepath).glob("*.jpg")])

    # Getting number of labels and images
    val_num_labels = len(val_labels)
    val_num_images = len(val_images)

    print("number of validation labels is {}".format(val_num_labels))
    print("number of validation images is {}".format(val_num_images))

    # Check if sample numbers are correct
    check_passed = False
    if (val_num_labels == val_num_images):
        check_passed = True
    
    # If all data checks have been passed
    if(check_passed):

        print('Beginning processing of validation data')
        

        # Looping through data
        for index in range(0, val_num_images):
            
            # Open images and pre-existing masks
            image = Image.open(str(val_images[index]))
            label = Image.open(str(val_labels[index]))

            # Create new Coarse Segmentation mask
            coarseSegColorMap, is_class_present  = createMask(label)

            if (is_class_present == True):
                # Save images
                image.save(images_save_path  + '/' + str(index_for_saving) + ".png","PNG")
                coarseSegColorMap.save(labels_save_path + '/' + str(index_for_saving) + ".png","PNG")
                print(f'Processing image {index} of {val_num_images-1}')
                index_for_saving = index_for_saving + 1
            else:
                print("the image {} does not have the required class/classes".format(index))
    
        print('----- Processing validation complete -----') 

    
    # Reading training dataset labels and images and sorting returned list in alphabetical order
    train_labels = sorted([f for f in pathlib.Path(train_labels_filepath).glob("*.png")])
    train_images = sorted([f for f in pathlib.Path(train_images_filepath).glob("*.jpg")])

    # Getting number of labels and images
    train_num_labels = len(train_labels)
    train_num_images = len(train_images)

    print("number of training labels is {}".format(train_num_labels))
    print("number of training images is {}".format(train_num_images))

    # Check if sample numbers are correct
    check_passed = False
    if (train_num_labels == train_num_images):
        check_passed = True
    
    # If all data checks have been passed
    if(check_passed):

        print('Beginning processing of training data')

        # Looping through data
        for index in range(0, train_num_images):
            
            # Open images and pre-existing masks
            image = Image.open(str(train_images[index]))
            label = Image.open(str(train_labels[index]))

            # Create new Coarse Segmentation mask
            coarseSegColorMap, is_class_present  = createMask(label)

            if (is_class_present == True):
                # Save images
                image.save(images_save_path + '/' + str(index_for_saving) + ".png","PNG")
                coarseSegColorMap.save(labels_save_path + '/' + str(index_for_saving) + ".png","PNG")
                print(f'Processing image {index} of {train_num_images-1}')
                index_for_saving = index_for_saving + 1
            else:
                print("the image {} does not have the required class/classes".format(index))
    
        print('----- Processing validation complete -----') 



if __name__ == '__main__':
    main()