#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image

class LoadDataSceneSeg():
    def __init__(self, labels_filepath, images_filepath, \
        dataset: Literal['ACDC', 'BDD100K', 'IDDAW', 'MUSES', 'MAPILLARY', 'COMMA10K']):

        self.dataset = dataset

        if(self.dataset != 'ACDC' and self.dataset != 'BDD100K' and  \
           self.dataset != 'IDDAW' and self.dataset != 'MUSES' \
           and self.dataset != 'MAPILLARY' and self.dataset != 'COMMA10K'):
            raise ValueError('Dataset type is not correctly specified')
        
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        if(self.num_images != self.num_labels):
            raise ValueError('Number of images and ground truth labels are mismatched')
        
        if (self.num_images == 0):
            raise ValueError('No images found - check the root path')
       
        if (self.num_labels == 0):
            raise ValueError('No ground truth masks found - check the root path')
        
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []
        
        self.num_train_samples = 0
        self.num_val_samples = 0

        for count in range (0, self.num_images):
    
            if((count+1) % 10 == 0):
                self.val_images.append(str(self.images[count]))
                self.val_labels.append(str(self.labels[count]))
                self.num_val_samples += 1 
            else:
                self.train_images.append(str(self.images[count]))
                self.train_labels.append(str(self.labels[count]))
                self.num_train_samples += 1

    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples
    
    def createGroundTruth(self, input_label):
        # Colourmaps for classes
        sky_colour = (61, 184, 255)
        background_objects_colour = (61, 93, 255)
        road_edge_delimiter_colour = (216, 255, 61)
        unlabelled_colour = (0,0,0)
        vulnerable_living_colour = (255, 61, 61)
        small_mobile_vehicle_colour = (255, 190, 61)
        large_mobile_vehicle_colour = (255, 116, 61)
        foreground_objects_colour = (255, 28, 145)
        road_colour = (0, 255, 220)

        # Image Size
        row, col = input_label.size
        num_pixels = row*col

        # Ground Truth Visualization
        vis = Image.new(mode="RGB", size=(row, col))

        # Ground Truth Multi-Channel Label
        ground_truth_sky = Image.new(mode="L", size=(row, col))
        ground_truth_background = Image.new(mode="L", size=(row, col))
        ground_truth_foreground = Image.new(mode="L", size=(row,col))
        ground_truth_road = Image.new(mode="L", size=(row,col))
   
        # Loading images
        px = input_label.load()
        vx = vis.load()
        sx = ground_truth_sky.load()
        bx = ground_truth_background.load()
        fx = ground_truth_foreground.load()
        rx = ground_truth_road.load()

        # Counters for pixel level class frequency in image
        background_class_freq = 0
        foreground_class_freq = 0
        road_class_freq = 0

        # Extracting classes and assigning to colourmap
        for x in range(row):
            for y in range(col):

                # BACKGROUND OBJECTS
                if px[x,y] == background_objects_colour or \
                    px[x,y] == road_edge_delimiter_colour or \
                    px[x,y] == unlabelled_colour or \
                    px[x,y] == sky_colour:

                    vx[x,y] = background_objects_colour
                    bx[x, y] = 255
                    background_class_freq += 1

                # FOREGROUND OBJECTS
                elif px[x,y] == vulnerable_living_colour or \
                    px[x,y] == small_mobile_vehicle_colour or \
                    px[x,y] == large_mobile_vehicle_colour or \
                    px[x,y] == foreground_objects_colour:

                    vx[x,y] = foreground_objects_colour
                    fx[x, y] = 255
                    foreground_class_freq += 1
                
                # ROAD
                elif px[x,y] == road_colour:

                    vx[x,y] = road_colour
                    rx[x, y] = 255
                    road_class_freq += 1

        # Calculate class weights for loss function
        class_weights = []

        background_class_weight = num_pixels/(background_class_freq + 5120)
        class_weights.append(background_class_weight)

        foreground_class_weight = num_pixels/(foreground_class_freq + 5120)
        class_weights.append(foreground_class_weight)

        road_class_weight = num_pixels/(road_class_freq + 5120)
        class_weights.append(road_class_weight)

        # Getting ground truth data
        ground_truth = []
        ground_truth.append(np.array(vis))
        ground_truth.append(np.array(ground_truth_background))
        ground_truth.append(np.array(ground_truth_foreground))
        ground_truth.append(np.array(ground_truth_road))

        return ground_truth, class_weights

    def extractROI(self, input_image, input_label):
        if(self.dataset == 'ACDC'):
            input_image = input_image.crop((0, 0, 1919, 990))
            input_label = input_label.crop((0, 0, 1919, 990))
        elif(self.dataset == 'BDD100K'):
            input_image = input_image.crop((0, 0, 1000, 500))
            input_label = input_label.crop((0, 0, 1000, 500))
        elif(self.dataset == 'IDDAW'):
            input_image = input_image.crop((0, 476, 2047, 1500))
            input_label = input_label.crop((0, 476, 2047, 1500))
        elif(self.dataset == 'MUSES'):
            input_image = input_image.crop((0, 0, 1919, 918))
            input_label = input_label.crop((0, 0, 1919, 918))
        elif(self.dataset == 'COMMA10K'):
            input_image_height = input_image.height 
            input_image_width = input_image.width 
            input_image = input_image.crop((0, 0, \
                input_image_width-1, int(input_image_height*(0.7))))
            input_label = input_label.crop((0, 0, \
                input_image_width-1, int(input_image_height*(0.7))))

        return input_image, input_label
    
    def getItemTrain(self, index):
        self.train_image = Image.open(str(self.train_images[index]))
        self.train_label = Image.open(str(self.train_labels[index]))
        
        self.train_image, self.train_label = \
            self.extractROI(self.train_image, self.train_label)
        self.train_ground_truth, self.train_class_weights = \
            self.createGroundTruth(self.train_label)
 
        return np.array(self.train_image), self.train_ground_truth, \
            self.train_class_weights

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])
    
    def getItemVal(self, index):
        self.val_image = Image.open(str(self.val_images[index]))
        self.val_label = Image.open(str(self.val_labels[index]))

        self.val_image, self.val_label = \
            self.extractROI(self.val_image, self.val_label)
        self.val_ground_truth, self.val_class_weights = \
            self.createGroundTruth(self.val_label)

        return np.array(self.val_image), self.val_ground_truth, \
            self.val_class_weights
    
    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])
