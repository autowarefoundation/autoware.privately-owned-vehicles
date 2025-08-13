#! /usr/bin/env python3
import pathlib
import numpy as np
from typing import Literal
from PIL import Image
from .check_data import CheckData

class LoadDataEgoSpace():
    def __init__(self, labels_filepath, images_filepath, dataset: Literal['ZENSEACT', 'MAPILLARY', 'COMMA10K']):

        # Validate the dataset name
        if(dataset != 'ZENSEACT' and dataset != 'MAPILLARY' and dataset != 'COMMA10K'):
            raise ValueError('Dataset type is not correctly specified')
        self.dataset = dataset

        # Sort data and get list of input images and ground truth labels
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.png")])
        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*.png")])

        # Number of input images and ground truth labels
        self.num_images = len(self.images)
        self.num_labels = len(self.labels)

        # Performing sanity checks to ensure samples are correct in number
        checkData = CheckData(self.num_images, self.num_labels)

        # Lists to store train/val data
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        # Number of train/val samples
        self.num_train_samples = 0
        self.num_val_samples = 0

        # If all checks have passed, get samples and assign to train/val splits
        if (checkData.getCheck()):
            # Assigning ground truth data to train/val split
            for count in range (0, self.num_images):

                if((count+1) % 10 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1

    # Getting number of train/val samples
    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    def extractROI(self, input_image, input_label):
        if(self.dataset == 'COMMA10K'):
            input_image_height = input_image.height
            input_image_width = input_image.width

            input_image = input_image.crop((0, 0, input_image_width-1, int(input_image_height*(0.7))))
            input_label = input_label.crop((0, 0, input_image_width-1, int(input_image_height*(0.7))))

        return input_image, input_label

    def createGroundTruth(self, input_label):
        # Colourmaps for classes
        if (self.dataset == 'ZENSEACT'):
            road_colour = (255, 255, 255)
        else:
            road_colour = (0, 255, 220)

        # Image Size
        row, col = input_label.size

        # Ground Truth Label
        ground_truth_road = Image.new(mode="L", size=(row,col))

        # Loading images
        px = input_label.load()
        rx = ground_truth_road.load()

        # Extracting classes and assigning to colourmap
        for x in range(row):
            for y in range(col):
                if px[x,y] == road_colour:
                    rx[x, y] = 255

        # Getting ground truth data
        ground_truth = np.array(ground_truth_road)

        return ground_truth

    def getItemTrainPath(self, index):
        return str(self.train_images[index]), str(self.train_labels[index])

    # Get training data in numpy format
    def getItemTrain(self, index):
        train_image = Image.open(str(self.train_images[index]))
        train_label = Image.open(str(self.train_labels[index]))

        train_image, train_label = self.extractROI(train_image, train_label)
        train_ground_truth = self.createGroundTruth(train_label)

        return  np.array(train_image), np.expand_dims(train_ground_truth, axis=-1)

    def getItemValPath(self, index):
        return str(self.val_images[index]), str(self.val_labels[index])

    # Get training data in numpy format
    def getItemVal(self, index):
        val_image = Image.open(str(self.val_images[index])).convert('RGB')
        val_label = Image.open(str(self.val_labels[index]))

        val_image, val_label = self.extractROI(val_image, val_label)
        val_ground_truth = self.createGroundTruth(val_label)

        return  np.array(val_image), np.expand_dims(val_ground_truth, axis=-1)
