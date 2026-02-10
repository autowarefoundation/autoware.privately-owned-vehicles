#! /usr/bin/env python3
import pathlib
import numpy as np
from PIL import Image
from .check_data import CheckData

class LoadDataScene3D():
    def __init__(self, labels_filepath, images_filepath):
        
        # Getting data list and counting number of training samples and ground truth samples
        self.labels = sorted([f for f in pathlib.Path(labels_filepath).glob("*.npy")])
        self.num_labels = len(self.labels)

        self.images = sorted([f for f in pathlib.Path(images_filepath).glob("*")])
        self.num_images = len(self.images)

        # Performing sanity checks to ensure samples are correct in number
        checkData = CheckData(self.num_images, self.num_labels)

        # Lists to store Train/Val split
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        # Number of Train/Val Samples
        self.num_train_samples = 0
        self.num_val_samples = 0
  
        # If all checks have passed, get samples and assign to train/val splits
        if (checkData.getCheck()):
            for count in range (0, self.num_images):

                if((count+1) % 20 == 0):
                    self.val_images.append(str(self.images[count]))
                    self.val_labels.append(str(self.labels[count]))
                    self.num_val_samples += 1 
                else:
                    self.train_images.append(str(self.images[count]))
                    self.train_labels.append(str(self.labels[count]))
                    self.num_train_samples += 1

    # Get number of Train/Val samples
    def getItemCount(self):
        return self.num_train_samples, self.num_val_samples

    # Get training data in numpy format
    def getItemTrain(self, index):
        train_image = Image.open(str(self.train_images[index])).convert('RGB')
        train_ground_truth = np.load(str(self.train_labels[index]))
        train_ground_truth =  \
            (train_ground_truth - np.min(train_ground_truth)) \
            / (np.max(train_ground_truth) - np.min(train_ground_truth)) 

        train_ground_truth = np.expand_dims(train_ground_truth, axis=-1)
        return  np.array(train_image), train_ground_truth
    
    # Get validation data in numpy format
    def getItemVal(self, index):
        val_image = Image.open(str(self.val_images[index])).convert('RGB')
        val_ground_truth = np.load(str(self.val_labels[index]))
        val_ground_truth =  \
            (val_ground_truth - np.min(val_ground_truth)) \
            / (np.max(val_ground_truth) - np.min(val_ground_truth)) 
        val_ground_truth = np.expand_dims(val_ground_truth, axis=-1)
        return  np.array(val_image), val_ground_truth

    
