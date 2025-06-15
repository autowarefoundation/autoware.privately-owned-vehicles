#! /usr/bin/env python3

import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import numpy as np
import sys

sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_path_network import EgoPathNetwork
from data_utils.augmentations import Augmentations


class EgoPathTrainer():
    def __init__(
        self,  
        checkpoint_path = "", 
        pretrained_checkpoint_path = "", 
        is_pretrained = False
    ):
        
        # Image and gts
        self.image = None
        self.xs = []
        self.ys = []
        self.flags = []

        # Dims
        self.height = 640
        self.width = 320

        # Tensors
        self.image_tensor = None
        self.xs_tensor = []
        self.flags_tensor = []

        # Model and pred
        self.model = None
        self.prediction = None

        # Losses
        self.loss = 0
        self.data_loss = 0
        self.smoothing_loss = 0
        self.flag_loss = 0
        self.gradient_type = "NUMERICAL"

        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        if (is_pretrained):

            # Instantiate model for validation or inference - load both pre-trained SceneSeg and SuperDepth weights
            if (len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = EgoPathNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load(
                    checkpoint_path, 
                    weights_only = True, 
                    map_location = self.device
                ))
                print("Loading pre-trained model weights of EgoPath from a saved checkpoint file")
            else:
                raise ValueError("Please ensure EgoPath network weights are provided for downstream elements")
            
        else:

            # Instantiate Model for training - load pre-trained SceneSeg weights only
            if (len(pretrained_checkpoint_path) > 0):
                
                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load(
                    pretrained_checkpoint_path, 
                    weights_only = True, 
                    map_location = self.device
                ))
                    
                # Loading model with pre-trained upstream weights
                self.model = EgoPathNetwork(sceneSegNetwork)
                print("Loading pre-trained backbone model weights only, EgoPath initialised with random weights")
            else:
                raise ValueError("Please ensure EgoPath network weights are provided for upstream elements")
            
        # Model to device
        self.model = self.model.to(self.device)
        
        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    
    # Zero gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Learning rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Image agumentations
    def apply_augmentations(self, is_train):
        if(is_train):
            # Augmenting data for training
            augTrain = Augmentations(
                is_train = True, 
                data_type = "KEYPOINTS"
            )
            augTrain.setImage(self.image)
            self.image = augTrain.applyTransformKeypoint(self.image)
        else:
            # Augmenting data for testing/validation
            augVal = Augmentations(
                is_train = False, 
                data_type = "KEYPOINTS"
            )
            augVal.setImage(self.image)
            self.image = augVal.applyTransformKeypoint(self.image)

    # Load Data as Pytorch Tensors
    def load_data(self):
        
        # Converting image to Pytorch Tensor
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

        # Converting keypoint list to Pytorch Tensor
        # List is in x0,y0,x1,y1,....xn, yn format
        gt_tensor = torch.from_numpy(self.gt)
        gt_tensor = gt_tensor.unsqueeze(0)
        self.gt_tensor = gt_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        self.loss = self.calc_loss(self.prediction, self.gt_tensor)

    # Calculate loss
    def calc_loss(self, prediction, ground_truth):

        # Endpoint loss - align the end control point of the Prediciton
        # vs Ground Truth Bezier Curves
        #self.endpoint_loss = self.calc_endpoints_loss(prediction, ground_truth)

        # Mid-point loss - similar to the BezierLaneNet paper, this loss ensures that
        # points along the curve have small x and y deviation - also acts as a regulariation term
        #self.mid_point_loss = self.calc_mid_points_loss(prediction, ground_truth)

        # Gradient Loss - either NUMERICAL tangent angle calcualation or
        # ANALYTICAL derviative of bezier curve, this loss ensures the curve is 
        # smooth and acts as a regularization term
        #if(self.gradient_type == 'NUMERICAL'):
        #    self.gradient_loss = self.calc_numerical_gradient_loss(prediction, ground_truth)
        #elif(self.gradient_type == 'ANALYTICAL'):
        #    self.gradient_loss = self.calc_analytical_gradient_loss(prediction, ground_truth)

        # Total loss is sum of individual losses multiplied by scailng factors
        #total_loss = self.gradient_loss*self.grad_scale_factor + \
        #    self.mid_point_loss*self.mid_point_scale_factor + \
        #    self.endpoint_loss*self.endpoint_loss_scale_factor

        self.start_point_x_offset_loss = self.calc_start_point_x_offset_loss(prediction, ground_truth)
        self.heading_angle_loss = self.calc_heading_angle_loss(prediction, ground_truth)
        #self.endpoint_loss = self.calc_endpoint_loss(prediction, ground_truth)

        #total_loss = self.start_point_x_offset_loss*self.start_point_x_offset_loss_scale_factor + \
        #            self.endpoint_loss*self.endpoint_loss_scale_factor
        total_loss = self.start_point_x_offset_loss + self.heading_angle_loss
        return total_loss 