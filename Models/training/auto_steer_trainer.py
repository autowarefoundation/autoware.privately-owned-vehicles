#! /usr/bin/env python3

import torch
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal, get_args
import sys

sys.path.append('..')
from model_components.auto_steer_network import AutoSteerNetwork
from data_utils.augmentations import Augmentations
from data_utils.load_data_auto_steer import VALID_DATASET_LIST


class AutoSteerTrainer():
    def __init__(
        self,  
        checkpoint_path = ""
    ):
        
        # Initializing Data
        self.homotrans_mat = None
        self.bev_image = None
        self.binary_seg = None
        self.perspective_image = None
        self.bev_egopath = None
        self.bev_egoleft = None
        self.bev_egoright = None
        self.reproj_egopath = None
        self.reproj_egoleft = None
        self.reproj_egoright = None
        self.perspective_H = None
        self.perspective_W = None
        self.BEV_H = None
        self.BEV_W = None

        # Initializing BEV to Image transformation matrix
        self.homotrans_mat_tensor = None

        # Initializing BEV Image tensor
        self.bev_image_tensor = None

        # Initializing Binary Segmentation Mask tensor
        self.binary_seg_tensor = None

        # Initializing Ground Truth Tensors
        self.gt_bev_egopath_tensor = None
        self.gt_bev_egoleft_lane_tensor = None
        self.gt_bev_egoright_lane_tensor = None
        self.gt_reproj_egopath_tensor = None
        self.gt_reproj_egoleft_lane_tensor = None
        self.gt_reproj_egoright_lane_tensor = None

        # Model predictions
        self.pred_bev_ego_path_tensor = None
        self.pred_bev_egoleft_lane_tensor = None
        self.pred_bev_egoright_lane_tensor = None

        # Losses
        self.BEV_loss = None
        self.reprojected_loss = None
        self.total_loss = None

        self.BEV_FIGSIZE = (4, 8)
        self.ORIG_FIGSIZE = (8, 4)

        # Currently limiting to available datasets only. Will unlock eventually
        self.VALID_DATASET_LIST = VALID_DATASET_LIST
        
        # Checking devices (GPU vs CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using {self.device} for inference.")

        # Instantiate model
        self.model = AutoSteerNetwork()
        
        if(checkpoint_path):
            print("Loading trained AutoSteer model from checkpoint")
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only = True))  
        else:
            print("Loading vanilla AutoSteer model for training")
            
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

        self.binary_seg_loader = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    # Zero gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Learning rate adjustment
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        
    # Assign input variables
    def set_data(self, homotrans_mat, bev_image, perspective_image, binary_seg, \
                bev_egopath, bev_egoleft, bev_egoright, reproj_egopath, \
                reproj_egoleft, reproj_egoright):

        self.homotrans_mat = np.array(homotrans_mat, dtype = "float32")
        self.bev_image = np.array(bev_image)
        self.binary_seg = np.array(binary_seg)
        self.perspective_image = np.array(perspective_image)
        self.bev_egopath = np.array(bev_egopath, dtype = "float32").transpose()
        self.bev_egoleft = np.array(bev_egoleft, dtype = "float32").transpose()
        self.bev_egoright = np.array(bev_egoright, dtype = "float32").transpose()
        self.reproj_egopath = np.array(reproj_egopath, dtype = "float32").transpose()
        self.reproj_egoleft = np.array(reproj_egoleft, dtype = "float32").transpose()
        self.reproj_egoright = np.array(reproj_egoright, dtype = "float32").transpose()
        self.perspective_H, self.perspective_W, _ = self.perspective_image.shape
        self.BEV_H, self.BEV_W, _ = self.bev_image.shape

    # Image agumentations
    def apply_augmentations(self, is_train):
        # Augmenting data for train or val/test
        aug = Augmentations(
            is_train = is_train, 
            data_type = "KEYPOINTS"
        )
        aug.setImage(self.bev_image)
        self.bev_image = aug.applyTransformKeypoint(self.bev_image)

    # Load data as Pytorch tensors
    def load_data(self):

        # BEV to Image matrix
        homotrans_mat_tensor = torch.from_numpy(self.homotrans_mat)
        homotrans_mat_tensor = homotrans_mat_tensor.type(torch.FloatTensor)
        self.homotrans_mat_tensor = homotrans_mat_tensor.to(self.device)

        # BEV Image
        bev_image_tensor = self.image_loader(self.bev_image)
        bev_image_tensor = bev_image_tensor.unsqueeze(0)
        self.bev_image_tensor = bev_image_tensor.to(self.device)

        # Binary Segmentation
        binary_seg_tensor = self.binary_seg_loader(self.binary_seg)
        binary_seg_tensor = binary_seg_tensor.unsqueeze(0)
        self.binary_seg_tensor = binary_seg_tensor.to(self.device)

        # BEV Egopath
        bev_egopath_tensor = torch.from_numpy(self.bev_egopath)
        bev_egopath_tensor = bev_egopath_tensor.type(torch.FloatTensor)
        self.gt_bev_egopath_tensor = bev_egopath_tensor.to(self.device)

        # BEV Egoleft Lane
        bev_egoleft_lane_tensor = torch.from_numpy(self.bev_egoleft)
        bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.type(torch.FloatTensor)
        self.gt_bev_egoleft_lane_tensor = bev_egoleft_lane_tensor.to(self.device)

        # BEV Egoright Lane
        bev_egoright_lane_tensor = torch.from_numpy(self.bev_egoright)
        bev_egoright_lane_tensor = bev_egoright_lane_tensor.type(torch.FloatTensor)
        self.gt_bev_egoright_lane_tensor = bev_egoright_lane_tensor.to(self.device)
        
        # Reprojected Egopath
        reproj_egopath_tensor = torch.from_numpy(self.reproj_egopath)
        reproj_egopath_tensor = reproj_egopath_tensor.type(torch.FloatTensor)
        self.gt_reproj_egopath_tensor = reproj_egopath_tensor.to(self.device)

        # Reprojected Egoleft Lane
        reproj_egoleft_lane_tensor = torch.from_numpy(self.reproj_egoleft)
        reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.type(torch.FloatTensor)
        self.gt_reproj_egoleft_lane_tensor = reproj_egoleft_lane_tensor.to(self.device)

        # Reprojected Egoright Lane
        reproj_egoright_lane_tensor = torch.from_numpy(self.reproj_egoright)
        reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.type(torch.FloatTensor)
        self.gt_reproj_egoright_lane_tensor = reproj_egoright_lane_tensor.to(self.device)
    
    # Run Model
    def run_model(self):
        self.pred_bev_ego_path_tensor, self.pred_bev_egoleft_lane_tensor, \
            self.pred_bev_egoright_lane_tensor = self.model(self.bev_image_tensor)

        # BEV Loss
        BEV_data_loss_driving_corridor = self.calc_BEV_data_loss_driving_corridor()
        BEV_gradient_loss_driving_corridor = self.calc_BEV_gradient_loss_driving_corridor()

        self.BEV_loss = BEV_data_loss_driving_corridor + \
            BEV_gradient_loss_driving_corridor*0.5
   
        # Reprojected Loss
        reprojected_data_loss_driving_corridor = self.calc_reprojected_data_loss_driving_corridor()
        reprojected_gradient_loss_driving_corridor = self.calc_reprojected_gradient_loss_driving_corridor()

        self.reprojected_loss = reprojected_data_loss_driving_corridor + \
            reprojected_gradient_loss_driving_corridor*0.5

        # Total Loss
        self.total_loss = self.BEV_loss + self.reprojected_loss

    # BEV Data Loss for the entire driving corridor
    def calc_BEV_data_loss_driving_corridor(self):

        BEV_egopath_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egopath_tensor, self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoleft_lane_tensor, self.pred_bev_egoleft_lane_tensor)
   
        BEV_egoright_lane_data_loss = \
            self.calc_BEV_data_loss(self.gt_bev_egoright_lane_tensor, self.pred_bev_egoright_lane_tensor)


        BEV_data_loss_driving_corridor = BEV_egopath_data_loss +  \
            BEV_egoleft_lane_data_loss + BEV_egoright_lane_data_loss
        
        return BEV_data_loss_driving_corridor
    
    # BEV Gradient Loss for the entire driving corridor
    def calc_BEV_gradient_loss_driving_corridor(self):

        BEV_egopath_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egopath_tensor, 
                                       self.pred_bev_ego_path_tensor)

        BEV_egoleft_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoleft_lane_tensor, 
                                       self.pred_bev_egoleft_lane_tensor)
   
        BEV_egoright_lane_gradient_loss = \
            self.calc_BEV_graient_loss(self.gt_bev_egoright_lane_tensor, 
                                       self.pred_bev_egoright_lane_tensor)


        BEV_gradient_loss_driving_corridor = BEV_egopath_gradient_loss +  \
            BEV_egoleft_lane_gradient_loss + BEV_egoright_lane_gradient_loss
        
        return BEV_gradient_loss_driving_corridor
    
    # Reprojected Data Loss for the entire driving corridor
    def calc_reprojected_data_loss_driving_corridor(self):

        reprojected_ego_path_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_data_loss =  \
            self.calc_reprojected_data_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_data_loss_driving_corridor = reprojected_ego_path_data_loss + \
            reprojected_egoleft_lane_data_loss + reprojected_egoright_lane_data_loss
        
        return reprojected_data_loss_driving_corridor
    
    # Reprojected Gradient Loss for the entire driving corridor
    def calc_reprojected_gradient_loss_driving_corridor(self):

        reprojected_ego_path_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egopath_tensor, 
                                            self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        
        reprojected_egoleft_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoleft_lane_tensor, 
                                            self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        
        reprojected_egoright_lane_gradient_loss =  \
            self.calc_reprojected_gradient_loss(self.gt_reproj_egoright_lane_tensor, 
                                            self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        reprojected_gradient_loss_driving_corridor = reprojected_ego_path_gradient_loss + \
            reprojected_egoleft_lane_gradient_loss + reprojected_egoright_lane_gradient_loss
        
        return reprojected_gradient_loss_driving_corridor
    
    # BEV Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_BEV_data_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        data_error_sum = 0

        for i in range(0, len(gt_tensor_x_vals)):
                
            error = torch.abs(gt_tensor_x_vals[i] - pred_tensor_x_vals[i])
            data_error_sum = data_error_sum + error

        bev_data_loss = data_error_sum/len(gt_tensor_x_vals)

        return bev_data_loss
    
    # BEV gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_BEV_graient_loss(self, gt_tensor, pred_tensor):

        gt_tensor_x_vals = gt_tensor[0,:]
        pred_tensor_x_vals = pred_tensor[0]

        bev_gradient_error = 0

        for i in range(0, len(gt_tensor_x_vals) - 1):

            gt_gradient = gt_tensor_x_vals[i+1] - gt_tensor_x_vals[i]
            pred_gradient = pred_tensor_x_vals[i+1] - pred_tensor_x_vals[i]

            error = torch.abs(gt_gradient - pred_gradient)
            bev_gradient_error = bev_gradient_error + error

        bev_gradient_loss = bev_gradient_error/len(gt_tensor_x_vals)
        return bev_gradient_loss
    
    # Reprojected Data Loss for a single lane/path element
    # Mean absolute error on predictions
    def calc_reprojected_data_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]
        gt_reprojected_tensor_y_vals = gt_reprojected_tesnor[1,:]

        data_error_sum = 0

        for i in range(0, len(gt_tensor_x_vals)):

            gt_reprojected_x = gt_reprojected_tensor_x_vals[i]
            prediction_reprojected_x = prediction_reprojected[i][0]
            
            gt_reprojected_y = gt_reprojected_tensor_y_vals[i]
            prediction_reprojected_y = prediction_reprojected[i][1]
            
            x_error = torch.abs(gt_reprojected_x - prediction_reprojected_x)
            y_error = torch.abs(gt_reprojected_y - prediction_reprojected_y)
            L1_error = x_error + y_error

            data_error_sum = data_error_sum + L1_error

        reprojected_data_loss = data_error_sum/len(gt_tensor_x_vals)
        return reprojected_data_loss
    
    # Reprojected points gradient loss for a single lane/path element
    # Sum of finite difference gradients
    def calc_reprojected_gradient_loss(self, gt_reprojected_tesnor, gt_tensor, pred_tensor):

        prediction_reprojected, _ = \
            self.getPerspectivePointsFromBEV(gt_tensor, pred_tensor)
        
        gt_tensor_x_vals = gt_tensor[0,:]
        gt_reprojected_tensor_x_vals = gt_reprojected_tesnor[0,:]

        reprojected_gradient_error = 0

        for i in range(0, len(gt_tensor_x_vals)-1):

            gt_reprojected_gradient = gt_reprojected_tensor_x_vals[i+1] \
                - gt_reprojected_tensor_x_vals[i]
            
            prediction_reprojected_gradient = prediction_reprojected[i+1][0] \
                - prediction_reprojected[i][0]
            
            error = torch.abs(gt_reprojected_gradient - prediction_reprojected_gradient)
            reprojected_gradient_error = reprojected_gradient_error + error

        reprojected_gradient_loss = reprojected_gradient_error/len(gt_tensor_x_vals)
            
        return reprojected_gradient_loss

    # Get the list of reprojected points from X,Y BEV coordinates
    def getPerspectivePointsFromBEV(self, gt_tensor, pred_tensor):
        gt_tensor_y_vals = gt_tensor[1,:]
        pred_tensor_x_vals = pred_tensor[0]

        perspective_image_points, perspective_image_points_normalized = \
            self.projectBEVtoImage(pred_tensor_x_vals, gt_tensor_y_vals)

        return perspective_image_points_normalized, perspective_image_points

    # Reproject BEV points to perspective image
    def projectBEVtoImage(self, bev_x_points, bev_y_points):

        perspective_image_points = []
        perspective_image_points_normalized = []

        for i in range(0, len(bev_x_points)):
            
            image_homogenous_point_x = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[0][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[0][1] + self.homotrans_mat_tensor[0][2]
            
            image_homogenous_point_y = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[1][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[1][1] + self.homotrans_mat_tensor[1][2]
            
            image_homogenous_point_scale_factor = self.BEV_W*bev_x_points[i]*self.homotrans_mat_tensor[2][0] + \
                self.BEV_H*bev_y_points[i]*self.homotrans_mat_tensor[2][1] + self.homotrans_mat_tensor[2][2]
            
            image_point = [(image_homogenous_point_x/image_homogenous_point_scale_factor), \
                (image_homogenous_point_y/image_homogenous_point_scale_factor)]
            
            image_point_normalized = [image_point[0]/self.perspective_W, image_point[1]/self.perspective_H]

            perspective_image_points.append(image_point)
            perspective_image_points_normalized.append(image_point_normalized)

        return perspective_image_points, perspective_image_points_normalized

    # Loss backward pass
    def loss_backward(self):
        self.total_loss.backward()

    # Get total loss value
    def get_total_loss(self):
        return self.total_loss.item()
    
    # Get total loss value
    def get_total_loss(self):
        return self.total_loss.detach().cpu().numpy()
    
    def get_bev_loss(self):
        return self.BEV_loss.item()
    
    def get_reprojected_loss(self):
        return self.reprojected_loss.item()

    # Logging losses - Total, BEV, Reprojected
    def log_loss(self, log_count):
        self.writer.add_scalars(
            "Training Loss", {
                "Total_loss" : self.get_total_loss(),
                "BEV_loss" : self.get_bev_loss(),
                "Reprojected_loss" : self.get_reprojected_loss()
            },
            (log_count)
        )

    # Run optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save model
    def save_model(self, model_save_path):
        torch.save(
            self.model.state_dict(), 
            model_save_path
        )

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("Finished training")

    # Save predicted visualization
    def save_visualization(self, log_count, bev_vis, vis_path = "", is_train = False):

        # Predicted Egopath (BEV)
        pred_bev_ego_path = self.pred_bev_ego_path_tensor.cpu().detach().numpy()
        
        # Predicted Egopath (Reprojected)
        _, pred_reprojected_ego_path_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egopath_tensor, 
                                            self.pred_bev_ego_path_tensor)
        pred_reprojected_ego_path_x_vals = []
        pred_reprojected_ego_path_y_vals = []

        for i in range(0, len(pred_reprojected_ego_path_tensor)):
            pred_reprojected_ego_path_x_val = pred_reprojected_ego_path_tensor[i][0].cpu().detach().numpy()
            pred_reprojected_ego_path_y_val = pred_reprojected_ego_path_tensor[i][1].cpu().detach().numpy()

            pred_reprojected_ego_path_x_vals.append(pred_reprojected_ego_path_x_val)
            pred_reprojected_ego_path_y_vals.append(pred_reprojected_ego_path_y_val)

        # Predicted Egoleft Lane (BEV)
        prev_bev_egoleft_lane = self.pred_bev_egoleft_lane_tensor.cpu().detach().numpy()

        # Predicted Egoleft Lane (Reprojected)
        _, pred_reprojected_egoleft_lane_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egoleft_lane_tensor, 
                                            self.pred_bev_egoleft_lane_tensor)
        pred_reprojected_egoleft_lane_x_vals = []
        pred_reprojected_egoleft_lane_y_vals = []

        for i in range(0, len(pred_reprojected_egoleft_lane_tensor)):
            pred_reprojected_egoleft_lane_x_val = pred_reprojected_egoleft_lane_tensor[i][0].cpu().detach().numpy()
            pred_reprojected_egoleft_lane_y_val = pred_reprojected_egoleft_lane_tensor[i][1].cpu().detach().numpy()

            pred_reprojected_egoleft_lane_x_vals.append(pred_reprojected_egoleft_lane_x_val)
            pred_reprojected_egoleft_lane_y_vals.append(pred_reprojected_egoleft_lane_y_val)

        # Predicted Egoright Lane (BEV)
        pred_bev_egoright_lane = self.pred_bev_egoright_lane_tensor.cpu().detach().numpy()

        # Predicted Egoright Lane (Reprojected)
        _, pred_reprojected_egoright_lane_tensor = \
                self.getPerspectivePointsFromBEV(self.gt_bev_egoright_lane_tensor, 
                                            self.pred_bev_egoright_lane_tensor)

        pred_reprojected_egoright_lane_x_vals = []
        pred_reprojected_egoright_lane_y_vals = []

        for i in range(0, len(pred_reprojected_egoright_lane_tensor)):
            pred_reprojected_egoright_lane_x_val = pred_reprojected_egoright_lane_tensor[i][0].cpu().detach().numpy()
            pred_reprojected_egoright_lane_y_val = pred_reprojected_egoright_lane_tensor[i][1].cpu().detach().numpy()

            pred_reprojected_egoright_lane_x_vals.append(pred_reprojected_egoright_lane_x_val)
            pred_reprojected_egoright_lane_y_vals.append(pred_reprojected_egoright_lane_y_val)

        # BEV fixed y-values of anchors
        bev_y_vals = self.gt_bev_egopath_tensor[1,:].cpu().detach().numpy()*self.BEV_H
   
        # Visualize Ground Truth and Predictions (BEV)
        fig_bev, axs = plt.subplots(1,2, figsize=(8, 8))

        # Prediction
        axs[0].set_title('Prediction',fontweight ="bold") 
        axs[0].set_xlim(0, self.BEV_W - 1)
        axs[0].set_ylim(self.BEV_H - 1, 0)
        axs[0].imshow(self.bev_image)

        pred_bev_ego_path_vals = pred_bev_ego_path[0]*self.BEV_W
        prev_bev_egoleft_lane_vals = prev_bev_egoleft_lane[0]*self.BEV_W
        pred_bev_egoright_lane_vals = pred_bev_egoright_lane[0]*self.BEV_W

        axs[0].plot(pred_bev_ego_path_vals, bev_y_vals, 'yellow')
        axs[0].plot(prev_bev_egoleft_lane_vals, bev_y_vals, 'green')
        axs[0].plot(pred_bev_egoright_lane_vals, bev_y_vals, 'cyan')

        # Ground Truth
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[1].imshow(bev_vis)
        
        # Save figure to Tensorboard
        if(is_train):
            self.writer.add_figure("Train (BEV)", fig_bev, global_step = (log_count))
        else:
            fig_bev.savefig(vis_path + '_BEV.png')

        # Visualize Ground Truth and Predictions (Perspective)
        fig_perspective, axs = plt.subplots(2,1, figsize=(8, 8))

        # Prediction
        axs[0].set_title('Prediction',fontweight ="bold") 
        axs[0].set_xlim(0, self.perspective_W - 1)
        axs[0].set_ylim(self.perspective_H - 1, 0)
        axs[0].imshow(self.perspective_image)

        axs[0].plot(pred_reprojected_ego_path_x_vals, pred_reprojected_ego_path_y_vals, 'yellow')
        axs[0].plot(pred_reprojected_egoleft_lane_x_vals, pred_reprojected_egoleft_lane_y_vals, 'chartreuse')
        axs[0].plot(pred_reprojected_egoright_lane_x_vals, pred_reprojected_egoright_lane_y_vals, 'cyan')

        # Visualize Ground Truth - Perspective
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[1].imshow(self.perspective_image)
        axs[1].set_xlim(0, self.perspective_W - 1)
        axs[1].set_ylim(self.perspective_H - 1, 0)

        gt_reprojected_ego_path_x_vals = self.reproj_egopath[0,:]*self.perspective_W
        gt_reprojected_ego_path_y_vals = self.reproj_egopath[1,:]*self.perspective_H
        gt_reprojected_egoleft_lane_x_vals = self.reproj_egoleft[0,:]*self.perspective_W
        gt_reprojected_egoleft_lane_y_vals = self.reproj_egoleft[1,:]*self.perspective_H
        gt_reprojected_egoright_lane_x_vals = self.reproj_egoright[0,:]*self.perspective_W
        gt_reprojected_egoright_lane_y_vals = self.reproj_egoright[1,:]*self.perspective_H
        
        axs[1].plot(gt_reprojected_ego_path_x_vals, gt_reprojected_ego_path_y_vals, 'yellow')
        axs[1].plot(gt_reprojected_egoleft_lane_x_vals, gt_reprojected_egoleft_lane_y_vals, 'chartreuse')
        axs[1].plot(gt_reprojected_egoright_lane_x_vals, gt_reprojected_egoright_lane_y_vals,'cyan')

        # Save figure to Tensorboard
        if(is_train):
            self.writer.add_figure("Train (Perspective)", fig_perspective, global_step = (log_count))
        else:
            fig_perspective.savefig(vis_path + '_Perspective.png')

        plt.close(fig_bev)
        plt.close(fig_perspective)
    
    # Log validation loss for each dataset to TensorBoard
    def log_validation_dataset(self, dataset, validation_loss_dataset_total, log_count):
         self.writer.add_scalar(f"{dataset} (Validation)", validation_loss_dataset_total, log_count)

    # Log overall validation loss across all datasets to TensorBoard
    def log_validation_overall(self, overall_val_score, log_count):
        self.writer.add_scalar("Overall (Validation)", overall_val_score, log_count)
