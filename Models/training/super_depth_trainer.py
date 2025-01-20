
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.super_depth_network import SuperDepthNetwork
from data_utils.augmentations import Augmentations

class SuperDepthTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', size=''):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.augmented = 0
        self.augmented_val = 0
        self.image_tensor = 0
        self.image_val_tensor = 0
        self.gt_tensor = 0
        self.gt_val_tensor = 0
        self.loss = 0
        self.prediction = 0
        self.calc_loss = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
        
        # Instantiate Model with pre-trained weights
        sceneSegNetwork = SceneSegNetwork()
        if(len(pretrained_checkpoint_path) > 0):
            sceneSegNetwork.load_state_dict(torch.load \
                (pretrained_checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No pre-trained model checkpoint path - pass in pretrained SceneSeg checkpoint path')
        
        self.model = SuperDepthNetwork(sceneSegNetwork)

        # If we are loading pre-trained weights for the SuperDepth network as well
        if(len(checkpoint_path) > 0):
            self.model.load_state_dict(torch.load \
                (self.checkpoint_path, weights_only=True))
        
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

    # Logging Training Loss
    def log_loss(self, log_count):
        print('Logging Training Loss', log_count, self.get_loss())
        self.writer.add_scalar("Loss/train", self.get_loss(), (log_count))

    # Logging Validation mAE
    def log_val_mAE(self, mAE, log_count):
        print('Logging Validation')      
        self.writer.add_scalar("Val/mAE", mAE, (log_count))
        

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    def set_val_data(self, image_val, gt_val):
        self.image_val = image_val
        self.gt_val = gt_val
    
    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Image
            aug_train = Augmentations(self.image, self.gt, True, data_type='DEPTH')
            self.image, self.augmented = aug_train.getAugmentedData()
        else:
            # Augmenting Image
            aug_val = Augmentations(self.image_val, self.gt_val, False, data_type='DEPTH')
            self.image_val, self.augmented_val = aug_val.getAugmentedData()
    
    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)


    # Run Model
    def run_model(self):     
        self.loss = nn.L1Loss()
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    # Loss Backward Pass
    def loss_backward(self): 
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Normalize ground truth value for visualization
    def shift_height(self, height):
        height = height + np.min(height)
        return height
    
    # Save predicted visualization
    def save_visualization(self, log_count):

        print('Saving Visualization')

        # Converting prediction output to visualization
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)
        prediction_vis = prediction_vis.numpy()
        prediction_vis = self.shift_height(prediction_vis)

        # Normalizing ground truth height to same range as predicition
        augmented_vis = self.shift_height(self.augmented)/7

        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(augmented_vis)
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[2].imshow(prediction_vis)
        axs[2].set_title('Prediction',fontweight ="bold") 
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))
    
    # Load Image as Tensor
    def load_image_tensor(self, is_train):

        if(is_train):
            image_tensor = self.image_loader(self.image)
            image_tensor = image_tensor.unsqueeze(0)
            self.image_tensor = image_tensor.to(self.device)
        else:
            image_val_tensor = self.image_loader(self.image_val)
            image_val_tensor = image_val_tensor.unsqueeze(0)
            self.image_val_tensor = image_val_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self, is_train):

        if(is_train):
            gt_tensor = torch.from_numpy(self.augmented)
            gt_tensor = gt_tensor.permute(2, 0, 1)
            gt_tensor = torch.div(gt_tensor, 7)
            gt_tensor = gt_tensor.unsqueeze(0)
            self.gt_tensor = gt_tensor.to(self.device)
        else:
            gt_val_tensor = torch.from_numpy(self.augmented_val)
            gt_val_tensor = gt_val_tensor.permute(2, 0, 1)
            gt_val_tensor = torch.div(gt_val_tensor, 7)
            gt_val_tensor = gt_val_tensor.unsqueeze(0)
            self.gt_val_tensor = gt_val_tensor.to(self.device)
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)
    
    # Run Validation and calculate metrics
    def validate(self, image_val, gt_val):

        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)
        
        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Running model
        output_val = self.model(self.image_val_tensor)

        # Conversions
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()

        ground_truth_val = self.gt_val_tensor.squeeze(0).cpu().detach()
        ground_truth_val = ground_truth_val.permute(1, 2, 0)
        ground_truth_val = ground_truth_val.numpy()

        # Calculating mean absolute normalized error
        accuracy = np.average(np.abs(ground_truth_val - output_val))
        return accuracy

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')