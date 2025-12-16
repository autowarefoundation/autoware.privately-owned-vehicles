# AutoSteer trainer
# Gets the data from the main train loop
# Applies data augmentations (noise and random horizontal flip)
# Converts data to a tensor and loads it into GPU memory
# Runs the network
# Calculates the loss
# Creates visualizations in tensorboard

import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image
import cv2
import sys
import random

sys.path.append('..')
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer_network import AutoSteerNetwork
from Models.data_utils.augmentations import Augmentations

np.Inf = np.inf


class AutoSteerTrainer():
    def __init__(self, checkpoint_path='', pretrained_checkpoint_path='', is_pretrained=False):

        # Image and ground truth as Numpy arrays and Pytorch tensors
        self.img_T_minus_1 = 0
        self.gt_T_minus_1 = 0
        self.gt_tensor_minus_1 = 0
        self.img_T = 0
        self.gt_T = 0
        self.gt_tensor = 0

        # Loss and prediction
        self.loss = 0
        self.prediction_prev = 0
        self.prediction = 0
        self.prediction_val = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if (is_pretrained):

            # Instantiate Model for validation or inference - load both pre-traiend EgoLanes and SuperDepth weights
            if (len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                self.egoLanesNetwork = EgoLanesNetwork()
                self.egoLanesNetwork.load_state_dict(torch.load \
                                                         (pretrained_checkpoint_path, weights_only=True,
                                                          map_location=self.device))
                self.model = AutoSteerNetwork()

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                                               (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of AutoSteer and upstream EgoLanes weights as well')
            else:
                raise ValueError('Please ensure AutoSteer network weights are provided for downstream elements')

        else:

            # Instantiate Model for training - load pre-traiend EgoLanes weights only
            if (len(pretrained_checkpoint_path) > 0):

                # Loading EgoLanes pre-trained for upstream weights
                self.egoLanesNetwork = EgoLanesNetwork()
                self.egoLanesNetwork.load_state_dict(torch.load \
                                                         (pretrained_checkpoint_path, weights_only=True,
                                                          map_location=self.device))

                # Loading model with pre-trained upstream weights
                self.model = AutoSteerNetwork()
                print(
                    'Loading pre-trained model weights of upstream EgoLanes only, AutoSteer initialised with random weights')
            else:
                raise ValueError('Please ensure EgoLanes network weights are provided for upstream elements')

        # Model to device
        self.egoLanesNetwork = self.egoLanesNetwork.to(self.device)
        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0005
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                # transforms.CenterCrop((1440, 2880)),  # e.g. (224, 224),
                # transforms.Resize((320, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.gt_loader = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))
            ]
        )

    # Assign input variables
    def set_data(self, img_T_minus_1, gt_T_minus_1, img_T, gt_T):
        self.img_T_minus_1 = img_T_minus_1
        self.gt_T_minus_1 = gt_T_minus_1
        self.img_T = img_T
        self.gt_T = gt_T

    # Set learning rate
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Image agumentations flip images horizontally
    def apply_augmentations(self):
        self.img_T_minus_1 = self.img_T_minus_1[:, ::-1, :].copy()
        self.gt_T_minus_1 = -self.gt_T_minus_1

        self.img_T = self.img_T[:, ::-1, :].copy()
        self.gt_T = -self.gt_T

    # Load Data
    def load_data(self):
        self.load_image_tensor()
        self.load_gt_tensor()

    # Load Image as Tensor
    def load_image_tensor(self):
        # T-1
        image_tensor_T_minus_1 = self.image_loader(self.img_T_minus_1)
        image_tensor_T_minus_1 = image_tensor_T_minus_1.unsqueeze(0)
        self.image_tensor_T_minus_1 = image_tensor_T_minus_1.to(self.device)

        # T
        image_tensor_T = self.image_loader(self.img_T)
        image_tensor_T = image_tensor_T.unsqueeze(0)
        self.image_tensor_T = image_tensor_T.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self):
        gt_tensor = self.gt_loader(self.gt_T)
        gt_tensor = gt_tensor.unsqueeze(0)
        self.gt_tensor = gt_tensor.to(self.device)

        gt_tensor_minus_1 = self.gt_loader(self.gt_T_minus_1)
        gt_tensor_minus_1 = gt_tensor_minus_1.unsqueeze(0)
        self.gt_tensor_minus_1 = gt_tensor_minus_1.to(self.device)

    def angle_to_tensor(self, angle, num_classes=61, angle_min=-30, angle_max=30):
        angle = max(angle_min, min(angle_max, angle))
        class_idx = int(round(angle - angle_min))  # 1 deg resolution
        gt_tensor = torch.tensor([class_idx], dtype=torch.long)  # shape [1]
        gt_tensor = gt_tensor.to(self.device)
        return gt_tensor

    def angle_to_class(self, angle, angle_min=-30, angle_max=30):
        angle = max(angle_min, min(angle_max, angle))
        class_idx = int(round(angle - angle_min))  # maps -30→0, +30→60
        return class_idx

    # Run Model
    def run_model(self):
        sigma = 1.0
        num_classes = 61

        with torch.no_grad():
            l1 = self.egoLanesNetwork(self.image_tensor_T_minus_1)
            l2 = self.egoLanesNetwork(self.image_tensor_T)
        lane_features_concat = torch.cat((l1, l2), dim=1)
        self.prediction_prev, self.prediction = self.model(lane_features_concat)
        # self.prediction = torch.argmax(self.prediction).item() - 30

        # # Cross Entropy Loss
        # CEL = nn.CrossEntropyLoss()
        # loss_prev = CEL(self.prediction_prev.unsqueeze(0), self.angle_to_tensor(self.gt_T_minus_1))
        # loss = CEL(self.prediction.unsqueeze(0), self.angle_to_tensor(self.gt_T))
        #
        # self.loss = loss_prev + loss

        # # L1 loss
        # prediction_class = torch.argmax(self.prediction, dim=1).item()
        # pred_tensor = torch.tensor([prediction_class - 30], dtype=torch.float32)
        # gt_tensor = torch.tensor([self.gt_T], dtype=torch.float32)
        #
        # l1 = torch.nn.L1Loss()
        # self.l1_loss = l1(pred_tensor, gt_tensor)

        # KL divergence
        # Create Gaussian target distribution
        log_softmax = nn.LogSoftmax(dim=1)
        classes = torch.arange(num_classes, dtype=torch.float32, device=self.device)

        # # T - 1
        prediction_prev = log_softmax(self.prediction_prev.unsqueeze(0))
        gt_prev_class = self.angle_to_class(self.gt_T_minus_1)
        prev_target = torch.exp(-(classes - gt_prev_class) ** 2 / (2 * sigma ** 2))
        prev_target /= prev_target.sum()  # normalize to sum=1

        # T
        prediction = log_softmax(self.prediction.unsqueeze(0))
        gt_class = self.angle_to_class(self.gt_T)
        target = torch.exp(-(classes - gt_class) ** 2 / (2 * sigma ** 2))
        target /= target.sum()  # normalize to sum=1

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        loss_T_minus_1 = kl_loss(prediction_prev, prev_target.unsqueeze(0))  # add batch dim
        loss_T = kl_loss(prediction, target.unsqueeze(0))  # add batch dim

        self.loss = loss_T_minus_1 + loss_T

    # Run Validation and calculate metrics
    def validate(self, img_T_minus_1_val, gt_T_minus_1_val, img_T_val, gt_T_val):

        # Set Data
        self.set_data(img_T_minus_1_val, gt_T_minus_1_val, img_T_val, gt_T_val)

        # Augmenting Image
        # self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data()

        # Calculate IoU score
        score = self.calc_score()

        return score

    # Run network on test image and visualize result
    def test(self, test_images, test_images_save_path, log_count):

        test_images_list = sorted([f for f in pathlib.Path(test_images).glob("*")])

        for i in range(0, len(test_images_list)):
            # Read test images
            frame = cv2.imread(str(test_images_list[i]), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame)

            # Resize to correct dimensions for network
            image_pil = image_pil.resize((640, 320))

            # Load test images and run inference
            test_image_tensor = self.image_loader(image_pil)
            test_image_tensor = test_image_tensor.unsqueeze(0)
            test_image_tensor = test_image_tensor.to(self.device)
            test_output = self.model(test_image_tensor)

            # Process the output and scale to match the input image size
            test_output = test_output.squeeze(0).cpu().detach()
            test_output = test_output.permute(1, 2, 0)
            test_output = test_output.numpy()

            # Resize to match original image dimension
            test_output = cv2.resize(test_output, (frame.shape[1], frame.shape[0]))

            # Create visualization
            alpha = 0.5
            test_visualization = cv2.addWeighted(self.make_visualization(test_output), \
                                                 alpha, frame, 1 - alpha, 0)
            test_visualization = cv2.cvtColor(test_visualization, cv2.COLOR_BGR2RGB)

            # Save visualization
            image_save_path = test_images_save_path + str(log_count) + '_' + str(i) + '.jpg'
            cv2.imwrite(image_save_path, test_visualization)

    # Loss Backward Pass
    def loss_backward(self):
        self.loss.backward()

    # Get loss value
    def get_loss(self):
        return self.loss.item()

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

    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)

    # Logging Training Loss
    def log_loss(self, log_count):
        self.writer.add_scalar("Loss/train", self.get_loss(), (log_count))

    # Logging Validation mIoU Score
    def log_val(self, val_cel_loss, val_accuracy, log_count):
        print('Logging Validation')
        self.writer.add_scalar("Val CEL loss", val_cel_loss, (log_count))
        self.writer.add_scalar("Val accuracy", val_accuracy, (log_count))

    # Calculate IoU score for validation
    def calc_score(self):
        with torch.no_grad():
            prediction_class = torch.argmax(self.prediction).item()
            gt_class = self.angle_to_class(self.gt_T)
            accuracy = int(abs(prediction_class - gt_class) <= 1)

        return self.loss.item(), accuracy

    # Save predicted visualization
    def save_visualization(self, log_count):

        # Get prediction
        prediction = self.prediction.squeeze(0).cpu().detach()

        # Get ground truth
        gt_vis = self.gt_T

        # Prediction visualization
        prediction_vis = torch.argmax(prediction).item() - 30

        # Visualize
        fig_img = plt.figure(figsize=(8, 4))

        plt.axis('off')
        plt.imshow(self.img_T)

        txt = (
            f"Pred: {prediction_vis:.3f}\n"
            f"GT: {gt_vis:.3f}"
        )
        plt.text(
            20, 40,  # position on the image
            txt,
            color="black",
            fontsize=8,
            bbox=dict(
                facecolor="white",  # light background
                alpha=0.7,
                edgecolor="black",
                boxstyle="round,pad=0.5"
            )
        )

        # plt.show()

        # Write the figure
        self.writer.add_figure('Image', fig_img, global_step=(log_count))

    # Visualize predicted result
    def make_visualization(self, result):

        # Getting size of prediction
        shape = result.shape
        row = shape[0]
        col = shape[1]

        # Creating visualization image
        vis_predict_object = np.zeros((row, col, 3), dtype="uint8")

        # Assigning background colour
        vis_predict_object[:, :, 0] = 61
        vis_predict_object[:, :, 1] = 93
        vis_predict_object[:, :, 2] = 255

        # Getting foreground object labels
        foreground_lables = np.where(result > 0)

        # Assigning foreground objects colour
        vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 255
        vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 234
        vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 0

        return vis_predict_object

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
