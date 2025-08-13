
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
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.ego_space_network import EgoSpaceNetwork
from data_utils.augmentations import Augmentations

class EgoSpaceTrainer():
    def __init__(self,  checkpoint_path = '', pretrained_checkpoint_path = '', is_pretrained = False):

        # Image and ground truth as Numpy arrays and Pytorch tensors
        self.image = 0
        self.gt = 0
        self.image_tensor = 0
        self.gt_tensor = 0

        # Loss and prediction
        self.loss = 0
        self.prediction = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        if(is_pretrained):

            # Instantiate Model for validation or inference - load both pre-traiend SceneSeg and SuperDepth weights
            if(len(checkpoint_path) > 0):

                # Loading model with full pre-trained weights
                sceneSegNetwork = SceneSegNetwork()
                self.model = EgoSpaceNetwork(sceneSegNetwork)

                # If the model is also pre-trained then load the pre-trained downstream weights
                self.model.load_state_dict(torch.load \
                    (checkpoint_path, weights_only=True, map_location=self.device))
                print('Loading pre-trained model weights of EgoSpace and upstream SceneSeg weights as well')
            else:
                raise ValueError('Please ensure EgoSpace network weights are provided for downstream elements')

        else:

            # Instantiate Model for training - load pre-traiend SceneSeg weights only
            if(len(pretrained_checkpoint_path) > 0):

                # Loading SceneSeg pre-trained for upstream weights
                sceneSegNetwork = SceneSegNetwork()
                sceneSegNetwork.load_state_dict(torch.load \
                    (pretrained_checkpoint_path, weights_only=True, map_location=self.device))

                # Loading model with pre-trained upstream weights
                self.model = EgoSpaceNetwork(sceneSegNetwork)
                print('Loading pre-trained model weights of upstream SceneSeg only, EgoSpace initialised with random weights')
            else:
                raise ValueError('Please ensure SceneSeg network weights are provided for upstream elements')

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

        self.gt_loader = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )

    # Assign input variables
    def set_data(self, image, gt):
        self.image = image
        self.gt = gt

    # Set learning rate
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Data for training
            augTrain = Augmentations(is_train=True, data_type='BINARY_SEGMENTATION')
            augTrain.setData(self.image, self.gt)
            self.image, self.gt  = \
                augTrain.applyTransformBinarySeg(image=self.image, ground_truth=self.gt)
            #self.image = augTrain.applyNoiseRoadWork()
        else:
            # Augmenting Data for testing/validation
            augVal = Augmentations(is_train=False, data_type='BINARY_SEGMENTATION')
            augVal.setData(self.image, self.gt)
            self.image, self.gt = \
                augVal.applyTransformBinarySeg(image=self.image, ground_truth=self.gt)

    # Load Data
    def load_data(self):
        self.load_image_tensor()
        self.load_gt_tensor()

    # Load Image as Tensor
    def load_image_tensor(self):
        image_tensor = self.image_loader(self.image)
        image_tensor = image_tensor.unsqueeze(0)
        self.image_tensor = image_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self):
        gt_tensor = self.gt_loader(self.gt)
        gt_tensor = gt_tensor.unsqueeze(0)
        self.gt_tensor = gt_tensor.to(self.device)

    # Run Model
    def run_model(self):
        self.prediction = self.model(self.image_tensor)
        BCELoss = nn.BCEWithLogitsLoss()
        self.loss = BCELoss(self.prediction, self.gt_tensor)

    # Run Validation and calculate metrics
    def validate(self, image, gt):

        # Set Data
        self.set_data(image, gt)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data()

        # Calculate IoU score
        iou_score = self.calc_IoU_val()

        return iou_score

    # Run network on test image and visualize result
    def test(self, test_images, test_images_save_path, log_count):

        # test_images_list = sorted([f for f in pathlib.Path(test_images).glob("*")])
        test_images_list = sorted(pathlib.Path(test_images).glob("*.png"), key=lambda f: int(f.stem))

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
            image_save_path = test_images_save_path + str(i) + '_' + str(log_count) + '.jpg'
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
        self.writer.add_scalar("Loss/train",self.get_loss(), (log_count))

    # Logging Validation mIoU Score
    def log_IoU(self, mIoU, log_count):
        print('Logging Validation')
        self.writer.add_scalar("IoU/val", mIoU, (log_count))

    # Calculate IoU score for validation
    def calc_IoU_val(self):
        output_val = self.model(self.image_tensor)
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()
        output_val[output_val <= 0] = 0.0
        output_val[output_val > 0] = 1.0
        iou_score = self.IoU(output_val, self.gt)

        return iou_score

    # IoU calculation
    def IoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        iou_score = (np.sum(intersection) + 1) / float(np.sum(union) + 1)
        return iou_score

    # Save predicted visualization
    def save_visualization(self, log_count):

        # Get prediction
        prediction_vis = self.prediction.squeeze(0).cpu().detach()
        prediction_vis = prediction_vis.permute(1, 2, 0)

        # Get ground truth
        gt_vis = self.gt_tensor.squeeze(0).cpu().detach()
        gt_vis = gt_vis.permute(1,2,0)

        # Create visualization
        # Blending factor
        alpha = 0.5

        # Predicttion visualization
        prediction_vis = cv2.addWeighted(self.make_visualization(prediction_vis), \
            alpha, self.image, 1 - alpha, 0)

        # Ground truth visualization
        gt_vis = cv2.addWeighted(self.make_visualization(gt_vis), \
            alpha, self.image, 1 - alpha, 0)

        # Visualize the Image
        fig_img = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(self.image)
        # Write the figure
        self.writer.add_figure('Image/train', \
            fig_img, global_step=(log_count))

        # Visualize the Prediction
        fig_pred = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(prediction_vis)
        # Write the figure
        self.writer.add_figure('Prediction/train', \
            fig_pred, global_step=(log_count))

        # Visualize the Ground Truth
        fig_gt = plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(gt_vis)
        # Write the figure
        self.writer.add_figure('Ground-Truth/train', \
            fig_gt, global_step=(log_count))

    # Visualize predicted result
    def make_visualization(self, result):

        # Getting size of prediction
        shape = result.shape
        row = shape[0]
        col = shape[1]

        # Creating visualization image
        vis_predict_object = np.zeros((row, col, 3), dtype = "uint8")

        # Assigning background colour
        vis_predict_object[:,:,0] = 61
        vis_predict_object[:,:,1] = 93
        vis_predict_object[:,:,2] = 255

        # Getting foreground object labels
        foreground_lables = np.where(result > 0)

        # Assigning foreground objects colour
        vis_predict_object[foreground_lables[0], foreground_lables[1], 0] = 88
        vis_predict_object[foreground_lables[0], foreground_lables[1], 1] = 243
        vis_predict_object[foreground_lables[0], foreground_lables[1], 2] = 162

        return vis_predict_object

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')
