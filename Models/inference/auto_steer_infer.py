# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import sys
import math
import torch
from PIL import Image
from torchvision import transforms

import numpy as np

sys.path.append('..')
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer_network import AutoSteerNetwork


class AutoSpeedNetworkInfer():
    def __init__(self, egolanes_checkpoint_path='', autosteer_checkpoint_path=''):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                # transforms.CenterCrop((1440, 2880)),  # e.g. (224, 224),
                # transforms.Resize((320, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model, load to device and set to evaluation mode
        if (len(egolanes_checkpoint_path) > 0 and len(autosteer_checkpoint_path) > 0):
            # Loading model with full pre-trained weights
            self.egoLanesNetwork = EgoLanesNetwork()
            self.egoLanesNetwork.load_state_dict(torch.load \
                                                     (egolanes_checkpoint_path, weights_only=True,
                                                      map_location=self.device))

            self.model = AutoSteerNetwork()

            # If the model is also pre-trained then load the pre-trained downstream weights
            self.model.load_state_dict(torch.load \
                                           (autosteer_checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')

        self.egoLanesNetwork = self.egoLanesNetwork.to(self.device)
        self.egoLanesNetwork = self.egoLanesNetwork.eval()
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        # self.feature = torch.zeros_like(torch.randn(1, 64, 10, 20)).to(self.device)
        self.feature = torch.zeros_like(torch.randn(1, 64, 10, 20)).to(self.device)

        self.image_T_minus_1 = Image.new("RGB", (640, 320), color=(0, 0, 0))

    def inference(self, image):

        width, height = image.size
        if (width != 640 or height != 320):
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        # self.image_T_mius_1.show()
        image_tensor_T_minus_1 = self.image_loader(self.image_T_minus_1)
        image_tensor_T_minus_1 = image_tensor_T_minus_1.unsqueeze(0)
        image_tensor_T_minus_1 = image_tensor_T_minus_1.to(self.device)

        image_tensor_T = self.image_loader(image)
        image_tensor_T = image_tensor_T.unsqueeze(0)
        image_tensor_T = image_tensor_T.to(self.device)

        # Run model
        with torch.no_grad():
            l1 = self.egoLanesNetwork(image_tensor_T_minus_1)
            l2 = self.egoLanesNetwork(image_tensor_T)
            lane_features_concat = torch.cat((l1, l2), dim=1)
            _, prediction = self.model(lane_features_concat)
            prediction = prediction.squeeze(0).cpu().detach()

        # prediction = self.model(image_tensor_T)

        # Get output, find max class probability and convert to steering angle
        # probs = torch.nn.functional.softmax(prediction, dim=0)
        output = torch.argmax(prediction).item() - 30

        self.image_T_minus_1 = image.copy()

        return output
