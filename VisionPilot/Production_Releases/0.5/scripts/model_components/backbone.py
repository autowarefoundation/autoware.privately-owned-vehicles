#! /usr/bin/env python3
import torch.nn as nn
from torchvision import models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Encoder Backbone
        self.encoder = models.efficientnet_b0(weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1').features

    def forward(self, image):
        # Sequential layers of efficient net encoder
        l0 = self.encoder[0](image)
        l1 = self.encoder[1](l0)
        l2 = self.encoder[2](l1)
        l3 = self.encoder[3](l2)
        l4 = self.encoder[4](l3)
        l5 = self.encoder[5](l4)
        l6 = self.encoder[6](l5)
        l7 = self.encoder[7](l6)
        l8 = self.encoder[8](l7)
        return [l0, l2, l3, l4, l8]      