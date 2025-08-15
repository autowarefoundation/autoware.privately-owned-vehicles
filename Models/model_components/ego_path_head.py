#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoPathHead(nn.Module):
    def __init__(self):
        super(EgoPathHead, self).__init__()

        # Context - MLP Layers
        self.ego_path_layer_0 = nn.Linear(800, 11)
 

    def forward(self, feature_vector):

        # Prediction
        ego_path = self.ego_path_layer_0(feature_vector)

        return ego_path