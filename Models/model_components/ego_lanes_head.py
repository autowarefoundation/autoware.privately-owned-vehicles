#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoLanesHead(nn.Module):
    def __init__(self):
        super(EgoLanesHead, self).__init__()

        # Context - MLP Layers
        self.ego_left_lane_layer_0 = nn.Linear(800, 11)
        self.ego_right_lane_layer_0 = nn.Linear(800, 11)

 

    def forward(self, feature_vector):

        # Prediction
        ego_left_lane = self.ego_left_lane_layer_0(feature_vector)
        ego_right_lane = self.ego_right_lane_layer_0(feature_vector)

        # Final result
        return ego_left_lane, ego_right_lane