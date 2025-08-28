#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoLanesHead(nn.Module):
    def __init__(self):
        super(EgoLanesHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.25)
        self.dropout_aggressize = nn.Dropout(p=0.4)

        # Ego Left Decode layers
        self.ego_left_lane_layer_0 = nn.Linear(1456, 1280)
        self.ego_left_lane_layer_1 = nn.Linear(1280, 1024)
        self.ego_left_lane_layer_2 = nn.Linear(1024, 800)
        self.ego_left_lane_layer_3 = nn.Linear(800, 11)

        # Ego Right Decode layers
        self.ego_right_lane_layer_0 = nn.Linear(1456, 1280)
        self.ego_right_lane_layer_1 = nn.Linear(1280, 1024)
        self.ego_right_lane_layer_2 = nn.Linear(1024, 800)
        self.ego_right_lane_layer_3 = nn.Linear(800, 11)

 

    def forward(self, feature_vector):

        # Features
        feature_vector = self.dropout(feature_vector)

        # Ego Left Lane MLP
        ego_left_lane = self.ego_left_lane_layer_0(feature_vector)
        ego_left_lane = self.dropout_aggressize(ego_left_lane)
        ego_left_lane = self.GeLU(ego_left_lane)

        ego_left_lane = self.ego_left_lane_layer_1(ego_left_lane)
        ego_left_lane = self.dropout_aggressize(ego_left_lane)
        ego_left_lane = self.GeLU(ego_left_lane)

        ego_left_lane = self.ego_left_lane_layer_2(ego_left_lane)
        ego_left_lane = self.dropout_aggressize(ego_left_lane)
        ego_left_lane = self.GeLU(ego_left_lane)        
        
        # Ego Right Lane MLP
        ego_right_lane = self.ego_right_lane_layer_0(feature_vector)
        ego_right_lane = self.dropout_aggressize(ego_right_lane)
        ego_right_lane = self.GeLU(ego_right_lane)

        ego_right_lane = self.ego_right_lane_layer_1(ego_right_lane)
        ego_right_lane = self.dropout_aggressize(ego_right_lane)
        ego_right_lane = self.GeLU(ego_right_lane)

        ego_right_lane = self.ego_right_lane_layer_2(ego_right_lane)
        ego_right_lane = self.dropout_aggressize(ego_right_lane)
        ego_right_lane = self.GeLU(ego_right_lane)

        # Final Lane Predictions
        ego_left_lane = self.ego_left_lane_layer_3(ego_left_lane) + 0.4
        ego_right_lane = self.ego_right_lane_layer_3(ego_right_lane) + 0.6

        # Final result
        return ego_left_lane, ego_right_lane