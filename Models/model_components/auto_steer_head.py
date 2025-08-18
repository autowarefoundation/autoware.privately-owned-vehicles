#! /usr/bin/env python3
import torch.nn as nn

class AutoSteerHead(nn.Module):
    def __init__(self):
        super(AutoSteerHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.25)
        self.dropout_aggressize = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()
   
        # Ego Path  Decode layers
        self.ego_path_layer_0 = nn.Linear(1456, 1280)
        self.ego_path_layer_1 = nn.Linear(1280, 1024)
        self.ego_path_layer_2 = nn.Linear(1024, 800)
        self.ego_path_layer_3 = nn.Linear(800, 11)
        self.ego_left_offset_layer = nn.Linear(1456, 1)
        self.ego_right_offset_layer = nn.Linear(1456, 1)
 

    def forward(self, feature_vector):

        # Features
        feature_vector = self.dropout(feature_vector)

        # Ego Path Lane MLP
        ego_path = self.ego_path_layer_0(feature_vector)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        ego_path = self.ego_path_layer_1(ego_path)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        ego_path = self.ego_path_layer_2(ego_path)
        ego_path = self.dropout_aggressize(ego_path)
        ego_path = self.GeLU(ego_path)

        # Ego Left Lane Offset Prediction
        ego_left_lane_offset = self.ego_left_offset_layer(feature_vector)
        ego_left_lane_offset = self.sigmoid(ego_left_lane_offset)*0.5

        # Ego Right Lane Offset Prediction
        ego_right_lane_offset = self.ego_right_offset_layer(feature_vector)
        ego_right_lane_offset = self.sigmoid(ego_right_lane_offset)*0.5

        # Final Path Prediction
        ego_path = self.ego_path_layer_3(ego_path) + 0.5
        ego_left_lane = ego_path - ego_left_lane_offset
        ego_right_lane = ego_path + ego_right_lane_offset

        return ego_path, ego_left_lane, ego_right_lane