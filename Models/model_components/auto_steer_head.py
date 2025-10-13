#! /usr/bin/env python3
import torch
import torch.nn as nn

class AutoSteerHead(nn.Module):
    def __init__(self):
        super(AutoSteerHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout_aggressize = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # Extraction Layers
        self.path_layer_0 = nn.Conv2d(1456, 512, 3, 1, 1)
        self.path_layer_1 = nn.Conv2d(512, 128, 3, 1, 1)
        self.path_layer_2 = nn.Conv2d(128, 5, 3, 1, 1)

        # Driving Corridor  Decode layers
        self.driving_corridor_layer_0 = nn.Linear(1000, 1000)
        self.driving_corridor_layer_1 = nn.Linear(1000, 1000)

        self.ego_left_x_offset = nn.Linear(1000, 1)
        self.ego_right_x_offset = nn.Linear(1000, 1)
        self.ego_path_x_offset = nn.Linear(1000, 1)
        self.angle_start = nn.Linear(1000, 1)
        self.angle_end = nn.Linear(1000, 1)
 

    def forward(self, features):

        # Calculating feature vector
        p0 = self.path_layer_0(features)
        p0 = self.GeLU(p0)
        p1 = self.path_layer_1(p0)
        p1 = self.GeLU(p1)
        p2 = self.path_layer_2(p1)
        p2 = self.GeLU(p2)
        
        feature_vector = torch.flatten(p2)

        # Extract Path Information
        driving_corridor = self.driving_corridor_layer_0(feature_vector)
        driving_corridor = self.GeLU(driving_corridor)
        driving_corridor = self.driving_corridor_layer_1(driving_corridor)
        driving_corridor = self.GeLU(driving_corridor)

        # Final Outputs

        # Anchor Points
        ego_path_x_offset = self.ego_path_x_offset(driving_corridor) + 0.5
        ego_left_x_offset = 0.3 + self.ego_left_x_offset(driving_corridor) 
        ego_right_x_offset = 0.7 + self.ego_right_x_offset(driving_corridor)
        
        # Start and End angles
        angle_start = self.angle_start(driving_corridor)

        path_prediction = torch.stack(
            [
                ego_left_x_offset, 
                ego_right_x_offset, 
                ego_path_x_offset,
                angle_start
            ], 
            dim = 0
        ).T
        
        return path_prediction