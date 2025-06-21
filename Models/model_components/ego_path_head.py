#! /usr/bin/env python3
import torch
import torch.nn as nn

class EgoPathHead(nn.Module):
    def __init__(self):
        super(EgoPathHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.dropout = nn.Dropout(p=0.25)

        # Context - MLP Layers
        self.ego_path_layer_0 = nn.Linear(1280, 800)
        self.ego_path_layer_1 = nn.Linear(800, 800)
        self.ego_path_layer_2 = nn.Linear(800, 200)
        self.ego_path_layer_3 = nn.Linear(200, 33)
        self.ego_path_layer_4 = nn.Linear(200, 33)

    def forward(self, features):
        # Pooling and averaging channel layers to get a single vector
        feature_vector = torch.mean(features, dim = [2,3])

        # MLP
        p0 = self.ego_path_layer_0(feature_vector)
        p0 = self.dropout(p0)
        p0 = self.GeLU(p0)
        p1 = self.ego_path_layer_1(p0)
        p1 = self.dropout(p1)
        p1 = self.GeLU(p1)
        p2 = self.ego_path_layer_2(p1)
        feature = self.GeLU(p2)

        # Regression output
        points_x_offset = self.ego_path_layer_3(feature)

        # Classification output
        points_validity = self.ego_path_layer_4(feature)

        # Final result
        return points_x_offset, points_validity