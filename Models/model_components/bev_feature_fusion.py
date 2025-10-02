#! /usr/bin/env python3
import torch
import torch.nn as nn

class BEVFeatureFusion(nn.Module):
    def __init__(self):
        super(BEVFeatureFusion, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.pool = nn.MaxPool2d(2, stride=2)
     

    def forward(self, features):
        
        # Downscaling
        feature_0 = features[0]
        feature_0 = self.pool(feature_0)
        feature_0 = self.pool(feature_0)
        feature_0 = self.pool(feature_0)
        feature_0 = self.pool(feature_0)

        feature_1 = features[1]
        feature_1 = self.pool(feature_1)
        feature_1 = self.pool(feature_1)
        feature_1 = self.pool(feature_1)

        feature_2 = features[2]
        feature_2 = self.pool(feature_2)
        feature_2 = self.pool(feature_2)

        feature_3 = features[3]
        feature_3 = self.pool(feature_3) 

        feature_4 = features[4]

        # Fusing via concatenation
        deep_features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4), 1)

        return deep_features