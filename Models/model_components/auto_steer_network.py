from .backbone import Backbone
from .backbone_feature_fusion import BackboneFeatureFusion
from .auto_steer_context import AutoSteerContext
from .auto_steer_head import AutoSteerHead


import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self):
        super(AutoSteerNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone()

        # Feature Fusion
        self.BackboneFeatureFusion = BackboneFeatureFusion()

        # BEV Path Context
        self.AutoSteerContext = AutoSteerContext()

        # AutoSteer Prediction Head
        self.AutoSteerHead = AutoSteerHead()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BackboneFeatureFusion(features)
        context = self.AutoSteerContext(fused_features)
        path_prediction = self.AutoSteerHead(context)
        
        return path_prediction