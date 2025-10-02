from .backbone import Backbone
from .bev_feature_fusion import BEVFeatureFusion
from .bev_path_context import BEVPathContext
from .bev_path_neck import BEVPathNeck
from .ego_path_head import EgoPathHead
from .auto_steer_head import AutoSteerHead


import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self):
        super(AutoSteerNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone()

        # BEV Neck
        self.BEVFeatureFusion = BEVFeatureFusion()

        # BEV Path Context
        self.BEVPathContext = BEVPathContext()

        # BEV Path Neck
        self.BEVPathNeck = BEVPathNeck()

        # EgoPath Prediction Head
        self.EgoPathHead = EgoPathHead()

        # AutoSteer Prediction Head
        self.AutoSteerHead = AutoSteerHead()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BEVFeatureFusion(features)
        context = self.BEVPathContext(fused_features)
        neck = self.BEVPathNeck(context, features)
        ego_path = self.EgoPathHead(neck, features)
        path_prediction = self.AutoSteerHead(neck)
        return ego_path, path_prediction