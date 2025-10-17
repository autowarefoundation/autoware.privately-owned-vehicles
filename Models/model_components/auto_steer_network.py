from .backbone import Backbone
from .backbone_feature_fusion import BackboneFeatureFusion
from .auto_steer_context import AutoSteerContext
from .auto_steer_head import AutoSteerHead
from .ego_path_neck import EgoPathNeck
from .ego_path_head import EgoPathHead
from .ego_lanes_head import EgoLanes


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

        # EgoPath Neck
        self.EgopathNeck = EgoPathNeck()

        # EgoPath Head
        self.EgoLanes = EgoLanes()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BackboneFeatureFusion(features)
        context = self.AutoSteerContext(fused_features)
        neck = self.EgopathNeck(context, features)
        ego_lanes = self.EgoLanes(neck, features)
        
        return ego_lanes