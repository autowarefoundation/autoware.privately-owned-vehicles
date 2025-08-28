from .backbone import Backbone
from .bev_path_context import BEVPathContext
from .bev_path_neck import BEVPathNeck
from .auto_steer_head import AutoSteerHead


import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self):
        super(AutoSteerNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone()

        # BEV Path Context
        self.BEVPathContext = BEVPathContext()

        # BEV Neck
        self.BEVPathNeck = BEVPathNeck()

        # AutoSteer Prediction Head
        self.AutoSteerHead = AutoSteerHead()
    

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BEVPathNeck(features)
        context = self.BEVPathContext(fused_features)
        ego_path, ego_left_lane, ego_right_lane = self.AutoSteerHead(context)
        return ego_path, ego_left_lane, ego_right_lane