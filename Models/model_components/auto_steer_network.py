from .auto_steer_upstream import AutoSteerUpstream
from .auto_steer_head import AutoSteerHead

import torch.nn as nn

class AutoSteerNetwork(nn.Module):
    def __init__(self, pretrained):
        super(AutoSteerNetwork, self).__init__()

        # Upstream blocks
        self.AutoSteerUpstream = AutoSteerUpstream(pretrained)

        # AutoSteer Head
        self.AutoSteerHead = AutoSteerHead()
    

    def forward(self, image, neck_prev, neck_prev_prev):
        ego_lanes, neck = self.AutoSteerUpstream(image)
        steering = self.AutoSteerHead(neck, neck_prev, neck_prev_prev)
        return ego_lanes, steering