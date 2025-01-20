from .depth_neck import DepthNeck
from .super_depth_head import SuperDepthHead
from .super_depth_upstream import SuperDepthUpstream
import torch.nn as nn

class SuperDepthNetwork(nn.Module):
    def __init__(self, pretrained):
        super(SuperDepthNetwork, self).__init__()

        # Upstream blocks
        self.SuperDepthUpstream = SuperDepthUpstream(pretrained)

        # Neck
        self.DepthNeck = DepthNeck()

        # Depth Head
        self.SuperDepthHead = SuperDepthHead()
    

    def forward(self, image):
        features, context = self.SuperDepthUpstream(image)
        neck = self.DepthNeck(context, features)
        prediction = self.SuperDepthHead(neck, features)
        return prediction