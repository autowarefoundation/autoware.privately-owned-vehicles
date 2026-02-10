from .ego_space_upstream import EgoSpaceUpstream
from .ego_space_head import EgoSpaceHead

import torch.nn as nn

class EgoSpaceNetwork(nn.Module):
    def __init__(self, pretrained):
        super(EgoSpaceNetwork, self).__init__()

        # Upstream blocks
        self.EgoSpaceUpstream = EgoSpaceUpstream(pretrained)

        # EgoSpace Head
        self.EgoSpaceHead = EgoSpaceHead()
    

    def forward(self, image):
        neck, features = self.EgoSpaceUpstream(image)
        prediction = self.EgoSpaceHead(neck, features)
        return prediction