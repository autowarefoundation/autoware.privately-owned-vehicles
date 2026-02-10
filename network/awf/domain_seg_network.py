from .domain_seg_upstream import DomainSegUpstream
from .domain_seg_head import DomainSegHead

import torch.nn as nn
import torch

class DomainSegNetwork(nn.Module):
    def __init__(self, pretrained, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        super(DomainSegNetwork, self).__init__()

        # Upstream blocks
        self.DomainSegUpstream = DomainSegUpstream(pretrained)

        # DomainSeg Head
        self.DomainSegHead = DomainSegHead()

        # Buffers for normalization. put inside the model to be saved/loaded with it, so everything is self-contained is inside the onnx
        mean = torch.tensor(mean, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(std , dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image):
        # assume input [N,3,H,W] con valori [0..1]
        image = (image - self.mean) / self.std

        neck, features = self.DomainSegUpstream(image)

        prediction = self.DomainSegHead(neck, features)
        
        return prediction