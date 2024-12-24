from .backbone import Backbone
from .scene_context import SceneContext
from .depth_neck import DepthNeck
from .super_depth_head import SuperDepthHead
import torch.nn as nn

class SuperDepthNetwork(nn.Module):
    def __init__(self):
        super(SuperDepthNetwork, self).__init__()
        
        # Encoder
        self.Backbone = Backbone()

        # Context
        self.SceneContext = SceneContext()

        # Neck
        self.DepthNeck = DepthNeck()

        # Depth Head
        self.SuperDepthHead = SuperDepthHead()
    

    def forward(self, image):
        features = self.Backbone(image)
        deep_features = features[4]
        context = self.SceneContext(deep_features)
        neck = self.DepthNeck(context, features)
        prediction = self.SuperDepthHead(neck, features)
        return prediction