from .backbone import Backbone
from .depth_context import DepthContext
from .scene_3d_neck import Scene3DNeck
from .scene_3d_head import Scene3DHead

import torch.nn as nn
import torch

class Scene3DNetwork(nn.Module):
    def __init__(self, pretrainedModel: object = None):
        super(Scene3DNetwork, self).__init__()

        if pretrainedModel is not None:
            print("[Scene3DNetwork] Using custom pretrained model as backbone...")
            #use the provided model instead. pretrained model is only Backbone
            try:
                print("[Scene3DNetwork] Loading custom pretrained model...")
                self.pretrainedBackBone = pretrainedModel.Backbone
            except:
                print("[Scene3DNetwork] Failed to load custom pretrained model. Using default backbone...")
        else:
            self.pretrainedBackBone = None

        # Upstream blocks
        self.Backbone = Backbone(pretrained_model=self.pretrainedBackBone)

        # Depth Context
        self.DepthContext = DepthContext()

        # Neck
        self.DepthNeck = Scene3DNeck()

        # Depth Head
        self.DepthHead = Scene3DHead()


    def forward(self, image):
        # assume input [N,3,H,W] con valori [0..1]
        # image = (image - self.mean) / self.std

        features = self.Backbone(image)     #extract intermediate features from the backbone (default stages : 0,2,3,4,8 for efficientnet_b0) 
        
        backbone_output = features[4]       #use the last feature map as input to the scene context module
        
        context = self.DepthContext(backbone_output)    #context only needs the last feature map
        
        neck = self.DepthNeck(context, features)        
        
        prediction = self.DepthHead(neck, features)
        
        return prediction