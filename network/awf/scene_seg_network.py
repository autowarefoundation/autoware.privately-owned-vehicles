from .backbone import Backbone
from .scene_context import SceneContext, SceneContextAdaptive
from .scene_neck import SceneNeck
from .scene_seg_head import SceneSegHead
import torch.nn as nn
import torch


class SceneSegNetwork(nn.Module):
    def __init__(self, classes=19, is_testing: bool = True, backbone: dict = None, context: str = "standard"):
        super().__init__()

        if backbone is None:
            backbone = {"type": "efficientnet_b0", "pretrained": True, "freeze": False}
            
        #build the backbone architecture based on the configuration
        backbone_name = backbone.get("type", "efficientnet_b0")
        pretrained = backbone.get("pretrained", True)
        freeze = backbone.get("freeze", False)

        self.Backbone = Backbone(
            name=backbone_name,
            pretrained=pretrained,
            freeze=freeze
        )
        

        #building the scene context module based on the configuration (standard or adaptive)
        if context == "standard":
            self.SceneContext = SceneContext()
        elif context == "adaptive":
            self.SceneContext = SceneContextAdaptive()
        else:
            raise ValueError(f"Unknown context type: {context}")

        self.SceneNeck = SceneNeck()
        self.SceneSegHead = SceneSegHead(N_classes=classes)

        self.is_testing = is_testing        #in training (and evaluation) we need the logits for loss computation
 
    def forward(self, image):
        features = self.Backbone(image) #extract intermediate features from the backbone (default stages : 0,2,3,4,8 for efficientnet_b0)
        
        backbone_output = features[4]   #use the last feature map as input to the scene context module
        
        context = self.SceneContext(backbone_output)    #context only needs the last feature map
        
        neck = self.SceneNeck(context, features)
        
        logits = self.SceneSegHead(neck, features)  # [N,C,H,W]

        if self.is_testing:
            return torch.argmax(logits, dim=1).to(torch.uint8)

        return logits
