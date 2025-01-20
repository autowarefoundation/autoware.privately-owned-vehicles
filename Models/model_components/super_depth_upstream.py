#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch.nn as nn

class SuperDepthUpstream(nn.Module):
    def __init__(self, pretrainedModel):
        super(SuperDepthUpstream, self).__init__()

        self.pretrainedBackBone = pretrainedModel.Backbone
        for param in self.pretrainedBackBone.parameters():
            param.requires_grad = False

        self.pretrainedContext = pretrainedModel.SceneContext
        for param in self.pretrainedContext.parameters():
            param.requires_grad = False

    def forward(self, image):
        features = self.pretrainedBackBone(image)
        deep_features = features[4]
        context = self.pretrainedContext(deep_features)
        return features, context
       