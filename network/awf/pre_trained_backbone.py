#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch.nn as nn

class PreTrainedBackbone(nn.Module):
    def __init__(self, pretrainedModel):
        super(PreTrainedBackbone, self).__init__()

        self.pretrainedBackBone = pretrainedModel.Backbone
        for param in self.pretrainedBackBone.parameters():
            param.requires_grad = False

    def forward(self, image):
        features = self.pretrainedBackBone(image)
        return features