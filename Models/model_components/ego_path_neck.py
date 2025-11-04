#! /usr/bin/env python3
import torch.nn as nn

class EgoPathNeck(nn.Module):
    def __init__(self):
        super(EgoPathNeck, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Decoder - Neck Layers 
        self.upsample_layer_0 = nn.ConvTranspose2d(1456, 1456, 2, 2)
        self.skip_link_layer_0 = nn.Conv2d(80, 1456, 1)
        self.decode_layer_0 = nn.Conv2d(1456, 768, 3, 1, 1)
        self.decode_layer_1 = nn.Conv2d(768, 768, 3, 1, 1)

        self.upsample_layer_1 = nn.ConvTranspose2d(768, 768, 2, 2)
        self.skip_link_layer_1 = nn.Conv2d(40, 768, 1)
        self.decode_layer_2 = nn.Conv2d(768, 512, 3, 1, 1)
        self.decode_layer_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.upsample_layer_2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.skip_link_layer_2 = nn.Conv2d(24, 512, 1)
        self.decode_layer_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.decode_layer_5 = nn.Conv2d(512, 256, 3, 1, 1)

    def forward(self, context, features):

        # Decoder upsample block 1
        # Upsample
        d0 = self.upsample_layer_0(context)
        # Add layer from Encoder
        d0 = d0 + self.skip_link_layer_0(features[3])
        # Double Convolution
        d1 = self.decode_layer_0 (d0)
        d1 = self.GeLU(d1)
        d2 = self.decode_layer_1(d1)
        d2 = self.GeLU(d2)

        # Decoder upsample block 2
        # Upsample
        d3 = self.upsample_layer_1(d2)
        # Expand and add layer from Encoder
        d3 = d3 + self.skip_link_layer_1(features[2])
        # Double convolution
        d3 = self.decode_layer_2(d3)
        d3 = self.GeLU(d3)
        d4 = self.decode_layer_3(d3)
        d5 = self.GeLU(d4)

        # Decoder upsample block 3
        # Upsample
        d5 = self.upsample_layer_2(d5)
         # Expand and add layer from Encoder
        d5 = d5 + self.skip_link_layer_2(features[1])
        # Double convolution
        d5 = self.decode_layer_4(d5)
        d5 = self.GeLU(d5)
        d6 = self.decode_layer_5(d5)
        neck = self.GeLU(d6)

        return neck