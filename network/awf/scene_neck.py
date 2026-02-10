#! /usr/bin/env python3
import torch.nn as nn

class SceneNeck(nn.Module):
    def __init__(self):
        super(SceneNeck, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Decoder - Neck Layers 

        #INPUT : (1280xH/32xW/32) from Backbone + Context (/32 since the efficientnet has a global stride of 32, from input to output)
        self.upsample_layer_0 = nn.ConvTranspose2d(1280, 1280, 2, 2)        
        self.skip_link_layer_0 = nn.Conv2d(80, 1280, 1)
        self.decode_layer_0 = nn.Conv2d(1280, 768, 3, 1, 1)
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
        d0 = self.upsample_layer_0(context)                 #upsample from (H/32, W/32) to (H/16, W/16) (example 10x20 -> 20x40)
        # Add layer from Encoder
        d0 = d0 + self.skip_link_layer_0(features[3])       #features[3] is (H/16, W/16)
        # Double Convolution
        d1 = self.decode_layer_0 (d0)                       #(H/16, W/16)
        d1 = self.GeLU(d1)                                  
        d2 = self.decode_layer_1(d1)                        #(H/16, W/16)
        d2 = self.GeLU(d2)

        # Decoder upsample block 2
        # Upsample
        d3 = self.upsample_layer_1(d2)                      #upsample from (H/16, W/16) to (H/8, W/8)
        # Expand and add layer from Encoder
        d3 = d3 + self.skip_link_layer_1(features[2])       #features[2] is (H/8, W/8)
        # Double convolution
        d3 = self.decode_layer_2(d3)                        #(H/8, W/8)
        d3 = self.GeLU(d3)
        d4 = self.decode_layer_3(d3)                        #(H/8, W/8)
        d5 = self.GeLU(d4)

        # Decoder upsample block 3
        # Upsample
        d5 = self.upsample_layer_2(d5)                      #upsample from (H/8, W/8) to (H/4, W/4)
         # Expand and add layer from Encoder
        d5 = d5 + self.skip_link_layer_2(features[1])       #features[1] is (H/4, W/4)
        # Double convolution
        d5 = self.decode_layer_4(d5)                        #(H/4, W/4)
        d5 = self.GeLU(d5)
        d6 = self.decode_layer_5(d5)                        #(H/4, W/4)
        neck = self.GeLU(d6)

        return neck