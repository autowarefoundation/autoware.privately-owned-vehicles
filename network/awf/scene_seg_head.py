#! /usr/bin/env python3
import torch.nn as nn

class SceneSegHead(nn.Module):
    def __init__(self, N_classes=19):
        super(SceneSegHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Segmentation Head - Output Layers
        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)

        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 64, 3, 1, 1)

        # Final output layer, apply N classes, as Cityscapes has 19 classes + void
        self.decode_layer_10 = nn.Conv2d(64, N_classes, 3, 1, 1)

    def forward(self, neck, features):

        # Decoder upsample block 4
        # Upsample
        d7 = self.upsample_layer_3(neck)                #upsample from (H/4, W/4) to (H/2, W/2)
         # Expand and add layer from Encoder    
        d7 = d7 + self.skip_link_layer_3(features[0])   #features[0] is (H/2, W/2)
        # Double convolution
        d7 = self.decode_layer_6(d7)                    #(H/2, W/2)
        d7 = self.GeLU(d7)
        d8 = self.decode_layer_7(d7)                    #(H/2, W/2)
        d8 = self.GeLU(d8)

        # Decoder upsample block 5
        # Upsample
        d8 = self.upsample_layer_4(d8)                  #upsample from (H/2, W/2) to (H, W)
        # Double convolution
        d8 = self.decode_layer_8(d8)                    #(H, W)
        d8 = self.GeLU(d8)
        d9 = self.decode_layer_9(d8)                    #(H, W)
        d10 = self.GeLU(d9)
        # Output
        output = self.decode_layer_10(d10)

        return output