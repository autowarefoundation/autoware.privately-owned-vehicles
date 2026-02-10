#! /usr/bin/env python3
import torch.nn as nn

class Scene3DHead(nn.Module):
    def __init__(self):
        super(Scene3DHead, self).__init__()
        # Standard
        self.GeLU = nn.GELU()

        # Segmentation Head - Output Layers
        self.upsample_layer_3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.skip_link_layer_3 = nn.Conv2d(32, 256, 1)
        self.decode_layer_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.decode_layer_7 = nn.Conv2d(256, 128, 3, 1, 1)
        self.upsample_layer_4 = nn.ConvTranspose2d(128, 128, 2, 2)

        # Prediction 1
        self.decode_layer_8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_9 = nn.Conv2d(128, 128, 3, 1, 1)
        self.decode_layer_10 = nn.Conv2d(128, 1, 3, 1, 1)


    def forward(self, neck, features):

        # Decoder upsample block 4
        # Upsample
        d7 = self.upsample_layer_3(neck)
         # Expand and add layer from Encoder
        d7 = d7 + self.skip_link_layer_3(features[0])
        # Double convolution
        d7 = self.decode_layer_6(d7)
        d7 = self.GeLU(d7)
        d8 = self.decode_layer_7(d7)
        d8 = self.GeLU(d8)

        # Decoder upsample block 5
        # Upsample
        up = self.upsample_layer_4(d8)

        # Triple convolution - output block 1
        d8_out = self.decode_layer_8(up)
        d8_out = self.GeLU(d8_out)
        d9_out = self.decode_layer_9(d8_out)
        d9_out = self.GeLU(d9_out)
        prediction = self.decode_layer_10(d9_out)

        # Final prediction
        return prediction