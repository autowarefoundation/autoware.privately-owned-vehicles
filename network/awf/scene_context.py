#! /usr/bin/env python3
import torch
import torch.nn as nn

class SceneContext(nn.Module):
    def __init__(self):
        super(SceneContext, self).__init__()
        # Standard
        self.GeLU = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        # Context - MLP Layers
        self.context_layer_0 = nn.Linear(1280, 800)
        self.context_layer_1 = nn.Linear(800, 800)
        self.context_layer_2 = nn.Linear(800, 200)      #resolution dependent block (chosen to be a gate of input of spatial size 10x20)

        # Context - Extraction Layers (3x3 convs)
        self.context_layer_3 = nn.Conv2d(1, 128, 3, 1, 1)   #(Cin, Cout, kernel, stride, padding)
        self.context_layer_4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.context_layer_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.context_layer_6 = nn.Conv2d(512, 1280, 3, 1, 1)
     

    def forward(self, features):
        # Pooling and averaging channel layers to get a single vector
        feature_vector = torch.mean(features, dim = [2,3])      

        # MLP
        c0 = self.context_layer_0(feature_vector)        
        c0 = self.dropout(c0)
        c0 = self.GeLU(c0)
        c1 = self.context_layer_1(c0)
        c1 = self.dropout(c1)
        c1 = self.GeLU(c1)
        c2 = self.context_layer_2(c1)
        c2 = self.dropout(c2)
        c2 = self.sigmoid(c2)
        
        # Reshape to [B, 1, H, W]
        c3 = c2.view(c2.size(0), 1, 10, 20)   #resolution dependent : make [H, W] (H*W must be 200)
        # c3 = c3.unsqueeze(0)
        # c3 = c3.unsqueeze(0)
        
        # Context
        c4 = self.context_layer_3(c3)
        c4 = self.GeLU(c4)
        c5 = self.context_layer_4(c4)
        c5 = self.GeLU(c5)
        c6 = self.context_layer_5(c5)
        c6 = self.GeLU(c6)
        c7 = self.context_layer_6(c6)
        context = self.GeLU(c7)

        # Attention
        context = context*features + features
        return context   
    
class SceneContextAdaptive(nn.Module):
    def __init__(self, channels=1280):
        super(SceneContextAdaptive, self).__init__()

        # --- attivazioni ---
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)

        # --- GLOBAL CONTEXT MLP (come prima, molto espressivo) ---
        self.fc0 = nn.Linear(channels, 800)
        self.fc1 = nn.Linear(800, 800)
        self.fc2 = nn.Linear(800, channels)   # ⬅️ NON 200, ma C

        # --- LOCAL CONTEXT EXTRACTION (sulle feature vere) ---
     
        self.conv0 = nn.Conv2d(channels, 1280, 3, padding=1)
        self.conv1 = nn.Conv2d(1280, 1280, 3, padding=1)
        self.conv2 = nn.Conv2d(1280, channels, 3, padding=1)

    def forward(self, features):
        """
        features: [B, C, H, W]
        """

        B, C, H, W = features.shape

        # --------------------------------------------------
        # 1. Global semantic embedding (resolution-free)
        # --------------------------------------------------
        g = features.mean(dim=(2, 3))          # [B, C]

        g = self.fc0(g)
        g = self.dropout(g)
        g = self.gelu(g)

        g = self.fc1(g)
        g = self.dropout(g)
        g = self.gelu(g)

        g = self.fc2(g)
        g = self.sigmoid(g)                    # [B, C]

        # reshape per broadcast
        g = g.view(B, C, 1, 1)                 # [B, C, 1, 1]

        # --------------------------------------------------
        # 2. Local context refinement (spatial, real)
        # --------------------------------------------------
        x = self.conv0(features)
        x = self.gelu(x)

        x = self.conv1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)

        # --------------------------------------------------
        # 3. Gated fusion (come prima, ma corretto)
        # --------------------------------------------------
        context = x * g + features

        return context
