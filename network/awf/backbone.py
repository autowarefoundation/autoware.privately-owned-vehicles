#! /usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    ResNet50_Weights,
)


class Backbone(nn.Module):
    """
    Generic CNN backbone with multi-scale feature extraction.

    Supported backbones:
      - efficientnet_b0 (default)
      - efficientnet_b1
      - resnet50
    """

    def __init__(
        self,
        name: str = "efficientnet_b0",
        pretrained: bool = True,
        freeze: bool = False,
        pretrained_model: object = None,  # custom pretrained backbone
    ):
        super().__init__()

        self.name = name.lower()
        self.pretrained = pretrained
        self.freeze = freeze

        self.pretrainedBackBone = None

        # --------------------------------------------------
        # Custom pretrained backbone
        # --------------------------------------------------
        if pretrained_model is not None:
            print("[Backbone] Using custom pretrained model as backbone...")
            self.pretrainedBackBone = pretrained_model
            self.pretrained = False

        # --------------------------------------------------
        # Built-in backbones
        # --------------------------------------------------
        elif self.name in ["efficientnet_b0", "efficientnet_b1"]:
            self._build_efficientnet()

        elif self.name == "resnet50":
            self._build_resnet50()

        else:
            raise ValueError(f"Unsupported backbone: {self.name}")

        if freeze:
            self._freeze_backbone()

    # --------------------------------------------------
    # Backbone builders
    # --------------------------------------------------
    def _build_efficientnet(self):
        if self.name == "efficientnet_b0":
            weights = (
                EfficientNet_B0_Weights.IMAGENET1K_V1
                if self.pretrained else None
            )
            model = models.efficientnet_b0(weights=weights)

        elif self.name == "efficientnet_b1":
            weights = (
                EfficientNet_B1_Weights.IMAGENET1K_V1
                if self.pretrained else None
            )
            model = models.efficientnet_b1(weights=weights)

        # EfficientNet encoder blocks
        self.stages = model.features

        # Multi-scale taps (VALID for both B0 and B1)
        self.out_indices = [0, 2, 3, 4, 8]

    def _build_resnet50(self):
        weights = ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _freeze_backbone(self):
        """
        Freeze backbone weights.
        NOTE: BatchNorm running stats are NOT frozen.
        """
        for p in self.parameters():
            p.requires_grad = False

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x):
        """
        Returns a list of feature maps at different scales.
        """
        # --------------------------------------------------
        # Custom pretrained backbone
        # --------------------------------------------------
        if self.pretrainedBackBone is not None:
            return self.pretrainedBackBone(x)

        # --------------------------------------------------
        # EfficientNet (B0 / B1)
        # --------------------------------------------------
        if self.name in ["efficientnet_b0", "efficientnet_b1"]:
            feats = []
            for i, block in enumerate(self.stages):
                x = block(x)
                if i in self.out_indices:
                    feats.append(x)
            return feats  # [l0, l2, l3, l4, l8]

        # --------------------------------------------------
        # ResNet50
        # --------------------------------------------------
        elif self.name == "resnet50":
            feats = []

            x = self.stem(x)     # 1/4
            feats.append(x)

            x = self.layer1(x)  # 1/4
            feats.append(x)

            x = self.layer2(x)  # 1/8
            feats.append(x)

            x = self.layer3(x)  # 1/16
            feats.append(x)

            x = self.layer4(x)  # 1/32
            feats.append(x)

            return feats
