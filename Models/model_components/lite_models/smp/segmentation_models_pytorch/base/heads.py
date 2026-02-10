import torch.nn as nn

from .modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.Upsample(mode="bilinear", scale_factor=upsampling, align_corners=True)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class DepthHead(nn.Sequential):
    """
    To attach to a Unet++ decoder for depth estimation. It maps the N channel feature map
    to 1 output channel (depth map).
    """
    def __init__(
        self, in_channels, out_channels=1, kernel_size=3, activation=None, upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        
        #identity since the resolution is already the input resolution (1:1 mapping)
        upsampling = (
            nn.Upsample(mode="bilinear", scale_factor=upsampling, align_corners=True)
            if upsampling > 1
            else nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class LanesHead(nn.Sequential):
    """
    To attach to a Unet++ or DeepLabV3+ decoder for lanes estimation. It maps the N channel feature map
    to multiple output channels (lanes map).
    The resolution of the output mask is 1/4 of the input resolution. This is because for lane detection,
    high-resolution output is not strictly necessary, and reducing the output size saves memory and computation.
    3 output channels correspond to ego-left, ego-right, and other lanes.
    """
    def __init__(self, in_channels: int, mid_channels: int = 256, out_channels: int = 3, activation: str = None):

        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            # # nn.BatchNorm2d(mid_channels),
            # # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

        activation = Activation(activation)

        super().__init__(block, activation)
        

class ClassificationHead(nn.Sequential):
    def __init__(
        self, in_channels, classes, pooling="avg", dropout=0.2, activation=None
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(
                "Pooling should be one of ('max', 'avg'), got {}.".format(pooling)
            )
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
