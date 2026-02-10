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
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
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



class RegressionHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int = 1,
        mid_channels: int | None = None,
        kernel_size: int = 3,
        activation: str | None = None,
        upsampling: int = 1,
    ):
        """
        Configurable regression head.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolution layers (default=1)
            mid_channels: number of channels for intermediate layers
                          (default = in_channels)
            kernel_size: convolution kernel size
            activation: activation name for Activation(...). If None, no activation is applied.
            upsampling: final upsampling factor (default=1)
        """
        assert depth >= 1, "depth must be >= 1"

        if mid_channels is None:
            mid_channels = in_channels

        layers = []
        padding = kernel_size // 2

        # --------------------------------------------------
        # Convolution stack
        # --------------------------------------------------
        for i in range(depth):
            is_last = (i == depth - 1)

            cin = in_channels if i == 0 else mid_channels
            cout = out_channels if is_last else mid_channels

            layers.append(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )
            )

            # Optional activation after each conv. Do not apply after last conv if it's the output layer
            if is_last == False:
                #apply activation only in internal layers of the head
                layers.append(Activation(activation))

        # --------------------------------------------------
        # Optional upsampling (applied after conv stack)
        # --------------------------------------------------
        if upsampling > 1:
            layers.append(
                nn.Upsample(
                    mode="bilinear",
                    scale_factor=upsampling,
                    align_corners=True,
                )
            )

        # --------------------------------------------------
        # Optional final activation (classic behavior)
        # --------------------------------------------------
        # if activation is not None:
        #     layers.append(Activation(activation))

        super().__init__(*layers)
