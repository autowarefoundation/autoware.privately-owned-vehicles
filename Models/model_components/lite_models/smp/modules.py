from typing import Any, Dict, Union

import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


def get_norm_layer(
    use_norm: Union[bool, str, Dict[str, Any]], out_channels: int
) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")

    # Step 1. Convert tot dict representation

    ## Check boolean
    if use_norm is True:
        norm_params = {"type": "batchnorm"}
    elif use_norm is False:
        norm_params = {"type": "identity"}

    ## Check string
    elif isinstance(use_norm, str):
        norm_str = use_norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in supported_norms:
            norm_params = {"type": norm_str}
        else:
            raise ValueError(
                f"Unrecognized normalization type string provided: {use_norm}. Should be in "
                f"{supported_norms}"
            )

    ## Check dict
    elif isinstance(use_norm, dict):
        norm_params = use_norm

    else:
        raise ValueError(
            f"Invalid type for use_norm should either be a bool (batchnorm/identity), "
            f"a string in {supported_norms}, or a dict like {{'type': 'batchnorm', **kwargs}}"
        )

    # Step 2. Check if the dict is valid
    if "type" not in norm_params:
        raise ValueError(
            f"Malformed dictionary given in use_norm: {use_norm}. Should contain key 'type'."
        )
    if norm_params["type"] not in supported_norms:
        raise ValueError(
            f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}"
        )
    if norm_params["type"] == "inplace" and InPlaceABN is None:
        raise RuntimeError(
            "In order to use `use_norm='inplace'` the inplace_abn package must be installed. Use:\n"
            "  $ pip install -U wheel setuptools\n"
            "  $ pip install inplace_abn --no-build-isolation\n"
            "Also see: https://github.com/mapillary/inplace_abn"
        )

    # Step 3. Initialize the norm layer
    norm_type = norm_params["type"]
    norm_kwargs = {k: v for k, v in norm_params.items() if k != "type"}

    if norm_type == "inplace":
        norm = InPlaceABN(out_channels, **norm_kwargs)
    elif norm_type == "batchnorm":
        norm = nn.BatchNorm2d(out_channels, **norm_kwargs)
    elif norm_type == "identity":
        norm = nn.Identity()
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(out_channels, **norm_kwargs)
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        norm = get_norm_layer(use_norm, out_channels)

        is_identity = isinstance(norm, nn.Identity)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )

        is_inplaceabn = InPlaceABN is not None and isinstance(norm, InPlaceABN)
        activation = nn.Identity() if is_inplaceabn else nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, norm, activation)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif name.lower() == "relu":
            self.activation = nn.ReLU(inplace=True, **params)
        elif name.lower() == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)



# ------------------------------------------------------------
# CBAM (lightweight, no BatchNorm)
# ------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=(2, 3))        # [B, C]
        mx  = torch.amax(x, dim=(2, 3))        # [B, C]

        attn = self.mlp(avg) + self.mlp(mx)
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ------------------------------------------------------------
# Bottleneck
# ------------------------------------------------------------
class Bottleneck(nn.Module):
    """
    Bottleneck module to better map backbone features to decoder input.

    Modes:
      - "fcn"
      - "fcn_cbam"
      - "fcn_skip"
      - "fcn_skip_cbam"
      - "none"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "none",
        hidden_ratio: float = 1.0,
        residual_scale: float = 1.0,
        use_depthwise: bool = False,
    ):
        super().__init__()

        assert mode in {
            "none",
            "fcn",
            "fcn_cbam",
            "fcn_skip",
            "fcn_skip_cbam",
        }, f"Invalid bottleneck mode: {mode}"

        self.mode = mode
        self.use_skip = "skip" in mode
        self.use_cbam = "cbam" in mode
        self.residual_scale = residual_scale

        if mode == "none":
            self.block = nn.Identity()
            self.cbam = None
            return

        hidden_channels = int(out_channels * hidden_ratio)

        # -------------------------
        # Convolution block (2 layers)
        # -------------------------
        if use_depthwise:
            # depthwise + pointwise (optional)
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=1,
                    bias=True,
                ),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )

        # -------------------------
        # Optional CBAM
        # -------------------------
        self.cbam = CBAM(out_channels) if self.use_cbam else None

        # -------------------------
        # Skip projection (if channels mismatch)
        # -------------------------
        if self.use_skip and in_channels != out_channels:
            self.skip_proj = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.skip_proj = None

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Apply bottleneck ONLY to the last feature map.
        """
        if self.mode == "none":
            return features

        x = features[-1]

        y = self.block(x)

        if self.cbam is not None:
            y = self.cbam(y)

        if self.use_skip:
            skip = x
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)

            y = skip + self.residual_scale * y

        features = list(features)
        features[-1] = y
        return features