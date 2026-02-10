import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Literal, Optional

from Models.model_components.lite_models.smp.segmentation_models_pytorch.base import (
    ClassificationHead,
    LanesHead,
    LanesModel,
)

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder

class DeepLabV3PlusLanes(LanesModel):
    """
    SMP-native DeepLabV3+ encoder+decoder, but with a lanes regression head.

    You keep:
      - check_input_shape
      - freeze_encoder / unfreeze_encoder behavior (BN stats handled)
      - all SMP encoder/decoder internals
    """

    # DeepLab-like models require divisible input (SMP enforces)
    requires_divisible_input_shape = True

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        encoder_depth: int = 5,
        encoder_output_stride: int = 16,                 # 8 or 16
        decoder_channels: int = 256,
        decoder_atrous_rates=(12, 24, 36),
        decoder_aspp_separable: bool = True,
        decoder_aspp_dropout: float = 0.5,
        head_channels: int = 256,
        activation: Optional[str] = None,
        upsampling: Optional[int] = None,
        aux_params: Optional[dict] = None,

        # loading from your previous segmentation ckpt
        segmentation_ckpt: str | None = None,  # expects ckpt["model_state"] with encoder./decoder. keys
        strict_load: bool = True,

        # loading control
        load_encoder: bool = True,
        load_decoder: bool = True,

        # freeze control
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,

        bottleneck: str = "none",       # "none" | "fcn" | "fcn_cbam" | "fcn_skip" | "fcn_skip_cbam"

        encoder_partial_load: bool = False,
        encoder_partial_depth: int = 4,   # used only if encoder_partial_load = True

        # kwargs forwarded to get_encoder (useful for timm)
        **encoder_kwargs,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError(
                "DeeplabV3 support output stride 8 or 16, got {}.".format(
                    encoder_output_stride
                )
            )
        # -----------------
        # Encoder (SMP)
        # -----------------
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **encoder_kwargs,
        )

        if upsampling is None:
            if encoder_depth <= 3:
                scale_factor = 2**encoder_depth
            else:
                scale_factor = encoder_output_stride
        else:
            scale_factor = upsampling


        # -----------------
        # Decoder (SMP DeepLabV3+)
        # -----------------
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
            aspp_separable=decoder_aspp_separable,
            aspp_dropout=decoder_aspp_dropout,
        )

        if bottleneck != "none":
            self.bottleneck = Bottleneck(
                in_channels=self.encoder.out_channels[-1],
                out_channels=self.encoder.out_channels[-1],
                mode=bottleneck,
                hidden_ratio=1.0,        # puoi provare 0.5 / 1.0 / 2.0
                residual_scale=0.1,      # 0.05–0.2 sono buoni range
                use_depthwise=False,     # True se vuoi ultra-light
            )
            print(f"Using bottleneck: {bottleneck}")



        # -----------------
        # Lanes head (regression)
        # -----------------
        self.lanes_head = LanesHead(
            in_channels=decoder_channels,
            mid_channels=head_channels,
            out_channels=3,  # 3 lane channels
            activation=activation,
        )

        # -----------------
        # Optional: load encoder/decoder weights from your segmentation ckpt
        # -----------------
        if segmentation_ckpt is not None:
            self.load_from_segmentation_checkpoint(
                segmentation_ckpt,
                load_encoder=load_encoder,
                load_decoder=load_decoder,
                strict=strict_load,
                encoder_partial_load=encoder_partial_load,
                encoder_partial_depth=encoder_partial_depth,
            )


        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
        self.name = "deeplabv3-{}".format(encoder_name)



        # initialize every weight
        self.initialize()


        # -----------------
        # Optional freezing
        # -----------------
        if freeze_encoder:
            if encoder_partial_load:
                print("[DeepLabV3PlusLanes] Using PARTIAL encoder freeze")
                self.freeze_encoder_partial()
            else:
                print("[DeepLabV3PlusLanes] Using FULL encoder freeze")
                self.freeze_encoder()


        if freeze_decoder:
            self.freeze_decoder()  # custom below (params + norm layers eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep SMP input checks (divisible by output_stride)
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self.check_input_shape(x)

        #encoder
        features = self.encoder(x)

        if hasattr(self, 'bottleneck'):
            #apply the bottleneck to the encoder output
            features = self.bottleneck(features)
        
        #decoder pass
        decoder = self.decoder(features)  # BxCxhxw

        lanes = self.lanes_head(decoder)  # Bx1xHxW
        return lanes


    def freeze_encoder_partial(self):
        """
        Freeze ONLY the encoder parameters that were loaded from checkpoint.
        The remaining encoder parameters stay trainable.
        """
        if not hasattr(self, "_encoder_loaded_param_names"):
            raise RuntimeError(
                "freeze_encoder_partial() called but no encoder weights were loaded yet."
            )

        frozen = 0
        trainable = 0

        for name, param in self.encoder.named_parameters():
            if name in self._encoder_loaded_param_names:
                param.requires_grad = False
                frozen += 1
                print(f"[LanesModel] Freezing encoder param: {name}")
            else:
                param.requires_grad = True
                trainable += 1

        # Handle normalization layers: freeze only those fully frozen
        for module_name, module in self.encoder.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._NormBase):
                # If all params of this BN belong to frozen set → freeze stats
                bn_params = [
                    f"{module_name}.weight",
                    f"{module_name}.bias",
                ]
                if all(p in self._encoder_loaded_param_names for p in bn_params):
                    module.eval()


        print(
            f"[LanesModel] Encoder PARTIALLY frozen | "
            f"frozen params={frozen} | trainable params={trainable}"
        )

        self._is_encoder_frozen = False   # important: avoid SMP global freeze logic
        return self



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