import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Literal, Optional

from network.smp.segmentation_models_pytorch.base import (
    ClassificationHead,
    DepthHead,
    DepthModel,
)

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder


class DeepLabV3PlusDepth(DepthModel):
    """
    SMP-native DeepLabV3+ encoder+decoder, but with a depth regression head.

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

        # -----------------
        # Depth head (regression)
        # -----------------
        self.depth_head = DepthHead(
            in_channels=decoder_channels,
            out_channels=1,
            upsampling=scale_factor,
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
            self.freeze_encoder()  # SMP built-in: freezes params + sets norm layers eval

        if freeze_decoder:
            self.freeze_decoder()  # custom below (params + norm layers eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep SMP input checks (divisible by output_stride)
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self.check_input_shape(x)

        in_h, in_w = x.shape[-2], x.shape[-1]

        features = self.encoder(x)
        dec = self.decoder(features)  # BxCxhxw

        depth = self.depth_head(dec)  # Bx1xHxW
        return depth