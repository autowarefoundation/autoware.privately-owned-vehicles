import torch
import torch.nn as nn

import warnings

from typing import Any, Dict, Sequence, Optional, Union

from Models.model_components.lite_models.heads import RegressionHead, ClassificationHead
from Models.model_components.lite_models.BaseModel import BaseModel
from Models.model_components.lite_models.modules import *

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

class UnetPlusPlus(BaseModel):
    """
    SMP-native DeepLabV3+ encoder+decoder, but with a regression head.

    You keep:
      - check_input_shape
      - freeze_encoder / unfreeze_encoder behavior (BN stats handled)
      - all SMP encoder/decoder internals
    """

    requires_divisible_input_shape = True

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        encoder_depth: int = 5,
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        aux_params: Optional[dict] = None,
        output_channels: int =1,

        # loading from previous segmentation ckpt
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


        # -----------------
        # Regression head config
        # -----------------
        head_depth: int = 1,
        head_mid_channels: int | None = None,
        head_activation: Optional[str] = None,

        **kwargs,
    ):
        super().__init__()

        if encoder_name.startswith("mit_b"):
            raise ValueError(
                "UnetPlusPlus is not support encoder_name={}".format(encoder_name)
            )

        decoder_use_batchnorm = kwargs.pop("decoder_use_batchnorm", None)
        if decoder_use_batchnorm is not None:
            warnings.warn(
                "The usage of decoder_use_batchnorm is deprecated. Please modify your code for decoder_use_norm",
                DeprecationWarning,
                stacklevel=2,
            )
            decoder_use_norm = decoder_use_batchnorm
        # -----------------
        # Encoder
        # -----------------
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        # -----------------
        # Decoder 
        # -----------------
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

        if bottleneck != "none":
            self.bottleneck = Bottleneck(
                in_channels=self.encoder.out_channels[-1],
                out_channels=self.encoder.out_channels[-1],
                mode=bottleneck,
                hidden_ratio=1.0,       
                residual_scale=0.1,    
                use_depthwise=False,  
            )
            print(f"Using bottleneck: {bottleneck}")



        # -----------------
        #  head (regression)
        # -----------------

        self.head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=output_channels,
            depth=head_depth,
            mid_channels=head_mid_channels,
            activation=head_activation,
            upsampling=1,           #unet++ already upsamples to input size
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
                print("[UnetPlusPlus] Using PARTIAL encoder freeze")
                self.freeze_encoder_partial()
            else:
                print("[UnetPlusPlus] Using FULL encoder freeze")
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
        decoder = self.decoder(features)

        output = self.head(decoder) 
        return output


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
                print(f"[UnetPlusPlus] Freezing encoder param: {name}")
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
            f"[UnetPlusPlus] Encoder PARTIALLY frozen | "
            f"frozen params={frozen} | trainable params={trainable}"
        )

        self._is_encoder_frozen = False   # important: avoid SMP global freeze logic
        return self