import torch
import torch.nn as nn

from typing import Optional

from Models.model_components.lite_models.heads import RegressionHead, ClassificationHead
from Models.model_components.lite_models.BaseModel import BaseModel
from Models.model_components.lite_models.modules import *

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder

class DeepLabV3Plus(BaseModel):

    # DeepLab-like models require divisible input
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
        aux_params: Optional[dict] = None,
        output_channels: int = 3,

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


        # -----------------
        # Regression head config
        # -----------------
        head_depth: int = 1,
        head_mid_channels: int | None = None,
        head_activation: Optional[str] = None,
        head_upsampling: Optional[int] = 4,
        head_kernel_size: int = 3,

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
        # Encoder
        # -----------------
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
            **encoder_kwargs,
        )

        # -----------------
        # Decoder DeepLabV3+
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
                hidden_ratio=1.0,        
                residual_scale=0.1,      
                use_depthwise=False,    
            )
            print(f"Using bottleneck: {bottleneck}")



        # -----------------
        #  head (regression)
        # -----------------

        self.head = RegressionHead(
            in_channels=decoder_channels,
            out_channels=output_channels,
            depth=head_depth,
            mid_channels=head_mid_channels,
            activation=head_activation,
            upsampling=head_upsampling,
            kernel_size=head_kernel_size
        )


        # -----------------
        # Optional: 
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
                print("[DeepLabV3Plus] Using PARTIAL encoder freeze")
                self.freeze_encoder_partial()
            else:
                print("[DeepLabV3Plus] Using FULL encoder freeze")
                self.freeze_encoder()


        if freeze_decoder:
            self.freeze_decoder()  

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

        output = self.head(decoder)  # Bx1xHxW
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
                print(f"[DeepLabV3Plus] Freezing encoder param: {name}")
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
            f"[DeepLabV3Plus] Encoder PARTIALLY frozen | "
            f"frozen params={frozen} | trainable params={trainable}"
        )

        self._is_encoder_frozen = False
        return self