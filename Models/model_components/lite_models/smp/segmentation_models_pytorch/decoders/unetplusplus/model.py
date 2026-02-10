import warnings
from typing import Any, Dict, Sequence, Optional, Union, Callable

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

from .decoder import UnetPlusPlusDecoder

import torch

class UnetPlusPlus(SegmentationModel):
    """Unet++ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Decoder of
    Unet++ is more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_norm:     Specifies normalization between Conv2D and activation.
            Accepts the following types:
            - **True**: Defaults to `"batchnorm"`.
            - **False**: No normalization (`nn.Identity`).
            - **str**: Specifies normalization type using default parameters. Available values:
              `"batchnorm"`, `"identity"`, `"layernorm"`, `"instancenorm"`, `"inplace"`.
            - **dict**: Fully customizable normalization settings. Structure:
              ```python
              {"type": <norm_type>, **kwargs}
              ```
              where `norm_name` corresponds to normalization type (see above), and `kwargs` are passed directly to the normalization layer as defined in PyTorch documentation.

            **Example**:
            ```python
            decoder_use_norm={"type": "layernorm", "eps": 1e-2}
            ```
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse** (https://arxiv.org/abs/1808.08127).
        decoder_interpolation: Interpolation mode used in decoder of the model. Available options are
            **"nearest"**, **"bilinear"**, **"bicubic"**, **"area"**, **"nearest-exact"**. Default is **"nearest"**.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
            **callable** and **None**. Default is **None**.
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **Unet++**

    Reference:
        https://arxiv.org/abs/1807.10165

    """

    _is_torch_scriptable = False


    """
    This module builds on top of UnetPlusPlusDecoder from SMP package.
    """
    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        decoder_interpolation: str = "nearest",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
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
        **kwargs: dict[str, Any],
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

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )


        #single layer segmentation head. it maps a 16 channel feature map to output classes
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)

        #initialize the weights
        self.initialize()

        #load weights from segmentation checkpoint eventually
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


        # -----------------
        # Optional freezing
        # -----------------
        if freeze_encoder:
            self.freeze_encoder()  # SMP built-in: freezes params + sets norm layers eval

        if freeze_decoder:
            self.freeze_decoder()  # custom below (params + norm layers eval)


# -------------------------
    # Loading utilities
    # -------------------------
    def load_from_segmentation_checkpoint(
        self,
        ckpt_path: str,
        load_encoder: bool = True,
        load_decoder: bool = True,
        strict: bool = True,
    ):
        print(f"[DeepLabV3PlusDepth] Loading from segmentation checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

        if load_encoder:
            enc = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
            missing, unexpected = self.encoder.load_state_dict(enc, strict=strict)
            if not strict:
                print("  [encoder] missing:", missing)
                print("  [encoder] unexpected:", unexpected)
        else:
            print("  [encoder] NOT loaded (random init)")

        if load_decoder:
            dec = {k.replace("decoder.", ""): v for k, v in state.items() if k.startswith("decoder.")}
            missing, unexpected = self.decoder.load_state_dict(dec, strict=strict)
            if not strict:
                print("  [decoder] missing:", missing)
                print("  [decoder] unexpected:", unexpected)
        else:
            print("  [decoder] NOT loaded (random init)")


    # -------------------------
    # Freezing utilities
    # -------------------------
    def freeze_decoder(self):
        """
        Freeze decoder params + put norm layers in eval to stop running stats updates.
        Mirrors SMP encoder freeze behavior.
        """
        for p in self.decoder.parameters():
            p.requires_grad = False

        for m in self.decoder.modules():
            if isinstance(m, torch.nn.modules.batchnorm._NormBase):
                m.eval()

        return self

