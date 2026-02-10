import segmentation_models_pytorch as smp

from utils.optimizer import build_optimizer, build_scheduler
from utils.loss import DepthLoss

from training.depth.trainer_base import DepthTrainerBase

# from network.smp.unetplusplus.depth.unetplusplus_depth import UnetPlusPlusDepth
from network.smp.UnetPlusPlus import UnetPlusPlus



class UnetPlusPlusTrainer(DepthTrainerBase):
    """
    Minimal trainer that reproduces the current script behaviour:
      - Train: ConcatDataset across multiple training sets
      - Val: one dataloader per validation set
      - Supports train_mode: "epoch" or "steps"
      - Supports val_mode: "epoch" or "steps"
      - Saves last.pth every time validation runs (if enabled)
      - Saves best_mIoU.pth on improvement
      - Resumes from checkpoint with optimizer/scheduler + counters
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(self, cfg: dict):

        # initialize base trainer with the configuration
        super().__init__(cfg)

        print("[UnetPlusPlusTrainer] Configuration : ", cfg)

        #build output directories, for checkpints and logs (without overwriting)
        self._build_output_dirs()

        #build wandb logger
        self._build_logger()

        
        #build data loaders (train + val)
        self._build_datasets()

        #choose the architecture to be used
        self._build_encoder_decoder()

        #build model, loss, optimizer, scheduler
        self._build_model_stack()

        #build training state (counters, best metrics, etc)
        self._build_training_state()

        #resume in case the checkpoint path is specified
        self._maybe_resume()

    # -------------------------
    # Builders
    # -------------------------

    def _build_encoder_decoder(self):

        self.network_cfg = self.cfg["network"]

        self.backbone_cfg = self.network_cfg.get("backbone", {})
        self.decoder_cfg = self.network_cfg.get("decoder", {})

        #modify the name of the backbone type from efficientnet_b0 to timm-efficientnet-b0
        self.backbone_cfg["type"] = "timm-" + self.backbone_cfg["type"].replace("_", "-")
        return 


    def _build_model_stack(self):
        print("Building model...")
        print(f"Backbone config: {self.backbone_cfg}")
        print(f"Decoder config: {self.decoder_cfg}")

        #pass configuration of the architecture to the model

        #standard model taken from the segmentation_models_pytorch library

        #
        # self.model = UnetPlusPlusDepth(
        #     encoder_name=self.backbone_cfg["type"],
        #     encoder_weights="imagenet" if self.backbone_cfg["pretrained"] else None,
        #     in_channels=3,
        #     classes=1,  # assuming 1 class for depth prediction
        #     decoder_interpolation=self.decoder_cfg["decoder_interpolation"],
        #     decoder_channels=self.decoder_cfg["unetplusplus_decoder_channels"],
        #     decoder_attention_type=self.decoder_cfg["decoder_attention"],
        #     segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
        #     load_encoder=self.backbone_cfg["load_encoder"],
        #     load_decoder=self.decoder_cfg["load_decoder"],
        #     freeze_encoder=self.backbone_cfg["freeze_encoder"],
        #     freeze_decoder=self.decoder_cfg["freeze_decoder"],
        # )

        self.model = UnetPlusPlus(
            encoder_name=self.backbone_cfg["type"],
            segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
            classes=1,  # assuming 1 class for depth prediction
            decoder_interpolation=self.decoder_cfg["decoder_interpolation"],
            decoder_channels=self.decoder_cfg["unetplusplus_decoder_channels"],
            decoder_attention_type=self.decoder_cfg["decoder_attention"],
            load_encoder=self.backbone_cfg.get("load_encoder", True),
            load_decoder=self.decoder_cfg.get("load_decoder", True),
            freeze_encoder=self.backbone_cfg.get("freeze_encoder", False),
            freeze_decoder=self.decoder_cfg.get("freeze_decoder", False),
            encoder_depth=self.backbone_cfg.get("encoder_depth", 5),
            # bottleneck=self.bottleneck_cfg.get("type", "none"),
            encoder_partial_load=self.backbone_cfg.get("encoder_partial_load", False),
            encoder_partial_depth=self.backbone_cfg.get("encoder_partial_depth", 4),
            head_activation=self.network_cfg.get("head_activation", None),       #activation function after each convolutional layer in the regression head (except the last one)
            head_depth=self.network_cfg.get("head_depth", 1),               #how many convolutional layers to use in the regression head
            head_mid_channels=self.network_cfg.get("head_mid_channels", None),
            output_channels=self.network_cfg.get("output_channels", 1),   # for depth estimation, 1 channel depth map
            )
            

                # self.model = DeepLabV3PlusLanes(
                #     encoder_name=self.backbone_cfg["type"],
                #     segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
                #     encoder_output_stride=self.backbone_cfg["output_stride"],
                #     aspp_dilations=self.decoder_cfg["aspp_dilations"],
                #     decoder_channels=self.decoder_cfg["deeplabv3plus_decoder_channels"],
                #     load_encoder=self.backbone_cfg["load_encoder"],
                #     load_decoder=self.decoder_cfg["load_decoder"],
                #     freeze_encoder=self.backbone_cfg["freeze_encoder"],
                #     freeze_decoder=self.decoder_cfg["freeze_decoder"],
                #     encoder_depth=self.backbone_cfg.get("encoder_depth", 5),
                #     bottleneck=self.bottleneck_cfg.get("type", "none"),
                #     encoder_partial_load=self.backbone_cfg.get("encoder_partial_load", False),
                #     encoder_partial_depth=self.backbone_cfg.get("encoder_partial_depth", 4),
                #     )
            
        print(f"[SMP] UNet++ | encoder={self.backbone_cfg['type']} | "
                f"pretrained={self.backbone_cfg['pretrained']}")
                
    
        #move model to device
        self.model.to(self.device)

        print("Building loss, optimizer, and scheduler...")
        self.loss_fn_train = DepthLoss(is_train=True)     #
        self.loss_fn_val = DepthLoss(is_train=False)     #
        
        self.optimizer = build_optimizer(self.optim_cfg, self.model.parameters())
        self.scheduler = build_scheduler(
            self.sched_cfg,
            self.optimizer,
            self.train_cfg,
            self.steps_per_epoch,
        )