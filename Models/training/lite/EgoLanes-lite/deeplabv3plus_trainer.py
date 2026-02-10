import segmentation_models_pytorch as smp

from utils.optimizer import build_optimizer, build_scheduler
from utils.loss import LanesLoss

from training.lane_detection.trainer_base import LanesTrainerBase

from network.smp.DeepLabv3Plus import DeepLabV3Plus


class DeepLabV3PlusTrainer(LanesTrainerBase):
    """
    Minimal trainer that reproduces the current script behaviour:
      - Train: ConcatDataset across multiple training sets
      - Val: one dataloader per validation set
      - Supports train_mode: "epoch" or "steps"
      - Supports val_mode: "epoch" or "steps"
      - Saves last.pth every time validation runs (if enabled)
      - Saves best.pth on improvement
      - Resumes from checkpoint with optimizer/scheduler + counters
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(self, cfg: dict):

        # initialize base trainer with the configuration
        super().__init__(cfg)

        print("[DeepLabV3PlusTrainer] Configuration : ", cfg)

        #build output directories, for checkpints and logs (without overwriting)
        self._build_output_dirs()

        #build wandb logger immediately, so every log is recorded
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
        self.bottleneck_cfg = self.network_cfg.get("bottleneck", {})
        self.head_cfg = self.network_cfg.get("head", {})
        
        #modify the name of the backbone type from efficientnet_b0 to timm-efficientnet-b0
        if "timm" not in self.backbone_cfg["type"]:
            self.backbone_cfg["type"] = "timm-" + self.backbone_cfg["type"].replace("_", "-")
        return 


    def _build_model_stack(self):
        print("Building model...")
        print(f"Backbone config: {self.backbone_cfg}")
        print(f"Decoder config: {self.decoder_cfg}")
        print(f"Head config: {self.head_cfg}")

        #pass configuration of the architecture to the model
                # SMP supporta output_stride = 8 o 16
        
        network_model = self.network_cfg.get("model", "deeplabv3plus")    #deeplabv3plus | fcn | unetplusplus

        if network_model == "deeplabv3plus":
            #choose between standard smp implementation or custom integrated one
            
            # if self.network_cfg["type"] == "smp":
            #     self.model = smp.DeepLabV3Plus(
            #         encoder_name=self.backbone_cfg["type"],
            #         encoder_weights="imagenet" if self.backbone_cfg["pretrained"] else None,
            #         in_channels=3,
            #         classes=3,  # assuming 3 classes for lane detection
            #         encoder_output_stride=self.backbone_cfg["output_stride"],
            #         upsampling=1,    #leave the output resolution to 1/4 of input
            #         activation=None,  # IMPORTANT: logits raw (CrossEntropy)
            #     )

            # else:
            #     #custom implementation of DeeplabV3Plus for lane detection
            #     self.model = DeepLabV3PlusLanes(
            #         encoder_name=self.backbone_cfg["type"],
            #         segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
            #         encoder_output_stride=self.backbone_cfg["output_stride"],
            #         aspp_dilations=self.decoder_cfg["aspp_dilations"],
            #         decoder_channels=self.decoder_cfg["deeplabv3plus_decoder_channels"],
            #         load_encoder=self.backbone_cfg["load_encoder"],
            #         load_decoder=self.decoder_cfg["load_decoder"],
            #         freeze_encoder=self.backbone_cfg["freeze_encoder"],
            #         freeze_decoder=self.decoder_cfg["freeze_decoder"],
            #         encoder_depth=self.backbone_cfg.get("encoder_depth", 5),
            #         bottleneck=self.bottleneck_cfg.get("type", "none"),
            #         encoder_partial_load=self.backbone_cfg.get("encoder_partial_load", False),
            #         encoder_partial_depth=self.backbone_cfg.get("encoder_partial_depth", 4),
            #         )
            

            self.model = DeepLabV3Plus(
                encoder_name=self.backbone_cfg["type"],
                segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
                encoder_output_stride=self.backbone_cfg.get("output_stride", 16),
                aspp_dilations=self.decoder_cfg.get("aspp_dilations", [12, 24, 36]),
                decoder_channels=self.decoder_cfg.get("deeplabv3plus_decoder_channels", 256),
                encoder_depth=self.backbone_cfg.get("encoder_depth", 5),
                encoder_partial_load=self.backbone_cfg.get("encoder_partial_load", False),
                encoder_partial_depth=self.backbone_cfg.get("encoder_partial_depth", 4),
                load_encoder=self.backbone_cfg.get("load_encoder", True),
                load_decoder=self.decoder_cfg.get("load_decoder", True),
                freeze_encoder=self.backbone_cfg.get("freeze_encoder", False),
                freeze_decoder=self.decoder_cfg.get("freeze_decoder", False),
                #head params
                head_upsampling=self.head_cfg.get("head_upsampling", 1),
                head_activation=self.head_cfg.get("head_activation", None),       #activation function after each convolutional layer in the regression head (except the last one)
                head_depth=self.head_cfg.get("head_depth", 1),               #how many convolutional layers to use in the regression head
                head_mid_channels=self.head_cfg.get("head_mid_channels", None),
                head_kernel_size=self.head_cfg.get("head_kernel_size", 1),
                output_channels=self.network_cfg.get("output_channels", 3),    #number of lane detection classes
                )
            print(f"[SMP] DeepLabV3Plus | encoder={self.backbone_cfg['type']} | "
                    f"pretrained={self.backbone_cfg['pretrained']} | os={self.backbone_cfg['output_stride']}")
            
        else:
            raise ValueError(f"Unsupported network model: {network_model}")
                
    
        #move model to device
        self.model.to(self.device)

        print("Building loss, optimizer, and scheduler...")

        #build the downsample factor for the gt (it is resized)
        #deeplabv3plus by default downsamples by 4x (output_stride=16, then upsampling=4)
        downsample_factor = int((4 / self.head_cfg.get("head_upsampling", 1)))

        print(f"Using downsample factor of {downsample_factor} for the GT in the loss function.")

        
        self.loss_fn= LanesLoss(downsample_factor=downsample_factor).to(self.device)
        
        self.optimizer = build_optimizer(self.optim_cfg, self.model.parameters())
        self.scheduler = build_scheduler(
            self.sched_cfg,
            self.optimizer,
            self.train_cfg,
            self.steps_per_epoch,
        )