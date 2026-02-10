import segmentation_models_pytorch as smp

from utils.optimizer import build_optimizer, build_scheduler
from utils.loss import DepthLoss

from training.depth.trainer_base import DepthTrainerBase

from network.smp.DeepLabv3Plus import DeepLabV3Plus


class DeepLabV3PlusTrainer(DepthTrainerBase):
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

        print("[DeepLabV3PlusTrainer] Configuration : ", cfg)

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
        if "timm" not in self.backbone_cfg["type"]:
            self.backbone_cfg["type"] = "timm-" + self.backbone_cfg["type"].replace("_", "-")
        return 


    def _build_model_stack(self):
        print("Building model...")
        print(f"Backbone config: {self.backbone_cfg}")
        print(f"Decoder config: {self.decoder_cfg}")

        #pass configuration of the architecture to the model
                # SMP supporta output_stride = 8 o 16
        
        network_model = self.network_cfg.get("model", "deeplabv3plus")    #deeplabv3plus | fcn | unetplusplus

        if network_model == "deeplabv3plus":
            #choose between standard smp implementation or custom integrated one
            
            #custom implementation of DeeplabV3Plus for depth estimation
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
                head_upsampling=self.network_cfg.get("head_upsampling", 4),
                head_activation=self.network_cfg.get("head_activation", None),       #activation function after each convolutional layer in the regression head (except the last one)
                head_depth=self.network_cfg.get("head_depth", 1),               #how many convolutional layers to use in the regression head
                head_mid_channels=self.network_cfg.get("head_mid_channels", None),

                output_channels=self.network_cfg.get("output_channels", 1),   # for depth estimation, 1 channel depth map
            )
        
            
            print(f"[SMP] DeepLabV3Plus | encoder={self.backbone_cfg['type']} | "
                    f"pretrained={self.backbone_cfg['pretrained']} | os={self.decoder_cfg.get('output_stride', 16)}")
            
        else:
            raise ValueError(f"Unsupported network model: {network_model}")
                
    
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