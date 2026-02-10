import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from abc import ABC, abstractmethod

from utils.training import (
    build_single_dataset,
    build_dataloader,
    save_checkpoint,
    get_unique_experiment_dir,
)
from utils.depth import validate_depth, denormalize_image, center_crop_vit_safe_lower, pad_to_target_center
from utils.logger import WandBLogger

import cv2

from network.smp.DeepLabv3Plus import DeepLabV3Plus
from network.smp.UnetPlusPlus import UnetPlusPlus


class LiteTrainerBase(ABC):
    """
    Trainer for monocular relative depth.

    Features:
      - Multi-dataset training via ConcatDataset
      - Per-dataset validation loaders
      - SSI + multi-scale edge loss (DepthLoss)
      - grad accumulation (correct micro-step handling)
      - step-based or epoch-based training
      - validation-based checkpointing:
          * last.pth saved at every validation
          * best_val_loss.pth saved on improvement
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.exp_cfg = cfg.get("experiment", {})
        self.dl_cfg = cfg.get("dataloader", {})
        self.train_cfg = cfg.get("training", {})
        self.optim_cfg = cfg.get("optimizer", {})
        self.sched_cfg = cfg.get("scheduler", {})
        self.loss_cfg = cfg.get("loss", {})
        self.ckpt_cfg = cfg.get("checkpoint", {})

        #choose the architecture to be used
        self.backbone = None

        self.pseudo_labeler = None
        self.pseudo_labeler_type = self.train_cfg.get("pseudo_labeler_generator", "vitl")  #default vit large (can choose between s, b, l)
        self.pseudo_labeling = False  #will be set to true if enabled in the dataset config

        #choose between cpu and cuda if available
        self.device = self._build_device()


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
                head_upsampling=self.head_cfg.get("head_upsampling", 1),
                head_activation=self.head_cfg.get("head_activation", None),
                head_depth=self.head_cfg.get("head_depth", 1),
                head_mid_channels=self.head_cfg.get("head_mid_channels", None),

                output_channels=self.network_cfg.get("output_channels", 1),   # for depth estimation, 1 channel depth map
            )
        
            
            print(f"[SMP] DeepLabV3Plus | encoder={self.backbone_cfg['type']} | "
                    f"pretrained={self.backbone_cfg['pretrained']} | os={self.decoder_cfg.get('output_stride', 16)}")
            

        elif network_model == "unetplusplus":

            self.model = UnetPlusPlus(
                encoder_name=self.backbone_cfg["type"],
                segmentation_ckpt=self.network_cfg.get("pretrained_model_path", None),
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
        
            
        else:
            raise ValueError(f"Unsupported network model: {network_model}")
        
        #move model to device
        self.model.to(self.device)
    
    # -------------------------
    # Builders
    # -------------------------

    def _build_device(self):
        device_str = self.exp_cfg.get("device", "cuda")
        return torch.device(device_str if torch.cuda.is_available() else "cpu")

    def _build_output_dirs(self):
        exp_name = self.exp_cfg.get("name", f"exp_{self.task}")
        root_out = self.exp_cfg.get("output_dir", f"runs/training/{self.task}/")

        # avoid overwriting
        exp_name, _ = get_unique_experiment_dir(root_out, exp_name)

        self.exp_name = exp_name
        self.root_out = root_out
        self.out_dir = os.path.join(root_out, exp_name)

        if exp_name == "val":
            print(f"Experiment name = {exp_name} : not saving the experiment in the training folder")
            return
        
        os.makedirs(self.out_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        self.log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _build_datasets(self):
        dataset_cfg = self.cfg["dataset"]
        augmentation_cfg = dataset_cfg.get("augmentations", {})
        print("Dataset augmentations config:", augmentation_cfg)

        self.training_sets = dataset_cfg.get("training_sets", [])
        self.validation_sets = dataset_cfg.get("validation_sets", [])

        if len(self.training_sets) == 0:
            Warning("No training set specified. Maybe performing validation only.")

        #depth specific : build pseudo-labeler if enabled
        if self.task == "DEPTH" :
            #retrive the pseudo labeling flag from the training config
            self.pseudo_labeling = dataset_cfg.get("pseudo_labeling", False)

            if self.pseudo_labeling:
                print("Pseudo-labeling is ENABLED.")
                #isntantiate the pseudo-labeling model
                self._build_pseudo_labeler()

                
        print("Training on:", self.training_sets)
        print("Validating on:", self.validation_sets)


        # ---- TRAIN: concat ----
        train_datasets = []
        for ds_name in self.training_sets:
            ds_yaml_key = f"{ds_name.lower()}_root"
            if ds_yaml_key not in dataset_cfg:
                raise ValueError(f"Missing root for dataset '{ds_name}' -> key '{ds_yaml_key}'")

            dataset_root = dataset_cfg[ds_yaml_key]
            dset = build_single_dataset(
                ds_name, dataset_root, aug_cfg=augmentation_cfg, mode="train", data_type=self.task, pseudo_labeling=self.pseudo_labeling)
            
            train_datasets.append(dset)

        if len(train_datasets) > 0 :
            self.train_dataset = ConcatDataset(train_datasets)
            self.train_loader = build_dataloader(self.train_dataset, self.dl_cfg, mode="train")
        else:
            self.train_loader = list()  #empty list
            Warning("training list empty, not performing dataset concatenation")

        # ---- VAL: one loader per dataset ----
        self.val_loaders = {}
        print("Preparing validation loaders...")
        for ds_name in self.validation_sets:
            ds_yaml_key = f"{ds_name.lower()}_root"
            if ds_yaml_key not in dataset_cfg:
                raise ValueError(f"Missing path for dataset '{ds_name}' -> key '{ds_yaml_key}'")

            dataset_root = dataset_cfg[ds_yaml_key]
            val_dset = build_single_dataset(name=ds_name, dataset_root=dataset_root, mode="val", data_type=self.task, aug_cfg=augmentation_cfg, pseudo_labeling=self.pseudo_labeling)
            self.val_loaders[ds_name] = build_dataloader(val_dset, self.dl_cfg, mode="val")

        self.steps_per_epoch = len(self.train_loader)


    def _build_training_state(self):
        # training controls
        self.train_mode = self.train_cfg.get("mode", "epoch")  # "epoch" or "steps"
        self.max_epochs = self.train_cfg.get("max_epochs", 10)
        self.max_steps = self.train_cfg.get("max_steps", None)

        self.grad_accum_steps = int(self.train_cfg.get("grad_accum_steps", 1))
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        # validation controls
        val_cfg = self.train_cfg.get("validation", {})
        self.val_mode = val_cfg.get("mode", "epoch")  # "epoch" or "steps"
        self.val_every_epochs = int(val_cfg.get("every_n_epochs", 1))
        self.val_every_steps = int(val_cfg.get("every_n_steps", 8000))

        # logging
        log_cfg = self.train_cfg.get("logging", {})
        self.log_every_steps = int(log_cfg.get("log_every_steps", 250))

        # checkpoint policy
        self.save_best = bool(self.train_cfg.get("save_best", True))
        self.save_last = bool(self.train_cfg.get("save_last", True))

        # counters/state
        self.epoch = 0
        self.samples_seen = 0  # you use samples_seen as the “step” in W&B
        self.global_step = 0   # optimizer updates count
        self.best_val_loss = float("inf")

        self.batch_size = int(self.dl_cfg["batch_size"])
        self.effective_batch = self.batch_size * self.grad_accum_steps

    def _build_logger(self):
        self.wb = None
        wandb_cfg = self.exp_cfg.get("wandb", {})
        enabled = bool(wandb_cfg.get("enabled", False))

        if enabled:
            self.wb = WandBLogger(run_name=self.exp_name, config=self.cfg, log_dir=self.log_dir)

    def _build_pseudo_labeler(self):
        """Build the pseudo-labeling model (DepthAnythingV2 large)."""
        from AEI.preprocessing.scene3d.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

        # Model configuration - we use vitl for pseudo labels
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        #use large vit model for pseudo-labeling (for now)
        encoder = self.pseudo_labeler_type.lower()
        if encoder not in model_configs:
            raise ValueError(f"Invalid pseudo-labeler encoder type '{encoder}'. Choose from 'vits', 'vitb', 'vitl', 'vitg'.")
        

        #insert path of depth aything v2
        checkpoint = f"AEI/preprocessing/scene3d/depth_anything_v2/depth_anything_v2_{encoder}.pth"

        depth_anything = DepthAnythingV2(**model_configs[encoder])

        depth_anything.load_state_dict(torch.load(checkpoint, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        
        depth_anything = depth_anything.to("cuda").eval()

        self.pseudo_labeler = depth_anything

        print(f"Pseudo-labeling enabled: using DepthAnythingV2 {encoder} model.")

    # -------------------------
    # Resume
    # -------------------------
    def _maybe_resume(self):
        ckpt_path = self.ckpt_cfg.get("load_from", None)
        strict_load = self.ckpt_cfg.get("strict_load", True)
        fine_tune = bool(self.ckpt_cfg.get("fine_tune", False))

        if not ckpt_path or not os.path.isfile(ckpt_path):
            print("No pretrained checkpoint loaded.")
            return

        print(f"Loading pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # ---- Always load model weights ----
        missing, unexpected = self.model.load_state_dict(
            ckpt["model_state"], strict=strict_load
        )

        if not strict_load:
            print("  [INFO] Non-strict load enabled")
            print("  Missing keys:", missing)
            print("  Unexpected keys:", unexpected)

        # ============================================================
        # Resume mode (true resume of training)
        # ============================================================
        if not fine_tune:
            print("  [INFO] Resuming optimizer / scheduler / counters")

            if "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])

            if "scheduler_state" in ckpt and self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])

            if "best" in ckpt:
                self.best = float(ckpt["best"])

            if "epoch" in ckpt:
                self.epoch = int(ckpt["epoch"])

            if "step" in ckpt:
                self.global_step = int(ckpt["step"])

        # ============================================================
        # Fine-tuning mode (weights only)
        # ============================================================
        else:
            print("  [INFO] Fine-tuning mode: resetting optimizer, scheduler and counters")

            self.epoch = 0
            self.global_step = 0
            self.samples_seen = 0

            # Important: make sure optimizer starts from config LR
            for group in self.optimizer.param_groups:
                group["lr"] = self.optim_cfg.get("lr", group["lr"])

            # Scheduler will naturally restart from step 0

    def _build_encoder_decoder(self):

        self.network_cfg = self.cfg["network"]

        self.backbone_cfg = self.network_cfg.get("backbone", {})
        self.decoder_cfg = self.network_cfg.get("decoder", {})
        self.head_cfg = self.network_cfg.get("head", {})
        
        #modify the name of the backbone type from efficientnet_b0 to timm-efficientnet-b0
        if "timm" not in self.backbone_cfg["type"]:
            self.backbone_cfg["type"] = "timm-" + self.backbone_cfg["type"].replace("_", "-")
        return 

    # -------------------------
    # Train internals
    # -------------------------

    def _optimizer_step(self, loss_unscaled: float):
        self.optimizer.step()
        self.optimizer.zero_grad()

        # scheduler step (only if step-based)
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        self.samples_seen += self.effective_batch

        # logging
        if (self.global_step % self.log_every_steps) == 0 and self.wb:
            self.wb.log_train(
                step=self.global_step,
                loss=loss_unscaled,
                lr=self.optimizer.param_groups[0]["lr"],
            )


    def _checkpoint_payload(self):
        return {
            "epoch": self.epoch,
            "step": self.global_step,  # keep your convention: samples as “step”
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best": self.best_val_loss,
            "wandb_run_id": self.wb.run.id if self.wb else None,
        }

    def _save_last(self):
        path = os.path.join(self.ckpt_dir, "last.pth")
        save_checkpoint(self._checkpoint_payload(), path)

    def _save_best(self):
        path = os.path.join(self.ckpt_dir, "best.pth")
        save_checkpoint(self._checkpoint_payload(), path)



    def _generate_pseudo_labels(self, batch):
        #image is CHW uint8, normalized
        #batch contains "images" and "gt" tensors. in pseudo-labeling mode, "gt" is a dummy tensor
        #in pseudo labeling, images have shape = B x 3 x H_crop x W_crop, NORMALIZED and NOISE ADDED.
        #so we need to generate pseudo-labels on the fly here, with :
        #   1)denormalizing the images
        #   2)center cropping to vit-safe resolution
        #   3)zero-padding the image to the vit input size (so the valid mask is applied correctly inside the pseudo-labeler)

        images = batch["image"]

        B, _, H_tgt, W_tgt = images.shape

        # generate the pseudo-labels if the flag is enabled
        if self.pseudo_labeling:

            pseudo_depths = []

            for i in range(B):

                #1) denormalize the images
                # ------------------------------------------
                # 1) Recover RAW image (HWC uint8)
                # ------------------------------------------
                raw_img = denormalize_image(images[i])

                # ------------------------------------------
                # 2) Center-crop RAW → vit-safe (lower bound)
                # ------------------------------------------
                raw_vit, (y0, x0) = center_crop_vit_safe_lower(raw_img,patch=14)

                H_vit, W_vit = raw_vit.shape[:2]

            
                # ---- teacher inference (vit space) ----
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda",dtype=torch.float16):

                        depth_vit = self.pseudo_labeler.infer_image(raw_vit)


                # ---- pad depth to student size (CENTERED) ----
                depth_vit_pad, (pad_top, pad_left) = pad_to_target_center(
                    depth_vit,
                    target_h=H_tgt,
                    target_w=W_tgt,
                    value=0.0,
                )

                # ---- build valid mask (centered) ----
                valid_mask = np.zeros((H_tgt, W_tgt), dtype=np.float32)

                valid_mask[pad_top : pad_top + H_vit, pad_left: pad_left + W_vit,] = 1.0

                # ---- apply mask to depth ----
                depth_vit_pad *= valid_mask

                # ------------------------------------------
                # 5) Apply SAME mask to student image
                # ------------------------------------------
                valid_mask_t = torch.from_numpy(valid_mask).to(images.device)
                valid_mask_t = valid_mask_t.unsqueeze(0)  # 1xHxW
                images[i] = images[i] * valid_mask_t

                pseudo_depths.append(depth_vit_pad)

            # ------------------------------------------
            # Stack pseudo depths → tensor
            # ------------------------------------------
            batch["gt"] = torch.from_numpy(
                np.stack(pseudo_depths, axis=0)
            ).unsqueeze(1).to(self.device)   # B x 1 x H x W
