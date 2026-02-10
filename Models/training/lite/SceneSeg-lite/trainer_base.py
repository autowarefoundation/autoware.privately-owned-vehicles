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
from utils.segmentation import validate_segmentation
from utils.logger import WandBLogger


class SegmentationTrainerBase(ABC):
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
        self.cfg = cfg

        self.is_training = True

        self.exp_cfg = cfg.get("experiment", {})
        self.dl_cfg = cfg.get("dataloader", {})
        self.train_cfg = cfg.get("training", {})
        self.optim_cfg = cfg.get("optimizer", {})
        self.sched_cfg = cfg.get("scheduler", {})
        self.loss_cfg = cfg.get("loss", {})
        self.ckpt_cfg = cfg.get("checkpoint", {})

        #choose the architecture to be used
        self.backbone = None

        #choose between cpu and cuda if available
        self.device = self._build_device()


    @abstractmethod
    def _build_model_stack(self):
        """Build model, loss, optimizer, scheduler."""
        raise NotImplementedError
    
    # -------------------------
    # Builders
    # -------------------------

    def _build_device(self):
        device_str = self.exp_cfg.get("device", "cuda")
        return torch.device(device_str if torch.cuda.is_available() else "cpu")

    def _build_device(self):
        device_str = self.exp_cfg.get("device", "cuda")
        return torch.device(device_str if torch.cuda.is_available() else "cpu")

    def _build_output_dirs(self):
        exp_name = self.exp_cfg.get("name", "scene3d")
        root_out = self.exp_cfg.get("output_dir", "runs/training/depth/")

        # avoid overwriting
        self.exp_name, _ = get_unique_experiment_dir(root_out, exp_name)
        self.exp_name = exp_name
        self.root_out = root_out
        self.out_dir = os.path.join(root_out, exp_name)

        if self.is_training is False:
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
                ds_name, dataset_root, aug_cfg=augmentation_cfg, mode="train", data_type="SEGMENTATION")
            
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
            val_dset = build_single_dataset(name=ds_name, dataset_root=dataset_root, mode="val", data_type="SEGMENTATION", aug_cfg=augmentation_cfg)
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
        self.best_miou = -1.0

        self.batch_size = int(self.dl_cfg["batch_size"])
        self.effective_batch = self.batch_size * self.grad_accum_steps

    def _build_logger(self):
        self.wb = None
        wandb_cfg = self.exp_cfg.get("wandb", {})
        enabled = bool(wandb_cfg.get("enabled", False))

        if enabled:
            self.wb = WandBLogger(run_name=self.exp_name, config=self.cfg, log_dir=self.log_dir)

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

            if "best_mIoU" in ckpt:
                self.best_miou = float(ckpt["best_mIoU"])

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
            self.best_miou = -1.0

            # Important: make sure optimizer starts from config LR
            for group in self.optimizer.param_groups:
                group["lr"] = self.optim_cfg.get("lr", group["lr"])

            # Scheduler will naturally restart from step 0


    # -------------------------
    # Public API
    # -------------------------
    def run(self):
        print("Starting training...")
        stop_training = False

        while not stop_training:
            # stop by epoch
            if self.train_mode == "epoch" and self.epoch >= self.max_epochs:
                break

            # start new epoch
            self.epoch += 1
            self.model.train()

            if self.train_mode == "epoch":
                print(f"\n===== Epoch {self.epoch}/{self.max_epochs} =====")
            else:
                print(f"\n===== Step {self.global_step}/{self.max_steps} =====")

            micro_step = 0  # batch counter inside epoch (for grad-accum)
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", ncols=120)

            for batch in pbar:
                # stop by global steps (optimizer updates)
                if self.train_mode == "steps" and self.global_step >= self.max_steps:
                    stop_training = True
                    break

                loss_val = self._train_micro_step(batch)
                micro_step += 1

                # update happens only every grad_accum_steps micro-steps
                update_now = (micro_step % self.grad_accum_steps == 0)
                if update_now:
                    self._optimizer_step(loss_val)

                    # step-based validation
                    if self.val_mode == "steps" and (self.global_step % self.val_every_steps == 0):
                        self._run_validation_and_checkpoint()

                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "step": f"{self.global_step}/{self.max_steps}" if self.train_mode == "steps" else f"{self.global_step}",
                    "samples": self.samples_seen
                })

            # epoch-based validation
            if self.val_mode == "epoch" and (self.epoch % self.val_every_epochs == 0):
                self._run_validation_and_checkpoint()

        # final save (optional; you can keep or remove since you save on validation)
        if self.save_last:
            self._save_last()
            print(f"Saved last checkpoint to {os.path.join(self.ckpt_dir, 'last.pth')}")

        if self.wb:
            self.wb.finish()

        print("Training complete.")

    # -------------------------
    # Train internals
    # -------------------------
    def _train_micro_step(self, batch) -> float:
        images = batch["image"].to(self.device, non_blocking=True)
        masks  = batch["gt"].to(self.device, non_blocking=True)

        logits = self.model(images)
        loss = self.loss_fn(logits, masks)
        loss_scaled = loss / self.grad_accum_steps
        loss_scaled.backward()

        return float(loss.item())

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

    # -------------------------
    # Validation + checkpoints
    # -------------------------
    def _run_validation_and_checkpoint(self):
        print(f"[samples={self.samples_seen}] Running validation...")
        dataset_metrics = {}

        for ds_name, val_loader in self.val_loaders.items():
            print(f"Validating on {ds_name}...")
            v_loss, v_miou, v_class_iou = validate_segmentation(
                model=self.model,
                dataloader=val_loader,
                loss_fn=self.loss_fn,
                device=self.device,
                loss_cfg=self.loss_cfg,
                logger=self.wb,
                step=self.global_step,
                dataset_name=ds_name,
            )
            dataset_metrics[ds_name] = {
                "loss": v_loss,
                "miou": v_miou,
                "class_iou": v_class_iou,
            }

        val_miou = float(np.mean([m["miou"] for m in dataset_metrics.values()]))

        # Save LAST every validation (your new policy)
        if self.save_last:
            self._save_last()
            print(f"Saved last checkpoint to {os.path.join(self.ckpt_dir, 'last.pth')}")

        # Save BEST on improvement
        if self.save_best and val_miou > self.best_miou:
            self.best_miou = val_miou
            self._save_best()
            print(f"New best mIoU={self.best_miou:.4f}, saved to {os.path.join(self.ckpt_dir, 'best_mIoU.pth')}")

    def _checkpoint_payload(self):
        return {
            "epoch": self.epoch,
            "step": self.global_step,  # keep your convention: samples as “step”
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_mIoU": self.best_miou,
            "wandb_run_id": self.wb.run.id if self.wb else None,
        }

    def _save_last(self):
        path = os.path.join(self.ckpt_dir, "last.pth")
        save_checkpoint(self._checkpoint_payload(), path)

    def _save_best(self):
        path = os.path.join(self.ckpt_dir, "best_mIoU.pth")
        save_checkpoint(self._checkpoint_payload(), path)

    def set_val_mode(self):
        """used for evaluation only, called externally"""
        self.is_training = False