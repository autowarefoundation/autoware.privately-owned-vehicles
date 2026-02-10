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
from utils.logger import WandBLogger
from utils.lanes import validate_lanes   


class LanesTrainerBase(ABC):
    """
    Base trainer for Ego-Lanes segmentation.

    - Train: ConcatDataset across multiple datasets
    - Val: one dataloader per validation set
    - Supports train_mode: "epoch" or "steps"
    - Supports val_mode: "epoch" or "steps"
    - Gradient accumulation
    - Checkpoint: last.pth + best_metric.pth
    """

    # -------------------------
    # Construction
    # -------------------------
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.exp_cfg   = cfg.get("experiment", {})
        self.dl_cfg    = cfg.get("dataloader", {})
        self.train_cfg = cfg.get("training", {})
        self.optim_cfg = cfg.get("optimizer", {})
        self.sched_cfg = cfg.get("scheduler", {})
        self.loss_cfg  = cfg.get("loss", {})
        self.ckpt_cfg  = cfg.get("checkpoint", {})

        self.backbone = None
        self.device = self._build_device()

    # -------------------------
    # Abstract API
    # -------------------------

    @abstractmethod
    def _build_model_stack(self):
        raise NotImplementedError

    # -------------------------
    # Builders
    # -------------------------
    def _build_device(self):
        device_str = self.exp_cfg.get("device", "cuda")
        return torch.device(device_str if torch.cuda.is_available() else "cpu")

    def _build_output_dirs(self):
        exp_name = self.exp_cfg.get("name", "egolanes")
        root_out = self.exp_cfg.get("output_dir", "runs/training/egolanes/")

        exp_name, _ = get_unique_experiment_dir(root_out, exp_name)

        self.exp_name = exp_name
        self.root_out = root_out
        self.out_dir  = os.path.join(root_out, exp_name)

        if exp_name == "val":
            print(f"Experiment name = {exp_name} : not saving the experiment in the training folder")
            return

        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        self.log_dir  = os.path.join(self.out_dir, "logs")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _build_datasets(self):
        dataset_cfg      = self.cfg["dataset"]
        augmentation_cfg = dataset_cfg.get("augmentations", {})

        self.training_sets   = dataset_cfg.get("training_sets", [])
        self.validation_sets = dataset_cfg.get("validation_sets", [])

        if len(self.training_sets) == 0:
            Warning("No training set specified. Maybe performing validation only.")

        # ---- TRAIN ----
        train_datasets = []
        for ds_name in self.training_sets:
            ds_yaml_key = f"{ds_name.lower()}_root"
            dataset_root = dataset_cfg[ds_yaml_key]

            dset = build_single_dataset(
                name=ds_name,
                dataset_root=dataset_root,
                aug_cfg=augmentation_cfg,
                mode="train",
                data_type="LANE_DETECTION",
            )
            train_datasets.append(dset)

        if len(train_datasets) > 0 :
            self.train_dataset = ConcatDataset(train_datasets)
            self.train_loader = build_dataloader(self.train_dataset, self.dl_cfg, mode="train")
        else:
            self.train_loader = list()  #empty list
            Warning("training list empty, not performing dataset concatenation")


        # ---- VAL ----
        self.val_loaders = {}
        for ds_name in self.validation_sets:
            ds_yaml_key = f"{ds_name.lower()}_root"
            dataset_root = dataset_cfg[ds_yaml_key]

            val_dset = build_single_dataset(
                name=ds_name,
                dataset_root=dataset_root,
                aug_cfg=augmentation_cfg,
                mode="val",
                data_type="LANE_DETECTION",
            )
            self.val_loaders[ds_name] = build_dataloader(val_dset, self.dl_cfg, mode="val")

        self.steps_per_epoch = len(self.train_loader)

    def _build_training_state(self):
        self.train_mode  = self.train_cfg.get("mode", "epoch")
        self.max_epochs  = self.train_cfg.get("max_epochs", 50)
        self.max_steps   = self.train_cfg.get("max_steps", None)

        self.grad_accum_steps = int(self.train_cfg.get("grad_accum_steps", 1))

        val_cfg = self.train_cfg.get("validation", {})
        self.val_mode          = val_cfg.get("mode", "epoch")
        self.val_every_epochs  = int(val_cfg.get("every_n_epochs", 1))
        self.val_every_steps   = int(val_cfg.get("every_n_steps", 5000))

        log_cfg = self.train_cfg.get("logging", {})
        self.log_every_steps = int(log_cfg.get("log_every_steps", 200))

        self.save_best = bool(self.train_cfg.get("save_best", True))
        self.save_last = bool(self.train_cfg.get("save_last", True))

        self.epoch = 0
        self.samples_seen = 0
        self.global_step  = 0
        self.best_metric  = -1.0

        self.batch_size       = int(self.dl_cfg["batch_size"])
        self.effective_batch  = self.batch_size * self.grad_accum_steps

    def _build_logger(self):
        self.wb = None
        wandb_cfg = self.exp_cfg.get("wandb", {})
        if bool(wandb_cfg.get("enabled", False)):
            self.wb = WandBLogger(
                run_name=self.exp_name,
                config=self.cfg,
                log_dir=self.log_dir,
            )

    # -------------------------
    # Resume
    # -------------------------
    def _maybe_resume(self):
        ckpt_path = self.ckpt_cfg.get("load_from", None)
        strict    = self.ckpt_cfg.get("strict_load", True)

        if not ckpt_path or not os.path.isfile(ckpt_path):
            print("No checkpoint loaded.")
            return

        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        #original awf pth files dont have model state, but directly the model
        try:
            self.model.load_state_dict(ckpt["model_state"], strict=strict)
        except:
            print("Loading original awf checkpoint format...")
            self.model.load_state_dict(ckpt, strict=strict)

            #it is for evaluation only, so return
            return

        self.optimizer.load_state_dict(ckpt["optimizer_state"])

        if self.scheduler and ckpt.get("scheduler_state"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.epoch       = ckpt.get("epoch", self.epoch)
        self.global_step = ckpt.get("step", self.global_step)

    # -------------------------
    # Public API
    # -------------------------
    def run(self):
        print("Starting Lane detection training...")
        stop_training = False

        while not stop_training:
            if self.train_mode == "epoch" and self.epoch >= self.max_epochs:
                break

            self.epoch += 1
            self.model.train()
            print(f"\n===== Epoch {self.epoch}/{self.max_epochs} =====")

            micro_step = 0
            pbar = tqdm(self.train_loader, ncols=120)

            for batch in pbar:
                if self.train_mode == "steps" and self.global_step >= self.max_steps:
                    stop_training = True
                    break

                loss_val = self._train_micro_step(batch)
                micro_step += 1

                update_now = (micro_step % self.grad_accum_steps == 0)
                if update_now:
                    self._optimizer_step(loss_val)

                    if self.val_mode == "steps" and (self.global_step % self.val_every_steps == 0):
                        self._run_validation_and_checkpoint()

                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "step": self.global_step,
                    "samples": self.samples_seen,
                })

            if self.val_mode == "epoch" and (self.epoch % self.val_every_epochs == 0):
                self._run_validation_and_checkpoint()

        if self.save_last:
            self._save_last()

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
        loss   = self.loss_fn(logits, masks)

        (loss / self.grad_accum_steps).backward()
        return float(loss.item())

    def _optimizer_step(self, loss_unscaled: float):
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step  += 1
        self.samples_seen += self.effective_batch

        if self.wb and (self.global_step % self.log_every_steps) == 0:
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
        mean_ious = []

        for ds_name, loader in self.val_loaders.items():
            metrics = validate_lanes(
                model=self.model,
                dataloader=loader,
                loss_fn=self.loss_fn,
                device=self.device,
                logger=self.wb,
                step=self.global_step,
                dataset_name=ds_name,
            )

            dataset_metrics[ds_name] = metrics
            mean_ious.append(metrics["mean_iou"])

            print(
                f"[VAL:{ds_name}] "
                f"loss={metrics['loss']:.4f} | "
                f"mIoU={metrics['mean_iou']:.4f} | "
                f"pixAcc={metrics['pixel_acc']:.4f}"
            )

        # --------------------------------------------------
        # Metric used for checkpoint selection
        # --------------------------------------------------
        mean_score = float(np.mean(mean_ious))
        print(f"[VAL] Mean mIoU across datasets = {mean_score:.4f}")

        # --------------------------------------------------
        # Checkpointing
        # --------------------------------------------------
        if self.save_last:
            self._save_last()

        if self.save_best and mean_score > self.best_metric:
            self.best_metric = mean_score
            self._save_best()
            print(f"New best model: mean mIoU = {self.best_metric:.4f}")


    def _checkpoint_payload(self):
        return {
            "epoch": self.epoch,
            "step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "wandb_run_id": self.wb.run.id if self.wb else None,
        }

    def _save_last(self):
        save_checkpoint(self._checkpoint_payload(), os.path.join(self.ckpt_dir, "last.pth"))

    def _save_best(self):
        save_checkpoint(self._checkpoint_payload(), os.path.join(self.ckpt_dir, "best_metric.pth"))
