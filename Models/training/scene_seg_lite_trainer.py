import segmentation_models_pytorch as smp

from Models.data_utils.lite_models.optimizer import build_optimizer, build_scheduler
from Models.data_utils.lite_models.loss import SegmentationLoss

from Models.training.lite_trainer_base import LiteTrainerBase

from tqdm import tqdm
import os

from Models.data_utils.lite_models.segmentation import validate_segmentation

import numpy as np

class SceneSegLiteTrainer(LiteTrainerBase):
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
        
        self.task = "SEGMENTATION"

        # initialize base trainer with the configuration
        super().__init__(cfg)


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

        self._build_loss()             #specific to this trainer since we have a custom loss and val function
                

    def _build_loss(self):
    
        print("Building loss, optimizer, and scheduler...")

        self.loss_fn = SegmentationLoss(self.loss_cfg).to(self.device)
        
        self.optimizer = build_optimizer(self.optim_cfg, self.model.parameters())
        self.scheduler = build_scheduler(self.sched_cfg, self.optimizer, self.train_cfg, self.steps_per_epoch)



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
        if self.save_best and val_miou > self.best:
            self.best = val_miou
            self._save_best()
            print(f"New best mIoU={self.best:.4f}, saved to {os.path.join(self.ckpt_dir, 'best.pth')}")
