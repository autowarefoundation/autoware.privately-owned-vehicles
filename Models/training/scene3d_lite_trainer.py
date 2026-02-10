
from Models.data_utils.lite_models.optimizer import build_optimizer, build_scheduler
from Models.data_utils.lite_models.loss import DepthLoss

from Models.training.lite_trainer_base import LiteTrainerBase


import os
import numpy as np
from tqdm import tqdm


from Models.data_utils.lite_models.depth import validate_depth


class Scene3DLiteTrainer(LiteTrainerBase):
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


        self.task = "DEPTH"
        
        # initialize base trainer with the configuration
        super().__init__(cfg)

        print("[Scene3DLiteTrainer] Configuration : ", cfg)

        #build output directories, for checkpints and logs (without overwriting)
        self._build_output_dirs()       #in trainer base

        #build wandb logger
        self._build_logger()            #in trainer base
        
        #build data loaders (train + val)
        self._build_datasets()          #in trainer base

        #choose the architecture to be used
        self._build_encoder_decoder()   #in trainer base

        #build model, loss, optimizer, scheduler
        self._build_model_stack()       #in trainer base

        #build training state (counters, best metrics, etc)
        self._build_training_state()    #in trainer base

        #resume in case the checkpoint path is specified
        self._maybe_resume()            #in trainer base


        self._build_loss()             #specific to this trainer since we have a custom loss and val function
    # -------------------------
    # Builders
    # -------------------------


    def _build_loss(self):
    
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
                

                # generate pseudo-labels on the fly if enabled
                if self.pseudo_labeling:
                    self._generate_pseudo_labels(batch)
                    
                loss_val = self._train_micro_step(batch)
                micro_step += 1

                # update happens only every grad_accum_steps micro-steps
                update_now = (micro_step % self.grad_accum_steps == 0)
                if update_now:
                    self._optimizer_step(loss_val)

                    # step-based validation
                    if self.val_mode == "steps" and (self.global_step % self.val_every_steps == 0):
                        # try:
                        self._run_validation_and_checkpoint()
                        # except Exception as e:
                        #     print(f"Validation failed at step {self.global_step}: {e}")
    
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
    # Validation + checkpoints
    # -------------------------
    def _run_validation_and_checkpoint(self):
        print(f"[samples={self.samples_seen}] Running validation...")

        all_totals = []

        for ds_name, loader in self.val_loaders.items():
            print(f"Validating on {ds_name}...")

            v_total, v_mAE, v_edge, v_absrel = validate_depth(
                model=self.model,
                dataloader=loader,
                loss_module=self.loss_fn_val,
                device=self.device,
                logger=self.wb,
                step=self.global_step,
                dataset_name=ds_name,
                pseudo_label_generator_model=self.pseudo_labeler if self.pseudo_labeling else None,
            )

            all_totals.append(v_total)
            print(f"  {ds_name}: total={v_total:.6f}, mAE={v_mAE:.6f}, edge={v_edge:.6f}, absrel={v_absrel:.6f}")

        mean_val = float(np.mean(all_totals)) if len(all_totals) else float("inf")

        if self.save_last:
            self._save_last()

        if self.save_best and mean_val < self.best_val_loss:
            self.best_val_loss = mean_val
            self._save_best()
            print(f"  New best_val_loss={mean_val:.6f}")


    def _train_micro_step(self, batch):
        images = batch["image"].to(self.device, non_blocking=True)
        depths = batch["gt"].to(self.device, non_blocking=True).float()

        if depths.ndim == 3:
            depths = depths.unsqueeze(1)

        #forward pass
        preds = self.model(images)

        
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)

        total_loss, mae_loss, edge_loss = self.loss_fn_train(preds, depths)

        loss_scaled = total_loss / self.grad_accum_steps
        
        loss_scaled.backward()

        return float(total_loss.item())