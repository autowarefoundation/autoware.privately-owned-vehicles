
from Models.data_utils.lite_models.helpers.optimizer import build_optimizer, build_scheduler
from Models.data_utils.lite_models.helpers.loss import LanesLoss

from Models.training.lite_trainer_base import LiteTrainerBase
from tqdm import tqdm

from Models.data_utils.lite_models.helpers.lanes import validate_lanes   
import numpy as np

class EgoLanesLiteTrainer(LiteTrainerBase):
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

        self.task = "LANE_DETECTION"

        # initialize base trainer with the configuration
        super().__init__(cfg)

        print("[EgoLanesLiteTrainer] Configuration : ", cfg)

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

    def _build_loss(self):
    
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
