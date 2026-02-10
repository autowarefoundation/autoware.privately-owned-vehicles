# utils/logger_wandb.py

import wandb
import os

class WandBLogger:
    def __init__(self, run_name, config: dict, log_dir: str):
        """
        Initialize W&B logger.
        """
        prev_wandb_id = config.get("resume_run_id", None)

        name=run_name if prev_wandb_id is None else None
        
        self.run = wandb.init(
            project=config["experiment"]["wandb"]["project_name"],
            name=name,
            config=config,
            dir=log_dir,
            id=config.get("resume_run_id", None),  # <-- RUN ID passed externally
            resume="allow",                        # <-- KEY PART
            reinit=True,
        )

    # --------------------------
    #       TRAIN METRICS
    # --------------------------
    def log_train(self, step, loss, lr=None):
        data = {"train/loss": loss, "step": step}
        if lr is not None:
            data["train/lr"] = lr
        wandb.log(data, step=step)

    # --------------------------
    #   Validation for segmentation tasks
    # --------------------------
    def log_validation_segmentation(
        self,
        step,
        val_loss,
        mean_iou,
        class_iou,
        confmat,
        y_true,
        y_pred,
        class_names,
        vis_images=None,
        dataset=None      
    ):
        """
        If dataset is provided, metrics are logged under:
            val/{dataset}/...
        Otherwise default:
            val/...
        """

        # ---- Prefix builder ----
        prefix = f"val/{dataset}" if dataset is not None else "val"

        # # ---- Confusion matrix plot ----
        # cm_plot = wandb.plot.confusion_matrix(
        #     y_true=y_true,
        #     preds=y_pred,
        #     class_names=class_names
        # )

        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names.keys())]

        # # ---- Raw confusion matrix table ----
        # cm_rows = []
        # for i in range(confmat.shape[0]):
        #     row = [class_names[i]] + confmat[i].tolist()
        #     cm_rows.append(row)

        # cm_columns = ["class"] + class_names
        # cm_table = wandb.Table(data=cm_rows, columns=cm_columns)

        # ---- Log base metrics ----
        wandb.log({
            f"{prefix}/loss": val_loss,
            f"{prefix}/mean_iou": mean_iou,
            f"{prefix}/class_iou": class_iou,
            # f"{prefix}/confusion_matrix": cm_plot,        #remove the confusion matrix for now
            # f"{prefix}/confmat_raw": cm_table,
        }, step=step)

        # ---- Log per-class IoU ----
        for cname, score in class_iou.items():
            wandb.log({f"{prefix}/class_iou/{cname}": score}, step=step)

        # ---- Log validation visualizations ----
        if vis_images is not None:
            wandb_images = [
                wandb.Image(v, caption=f"{prefix}_vis_{i}")
                for i, v in enumerate(vis_images)
            ]
            wandb.log({f"{prefix}/visuals": wandb_images}, step=step)



    # --------------------------
    #   Validation for lane detection tasks
    # --------------------------
    def log_validation_lanes(
        self,
        step,
        dataset,
        val_loss,
        mean_iou,
        pixel_acc,
        class_iou,
        class_acc,
        vis_images,
    ):
        """
        Logs lane segmentation validation metrics to Weights & Biases.

        Metrics layout:
            val/<dataset>/loss
            val/<dataset>/mean_iou
            val/<dataset>/pixel_acc
            val/<dataset>/class_iou/<class_name>
            val/<dataset>/class_acc/<class_name>
            val/<dataset>/visuals
        """

        import wandb
        import numpy as np

        # ---- Prefix builder ----
        prefix = f"val/{dataset}" if dataset is not None else "val"

        # --------------------------------------------------
        # Scalar metrics
        # --------------------------------------------------
        log_dict = {
            f"{prefix}/loss": float(val_loss),
            f"{prefix}/mean_iou": float(mean_iou),
            f"{prefix}/pixel_acc": float(pixel_acc),
        }

        # Per-class IoU
        for cname, score in class_iou.items():
            log_dict[f"{prefix}/class_iou/{cname}"] = float(score)

        # Per-class Pixel Accuracy
        for cname, score in class_acc.items():
            log_dict[f"{prefix}/class_acc/{cname}"] = float(score)

        wandb.log(log_dict, step=step)

        # --------------------------------------------------
        # Visualization logging
        # --------------------------------------------------
        if vis_images is not None and len(vis_images) > 0:

            wandb_images = []
            for i, img in enumerate(vis_images):

                # Safety: ensure uint8 RGB
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8)

                wandb_images.append(
                    wandb.Image(img, caption=f"{prefix}_vis_{i}")
                )

            wandb.log(
                {f"{prefix}/visuals": wandb_images},
                step=step
            )



    # ==========================================================
    # VALIDATION — DEPTH
    # ==========================================================
    def log_validation_depth(
        self,
        step,
        val_loss,
        mAE,
        edge,
        absrel,
        vis_images=None,
        dataset=None,
    ):
        """
        Logs depth estimation validation metrics.

        Metrics:
            - total loss
            - mAE (SSI)
            - edge loss
            - absrel

        Visuals:
            - input | pred | gt | scale
        """

        prefix = f"val/{dataset}" if dataset is not None else "val"

        # ---- Scalar metrics ----
        wandb.log(
            {
                f"{prefix}/loss": val_loss,
                f"{prefix}/mAE": mAE,
                f"{prefix}/edge": edge,
                f"{prefix}/absrel": absrel,
            },
            step=step,
        )

        # ---- Visualizations ----
        if vis_images is not None:
            wandb_images = [
                wandb.Image(v, caption=f"{prefix}_depth_vis_{i}")
                for i, v in enumerate(vis_images)
            ]
            wandb.log(
                {f"{prefix}/visuals": wandb_images},
                step=step,
            )



    # --------------------------
    #       CHECKPOINTS
    # --------------------------
    def log_checkpoint(self, path, step):
        wandb.save(path)
        wandb.log({"checkpoint": path}, step=step)

    # --------------------------
    #       FINALIZE
    # --------------------------
    def finish(self):
        self.run.finish()
