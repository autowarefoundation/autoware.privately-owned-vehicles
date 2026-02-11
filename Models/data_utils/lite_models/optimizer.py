# utils/utils_optim.py
import torch
from torch.optim import AdamW, SGD
import math
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR


def build_optimizer(cfg_optim, model_params):
    opt_type = cfg_optim.get("type", "adamw").lower()
    lr = cfg_optim.get("lr", 1e-4)

    #str to float if lr is given as string
    if isinstance(lr, str):
        lr = float(lr)
        
    weight_decay = cfg_optim.get("weight_decay", 1e-4)

    if opt_type == "adamw":
        betas = tuple(cfg_optim.get("betas", [0.9, 0.999]))
        optimizer = AdamW(model_params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == "sgd":
        momentum = cfg_optim.get("momentum", 0.9)
        optimizer = SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
    return optimizer


def build_scheduler(cfg_sch, optimizer, train_cfg, steps_per_epoch):

    train_mode = train_cfg["mode"]      #"epoch" or "step"

    if train_mode == "epoch":
        training_epochs = train_cfg["max_epochs"]
        total_steps = training_epochs * steps_per_epoch
    elif train_mode == "steps":
        total_steps = train_cfg["max_steps"]
    else:
        raise ValueError(f"Unsupported training mode: {train_mode}")
    
    sch_type = cfg_sch.get("type", "none").lower()

    # --------------------------------------------------
    # NO SCHEDULER
    # --------------------------------------------------
    if sch_type == "none":
        return None

    # ==================================================
    # STEP DECAY (unchanged)
    # ==================================================
    elif sch_type == "step":
        step_size = cfg_sch.get("step_size", 30)
        gamma = cfg_sch.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ==================================================
    # PURE COSINE DECAY
    # ==================================================
    elif sch_type == "cosine":
        if total_steps is None:
            raise ValueError("cosine scheduler requires total_steps")
        min_lr = cfg_sch.get("min_lr", 1e-6)
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    # ==================================================
    # WARMUP + COSINE DECAY 
    # ==================================================
    elif sch_type == "warmup_cosine":
        warmup_steps = cfg_sch.get("warmup_steps", 3)
        min_lr = cfg_sch.get("min_lr", 1.0e-6)
        base_lr = optimizer.defaults["lr"]

        def lr_lambda(epoch):
            # ---- Warmup phase ------------------------------------
            if epoch < warmup_steps:
                return float(epoch) / float(max(1, warmup_steps))

            # ---- Cosine phase ------------------------------------
            progress = (epoch - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            # Scale so that lr >= min_lr
            return max(min_lr / base_lr, cosine_decay)

        return LambdaLR(optimizer, lr_lambda)

    # ==================================================
    # UNSUPPORTED
    # ==================================================
    else:
        raise ValueError(f"Unsupported scheduler type: {sch_type}")
