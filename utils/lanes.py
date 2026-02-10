# utils/utils_lanes.py

import numpy as np
import cv2
import torch
from tqdm import tqdm
from typing import Dict

import torch.nn.functional as F


# ============================================================
# Visualization utils (unchanged logic)
# ============================================================

LANE_COLORS_RGB = {
    "egoleft":  (0, 255, 255),   # cyan
    "egoright": (255, 0, 200),   # magenta-ish
    "other":    (0, 255, 145),   # green
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_image_chw_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().float().numpy()  # CHW
    img = img.transpose(1, 2, 0)  # HWC
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _apply_lane_colors_rgb(canvas: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    out = canvas.copy()

    if mask3.dtype != np.bool_:
        if mask3.max() > 1.5:
            mask3 = (mask3 > 127)
        else:
            mask3 = (mask3 > 0.5)

    # left
    ys, xs = np.where(mask3[..., 0])
    out[ys, xs, :] = LANE_COLORS_RGB["egoleft"]

    # right
    ys, xs = np.where(mask3[..., 1])
    out[ys, xs, :] = LANE_COLORS_RGB["egoright"]

    # other
    ys, xs = np.where(mask3[..., 2])
    out[ys, xs, :] = LANE_COLORS_RGB["other"]

    return out


def logits_to_lane_mask3(
    logits_chw: torch.Tensor,
    threshold: float = 0.0,
    use_sigmoid: bool = False,
) -> np.ndarray:
    """
    logits_chw: [3,H,W]
    return: HxWx3 bool
    """
    x = logits_chw.detach()
    if use_sigmoid:
        x = torch.sigmoid(x)
        mask = (x > threshold)
    else:
        mask = (x > threshold)

    return mask.permute(1, 2, 0).cpu().numpy().astype(bool)


def tensor_mask3_to_numpy(mask3_chw: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    GT: [3,H,W] -> HxWx3 bool
    """
    m = mask3_chw.detach().cpu().float()
    mask = (m > threshold)
    return mask.permute(1, 2, 0).numpy().astype(bool)


def make_lane_vis_pair(
    image_chw: torch.Tensor,
    pred_logits_chw: torch.Tensor,
    gt_mask3_chw: torch.Tensor,
    *,
    alpha: float = 0.5,
    pred_threshold: float = 0.0,
    pred_use_sigmoid: bool = False,
    out_size_hw=None,
):
    """
    Returns 2x2 tile:
        [ Pred NORMAL | GT NORMAL ]
        [ Pred RAW    | GT RAW    ]
    """

    base_img = denorm_image_chw_to_uint8(image_chw)
    H_img, W_img, _ = base_img.shape
    Hm, Wm = int(gt_mask3_chw.shape[1]), int(gt_mask3_chw.shape[2])

    if (H_img != Hm) or (W_img != Wm):
        base_img_small = cv2.resize(base_img, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    else:
        base_img_small = base_img

    pred_mask3 = logits_to_lane_mask3(
        pred_logits_chw, threshold=pred_threshold, use_sigmoid=pred_use_sigmoid
    )
    gt_mask3 = tensor_mask3_to_numpy(gt_mask3_chw)

    pred_colored = _apply_lane_colors_rgb(base_img_small, pred_mask3)
    gt_colored   = _apply_lane_colors_rgb(base_img_small, gt_mask3)

    pred_normal = cv2.addWeighted(pred_colored, alpha, base_img_small, 1 - alpha, 0)
    gt_normal   = cv2.addWeighted(gt_colored,   alpha, base_img_small, 1 - alpha, 0)

    black = np.zeros((Hm, Wm, 3), dtype=np.uint8)
    pred_raw = _apply_lane_colors_rgb(black, pred_mask3)
    gt_raw   = _apply_lane_colors_rgb(black, gt_mask3)

    top = np.concatenate([pred_normal, gt_normal], axis=1)
    bot = np.concatenate([pred_raw, gt_raw], axis=1)
    tile = np.concatenate([top, bot], axis=0)

    if out_size_hw is not None:
        out_h, out_w = out_size_hw
        tile = cv2.resize(tile, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    return tile


# ============================================================
# Metrics utils (binary per-channel)
# ============================================================

def update_binary_confmat(confmat, pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt: HxW bool
    confmat: dict with keys TP, FP, FN, TN
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    confmat["TP"] += int(tp)
    confmat["FP"] += int(fp)
    confmat["FN"] += int(fn)
    confmat["TN"] += int(tn)


def compute_iou_from_confmat(cm):
    denom = cm["TP"] + cm["FP"] + cm["FN"]
    if denom == 0:
        return np.nan
    return cm["TP"] / denom


def compute_pixel_acc_from_confmat(cm):
    denom = cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"]
    if denom == 0:
        return np.nan
    return (cm["TP"] + cm["TN"]) / denom


# ============================================================
# Validation loop
# ============================================================

def validate_lanes(
    model,
    dataloader,
    loss_fn,
    device,
    *,
    logger=None,
    step=None,
    dataset_name=None,
    vis_count: int = 25,
    alpha: float = 0.5,
    pred_threshold: float = 0.0,
    pred_use_sigmoid: bool = False,
):
    """
    Lane segmentation validation with:
      - loss
      - IoU per class (egoleft / egoright / other)
      - mean IoU
      - pixel accuracy
      - visualizations (raw + overlay)

    Returns:
        val_loss (float)
        mean_iou (float)
        pixel_acc (float)
        class_iou_dict (dict)
        vis_images (list[np.ndarray])
    """

    model.eval()
    total_loss = 0.0
    num_batches = 0

    # one confusion matrix per channel
    confmats = [
        {"TP": 0, "FP": 0, "FN": 0, "TN": 0},  # egoleft
        {"TP": 0, "FP": 0, "FN": 0, "TN": 0},  # egoright
        {"TP": 0, "FP": 0, "FN": 0, "TN": 0},  # other
    ]
    class_names = ["egoleft", "egoright", "other"]

    vis_images = []
    vis_done = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Validation Lanes]", leave=False):
            images = batch["image"].to(device, non_blocking=True)   # [B,3,H,W]
            gt_full     = batch["gt"].to(device, non_blocking=True)      # [B,3,H',W']

            logits = model(images)                                  # [B,3,H',W']
            loss = loss_fn(logits, gt_full)


            # -------------------------------------------------
            # Downsample GT for metrics + visualization (EgoLanes behavior)
            # -------------------------------------------------
            gt = gt_full.float()

            # downsample with maxpooling the gt n case the downsample factor is different than 1
            for _ in range(loss_fn.downsample_factor // 2):
                gt = F.max_pool2d(gt, kernel_size=2, stride=2)

            # Binarize
            gt = (gt > 0.5)


            total_loss += float(loss.item())
            num_batches += 1

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            preds_bin = (logits > pred_threshold)   # [B,3,H',W'] bool
            gt_bin    = (gt > 0.5)

            preds_np = preds_bin.cpu().numpy()
            gt_np    = gt_bin.cpu().numpy()

            B = preds_np.shape[0]
            for b in range(B):
                for c in range(3):
                    update_binary_confmat(
                        confmats[c],
                        pred=preds_np[b, c],
                        gt=gt_np[b, c],
                    )

            # -------------------------------------------------
            # Visuals
            # -------------------------------------------------
            if vis_done < vis_count:
                for b in range(images.shape[0]):
                    if vis_done >= vis_count:
                        break

                    tile = make_lane_vis_pair(
                        image_chw=images[b],
                        pred_logits_chw=logits[b],
                        gt_mask3_chw=gt[b],
                        alpha=alpha,
                        pred_threshold=pred_threshold,
                        pred_use_sigmoid=pred_use_sigmoid,
                    )
                    vis_images.append(tile)
                    vis_done += 1

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------
    val_loss = total_loss / max(1, num_batches)

    class_iou = {}
    class_acc = {}

    for name, cm in zip(class_names, confmats):
        class_iou[name] = float(compute_iou_from_confmat(cm))
        class_acc[name] = float(compute_pixel_acc_from_confmat(cm))

    mean_iou = float(np.nanmean(list(class_iou.values())))
    pixel_acc = float(np.nanmean(list(class_acc.values())))


    # combine the metrics results into a dict to return 
    results = {
            "loss": val_loss,
            "mean_iou": mean_iou,
            "pixel_acc": pixel_acc,
            "class_iou": class_iou,
            "class_acc": class_acc,
        }
    

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
    if logger is not None and hasattr(logger, "log_validation_lanes"):
        logger.log_validation_lanes(
            step=step,
            dataset=dataset_name,
            val_loss=val_loss,
            mean_iou=mean_iou,
            pixel_acc=pixel_acc,
            class_iou=class_iou,
            class_acc=class_acc,
            vis_images=vis_images,
        )

        return results
    
    #if logger is None, return the results + visualization images (for the evaluation script)
    return results, vis_images
    
