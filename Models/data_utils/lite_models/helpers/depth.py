import torch.nn as nn
import torch
import numpy as np
import cv2
from tqdm import tqdm
import os
from pathlib import Path


# ============================================================
# COLORMAP UTILS
# ============================================================

def apply_colormap(depth, colormap="turbo"):
    d_min = float(depth.min())
    d_max = float(depth.max())

    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_u8 = (depth_norm * 255).astype(np.uint8)

    cmap = {
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
    }[colormap]

    color = cv2.applyColorMap(depth_u8, cmap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    return color, d_min, d_max


def visualize_depth_triplet(image_t, pred_t, gt_t, colormap="turbo"):
    """
    image: tensor (3,H,W)
    pred:  tensor (1,H,W)
    gt:    tensor (1,H,W)
    """

    # --------------------------------------------------
    # Undo normalization (image)
    # --------------------------------------------------
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = image_t.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # --------------------------------------------------
    # Depth maps
    # --------------------------------------------------
    pred = pred_t.squeeze().cpu().numpy()
    gt   = gt_t.squeeze().cpu().numpy()

    # Metrics (per-sample)
    eps = 1e-6
    mae = np.mean(np.abs(pred - gt))
    absrel = np.mean(np.abs(pred - gt) / (gt + eps))

    pred_col, dmin_p, dmax_p = apply_colormap(pred, colormap)
    gt_col,   dmin_g, dmax_g = apply_colormap(gt, colormap)

    dmin = min(dmin_p, dmin_g)
    dmax = max(dmax_p, dmax_g)

    # scale = draw_depth_scale(img.shape[0], dmin, dmax, colormap)

    # # --------------------------------------------------
    # # Horizontal spacing before scale
    # # --------------------------------------------------
    # spacer = np.ones((img.shape[0], 20, 3), dtype=np.uint8) * 255

    # --------------------------------------------------
    # Stack main row
    # --------------------------------------------------
    top = np.concatenate([img, pred_col, gt_col], axis=1)

    # --------------------------------------------------
    # Bottom text bar
    # --------------------------------------------------
    H, W, _ = top.shape
    text_bar = np.ones((40, W, 3), dtype=np.uint8) * 255

    text = f"MAE: {mae:.4f}    AbsRel: {absrel:.4f}"

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x = (W - text_w) // 2
    y = (text_bar.shape[0] + text_h) // 2

    cv2.putText(
        text_bar,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    # --------------------------------------------------
    # Final image
    # --------------------------------------------------
    combined = np.concatenate([top, text_bar], axis=0)
    return combined



# ============================================================
# VALIDATION LOOP
# ============================================================

def validate_depth(
    model: nn.Module,
    dataloader,
    loss_module,
    device,
    logger=None,
    step=None,
    dataset_name=None,
    vis_count=10,
    colormap="turbo",
    pseudo_label_generator_model: nn.Module | None = None,
):
    """
    Depth validation loop.

    Supports:
    - GT depth (standard validation)
    - Pseudo-label depth (teacher ViT)
    - Caching of pseudo-labels (depth + valid_mask)
    - Geometry consistency via center-crop + centered padding
    - Zeroing padded pixels in student input
    """

    # --------------------------------------------------
    # Cache setup
    # --------------------------------------------------
    cache_root = Path("cache/pseudo_depth")
    if dataset_name is None:
        dataset_name = "unknown_dataset"

    model.eval()

    sums = dict(
        total=0.0,
        mae=0.0,
        edge=0.0,
        abs_rel=0.0,
        sq_rel=0.0,
        rmse=0.0,
        rmse_log=0.0,
        a1=0.0,
        a2=0.0,
        a3=0.0,
    )
    n_batches = 0

    vis_images = []

    # --------------------------------------------------
    # Main validation loop
    # --------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc=f"[Depth Validation] {dataset_name}",
            leave=False,
        ):

            # --------------------------------------------------
            # Student images (already OS=16 padded by dataloader)
            # --------------------------------------------------
            images = batch["image"].to(device, non_blocking=True)
            B, _, H_tgt, W_tgt = images.shape

            # ==================================================
            # PSEUDO-LABEL BRANCH
            # ==================================================
            if pseudo_label_generator_model is not None:

                pseudo_depths = []

                for i in range(B):

                    # ------------------------------------------
                    # 1) Recover RAW image (HWC uint8)
                    # ------------------------------------------
                    raw_img = denormalize_image(images[i])

                    # ------------------------------------------
                    # 2) Center-crop RAW → vit-safe (lower bound)
                    # ------------------------------------------
                    raw_vit, (y0, x0) = center_crop_vit_safe_lower(
                        raw_img,
                        patch=14,
                    )
                    H_vit, W_vit = raw_vit.shape[:2]

                    # ------------------------------------------
                    # 3) Cache path (dataset / vit / resolution)
                    # ------------------------------------------
                    cache_dir = (
                        cache_root
                        / dataset_name
                        / "vit14"
                        / f"H{H_tgt}_W{W_tgt}"
                    )
                    cache_dir.mkdir(parents=True, exist_ok=True)

                    # sample index (prefer dataset-provided index)
                    sample_idx = n_batches * B + i

                    cache_path = cache_dir / f"sample_{sample_idx:06d}.npz"

                    # ------------------------------------------
                    # 4) Load or compute pseudo-label (SAFE)
                    # ------------------------------------------
                    if not cache_path.exists():
                        compute_pseudo_depth_with_caching(
                            raw_vit,
                            pseudo_label_generator_model,
                            H_tgt,
                            W_tgt,
                            H_vit,
                            W_vit,
                            cache_path,
                        )

                    # try loading (also after recompute)
                    try:
                        cached = np.load(cache_path)
                        depth_vit_pad = cached["depth"]
                        valid_mask    = cached["valid_mask"]
                    except Exception as e:
                        print(f"[ERROR] Cache broken, recomputing: {cache_path} ({e})")
                        compute_pseudo_depth_with_caching(
                            raw_vit,
                            pseudo_label_generator_model,
                            H_tgt,
                            W_tgt,
                            H_vit,
                            W_vit,
                            cache_path,
                        )
                        cached = np.load(cache_path)
                        depth_vit_pad = cached["depth"]
                        valid_mask    = cached["valid_mask"]


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
                depths = torch.from_numpy(
                    np.stack(pseudo_depths, axis=0)
                ).unsqueeze(1).to(device)

            # ==================================================
            # GT BRANCH (standard depth validation)
            # ==================================================
            else:
                depths = batch["gt"]
                if depths.ndim == 3:
                    depths = depths.unsqueeze(1)
                depths = depths.to(device, non_blocking=True).float()

            # ==================================================
            # STUDENT FORWARD
            # ==================================================
            preds = model(images)
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)

            # ==================================================
            # LOSS
            # ==================================================
            total, mae, edge, metrics = loss_module(preds, depths)


            sums["total"] += total.item()
            sums["mae"] += mae.item()
            sums["edge"] += edge.item()

            for k in metrics:
                sums[k] += metrics[k].item()


            n_batches += 1

            # ==================================================
            # VISUALIZATION
            # ==================================================
            if len(vis_images) < vis_count:
                for i in range(B):
                    if len(vis_images) >= vis_count:
                        break

                    vis = visualize_depth_triplet(
                        images[i],
                        preds[i],
                        depths[i],
                        colormap=colormap,
                    )
                    vis_images.append(vis)

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    for k in sums:
        sums[k] /= max(n_batches,1)

    avg_mAE = sums["mae"]
    avg_edge = sums["edge"]
    avg_total = sums["total"]
    avg_absrel = sums["abs_rel"]


    # --------------------------------------------------
    # Logger
    # --------------------------------------------------
    if logger is not None:
        logger.log_validation_depth(
            step=step,
            dataset=dataset_name,
            val_loss=avg_total,
            mAE=avg_mAE,
            edge=avg_edge,
            absrel=avg_absrel,
            vis_images=vis_images,
        )
        return avg_total, avg_mAE, avg_edge, avg_absrel

    else:
        return avg_total, avg_mAE, avg_edge, avg_absrel, vis_images


def denormalize_image(
    image_t: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    image_t: (3, H, W) tensor, normalized
    return: HWC uint8 RGB
    """
    mean = torch.tensor(mean, device=image_t.device).view(3, 1, 1)
    std = torch.tensor(std, device=image_t.device).view(3, 1, 1)

    img = image_t * std + mean
    img = img.clamp(0, 1)
    img = (img * 255).byte()

    return img.permute(1, 2, 0).cpu().numpy()


def vit_safe_lower(x, patch=14):
    """Largest value <= x divisible by patch."""
    return max((x // patch) * patch, patch)


def resize_vit_safe_lower(img, patch=14):
    h, w = img.shape[:2]
    new_h = vit_safe_lower(h, patch)
    new_w = vit_safe_lower(w, patch)

    if new_h == h and new_w == w:
        return img, (h, w)

    img_rs = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return img_rs, (h, w)


def center_crop_vit_safe_lower(img, patch=14):
    """
    Center-crop to the largest vit-safe size <= original.
    NO resize, only crop.
    """
    h, w = img.shape[:2]
    new_h = vit_safe_lower(h, patch)
    new_w = vit_safe_lower(w, patch)

    if new_h == h and new_w == w:
        return img, (0, 0)

    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2

    img_crop = img[y0:y0 + new_h, x0:x0 + new_w]

    return img_crop, (y0, x0)



def pad_to_target_center(depth, target_h, target_w, value=0.0):
    h, w = depth.shape

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    depth_pad = np.pad(
        depth,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=value,
    )

    return depth_pad, (pad_top, pad_left)


def compute_pseudo_depth_with_caching(
    raw_vit,
    pseudo_label_generator_model,
    H_tgt,
    W_tgt,
    H_vit,
    W_vit,
    cache_path: Path,
):
    with torch.amp.autocast(device_type="cuda",dtype=torch.float16):
        
        depth_vit = pseudo_label_generator_model.infer_image(raw_vit)

    # ---- pad depth to student size (CENTERED) ----
    depth_vit_pad, (pad_top, pad_left) = pad_to_target_center(
        depth_vit,
        target_h=H_tgt,
        target_w=W_tgt,
        value=0.0,
    )

    # ---- build valid mask (centered) ----
    valid_mask = np.zeros(
        (H_tgt, W_tgt),
        dtype=np.float32,
    )
    valid_mask[
        pad_top : pad_top + H_vit,
        pad_left: pad_left + W_vit,
    ] = 1.0

    # ---- apply mask to depth ----
    depth_vit_pad *= valid_mask

    # ---- save cache (depth + mask) ----
    np.savez_compressed(
        cache_path,
        depth=depth_vit_pad.astype(np.float32),
        valid_mask=valid_mask.astype(np.float32),
    )