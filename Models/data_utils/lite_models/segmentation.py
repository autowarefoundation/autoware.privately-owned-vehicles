# utils/utils_segmentation.py
import numpy as np
import torch
from tqdm import tqdm

import cv2


def validate_segmentation(model, dataloader, loss_fn, device,
                          loss_cfg, logger=None, step=None, dataset_name=None, vis_count=10):
    """
    Professional validation loop for semantic segmentation.

    Returns:
        val_loss (float)
        mean_iou (float)
        class_iou_dict (dict: class_name -> IoU)
        y_true (np.ndarray)  # flattened ground truth labels
        y_pred (np.ndarray)  # flattened predicted labels
    """

    model.eval()
    total_loss = 0.0
    num_batches = 0

    # ---- Load class names ----
    class_names = loss_cfg.get("class_names", None)
    if class_names is None:
        raise ValueError("loss_cfg['class_names'] must contain class_names (Cityscapes 19).")

    num_classes = len(class_names)

    # ---- Create confusion matrix ----
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # For W&B confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Validation]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["gt"].to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            loss = loss_fn(logits, masks)
            prediction = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            num_batches += 1

            # Predictions
            pred = torch.argmax(logits, dim=1)     # [N,H,W]

            # Accumulate confusion matrix
            batch_conf = compute_confusion_matrix(
                pred.cpu().numpy(),
                masks.cpu().numpy(),
                num_classes=num_classes
            )
            confmat += batch_conf

    # ---- Final metrics ----
    val_loss = total_loss / max(1, num_batches)

    # Compute IoU
    class_iou = compute_iou_from_confmat(confmat)
    mean_iou = float(np.nanmean(class_iou))

    class_iou_dict = {class_names[i]: float(class_iou[i]) for i in range(num_classes)}
    val_loss = total_loss / max(1, num_batches)

    # ---- PREPARE VECTORS FOR W&B ----
    y_true = masks.cpu().numpy().reshape(-1)
    y_pred = pred.cpu().numpy().reshape(-1)

    # Remove ignored 255 pixels
    valid_mask = y_true != 255
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    # Clamp predicted classes (safety)
    y_pred = np.clip(y_pred, 0, num_classes - 1)


    # ---- Select sample visualizations ----
    VIS_COUNT = vis_count      #do visualizationfor 10 samples across the validation set only
    sample_images = []
    sample_masks = []

    dataloader_iter = iter(dataloader)
    for _ in range(VIS_COUNT):
        try:
            batch_sample = next(dataloader_iter)
            sample_images.append(batch_sample["image"][0])
            sample_masks.append(batch_sample["gt"][0])
        except StopIteration:
            break

    vis_pairs = []
    with torch.no_grad():
        for img_t, gt_t in zip(sample_images, sample_masks):
            img_t = img_t.to(device).unsqueeze(0)
            logits = model(img_t)
            pred_mask = torch.argmax(logits, dim=1)[0]

            vis_img = visualize_triplet(img_t[0], pred_mask, gt_t, class_names)
            vis_pairs.append(vis_img)



    # ---- Logging ----. send images to w&b logger
    if logger is not None:
        logger.log_validation_segmentation(
            step=step,
            dataset=dataset_name,
            val_loss=val_loss,
            mean_iou=mean_iou,
            class_iou=class_iou_dict,
            confmat=confmat,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            vis_images=vis_pairs,
        )
        
        return val_loss, mean_iou, class_iou_dict

    #return images in case no logger is provided
    return val_loss, mean_iou, class_iou_dict, vis_pairs

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_confusion_matrix(preds, labels, num_classes):
    """
    Vectorized confusion matrix for segmentation.
    
    preds:  [N,H,W] int
    labels: [N,H,W] int
    """
    mask = (labels >= 0) & (labels < num_classes)
    preds = preds[mask]
    labels = labels[mask]

    idxs = num_classes * labels.astype(int) + preds.astype(int)
    confmat = np.bincount(idxs, minlength=num_classes*num_classes)
    return confmat.reshape(num_classes, num_classes)


def compute_iou_from_confmat(confmat):
    """
    Compute IoU per class from confusion matrix.
    """
    tp = np.diag(confmat)
    fp = confmat.sum(axis=0) - tp
    fn = confmat.sum(axis=1) - tp

    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, np.nan)
    return iou




# Cityscapes 19-class color palette, RGB
CITYSCAPES_COLORS = np.array([
    [128,  64, 128],  # road
    [244,  35, 232],  # sidewalk
    [ 70,  70,  70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170,  30],  # traffic light
    [220, 220,   0],  # traffic sign
    [107, 142,  35],  # vegetation
    [152, 251, 152],  # terrain
    [ 70, 130, 180],  # sky
    [220,  20,  60],  # person
    [255,   0,   0],  # rider
    [  0,   0, 142],  # car
    [  0,   0,  70],  # truck
    [  0,  60, 100],  # bus
    [  0,  80, 100],  # train
    [  0,   0, 230],  # motorcycle
    [119,  11,  32],  # bicycle
    [  0,   0,   0] # 255 - unlabeled
], dtype=np.uint8)


def mask_to_cityscapes(mask):
    """
    Convert HxW class ID mask → HxWx3 RGB visualization
    """
    mask = np.asarray(mask)

    # Handle ignore label
    safe_mask = mask.copy()
    safe_mask[safe_mask == 255] = 19  # last color = black

    colored = CITYSCAPES_COLORS[safe_mask]
    return colored.astype(np.uint8)



def visualize_triplet(image_tensor, mask_pred, gt, class_names):
        # Undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    # --- IMAGE ---
    img = image_tensor.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)   # CHW → HWC
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- PRED MASK ---
    pred_np = mask_pred.detach().cpu().numpy()
    pred_colored = mask_to_cityscapes(pred_np)

    # --- GT MASK ---
    gt_np = gt.detach().cpu().numpy()
    gt_colored = mask_to_cityscapes(gt_np)

    # --- SANITY CHECK ---
    assert img.ndim == 3
    assert pred_colored.ndim == 3
    assert gt_colored.ndim == 3

    # --- STACK HORIZONTALLY ---
    combined = np.concatenate(
        [img, pred_colored, gt_colored],
        axis=1
    )

    H, W, _ = combined.shape
    num_classes = len(class_names)

    legend_width = int(W * 0.05)
    legend = np.ones((H, legend_width, 3), dtype=np.uint8) * 255

    row_h = H // num_classes

    for i in range(num_classes):
        color = CITYSCAPES_COLORS[i].tolist()

        y1 = i * row_h
        y2 = min(H, y1 + row_h)

        # rectangle swatch
        cv2.rectangle(
            legend,
            (10, y1 + 5),
            (40, y2 - 5),
            color,      #stays in RGB
            thickness=-1
        )

        # text
        text_y = y1 + row_h // 2 + 5
        cv2.putText(
            legend,
            str(class_names[i]).upper(),   # MAIUSCOLO
            (50, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,                           # font scale (prima era 0.5)
            (0, 0, 0),
            2,                             # thickness (prima era 1)
            cv2.LINE_AA
        )


    return np.concatenate([combined, legend], axis=1)

