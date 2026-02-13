# dataloader/augmentations/depth.py

import albumentations as A
import cv2
from Models.data_utils.lite_models.augmentation.BaseAugmentation import BaseAugmentation

import random

def vit_safe(x, patch=14):
    """Return smallest value >= x divisible by patch."""
    return ((x + patch - 1) // patch) * patch


class DepthAugmentation(BaseAugmentation):
    """
    Depth augmentation pipeline.

    PSEUDO-LABELING (pseudo_labeling == True):
        TRAIN : 
            -random crop → (out_h, out_w) to IMAGE
            -random
      - Random crop → (out_h, out_w)
      - Shared random ops (student + teacher)
      - STUDENT: noise + normalize
      - TEACHER: resize → vit-safe (NO noise, NO normalize)
    """

    # =====================================================
    # Build geometry pipelines
    # =====================================================
    def _build(self):
        """
        Depth geometry pipeline.

        Supports:
        - fixed_resize
        - random_crop
        - stride-aligned padding for val/test
        - pseudo-labeling (student / teacher split)
        """

        rescaling = self.cfg.get("rescaling", {}) or {}
        mode = rescaling.get("mode", "fixed_resize")

        self.out_w = int(rescaling.get("width", 640))
        self.out_h = int(rescaling.get("height", 320))

        flip_p = float(self.cfg.get("flip_prob", 0.0))
        gs_cfg = self.cfg.get("random_grid_shuffle", {}) or {}

        val_transformations = []
        train_transformations = []

        # ============================================================
        #  pseudo-labeling
        # ============================================================

        # -------------------------
        # TRAIN
        # -------------------------
        if self.mode == "train":

            if mode == "fixed_resize":
                train_transformations.append(
                    A.Resize(
                        height=self.out_h,
                        width=self.out_w,
                        interpolation=cv2.INTER_LINEAR,
                    )
                )

            elif mode == "random_crop":
                crop_h = self.out_h
                crop_w = self.out_w
                scale_min, scale_max = rescaling.get("scale_range", [0.5, 2.0])

                shared = {}

                def sample_valid_scale(h, w):
                    for _ in range(10):
                        scale = (
                            random.randint(
                                int(scale_min * 10), int(scale_max * 10)
                            )
                            / 10.0
                        )
                        if h * scale >= crop_h and w * scale >= crop_w:
                            return scale
                    return max(crop_h / h, crop_w / w)

                def scale_image(img, **kwargs):
                    h, w = img.shape[:2]
                    scale = sample_valid_scale(h, w)
                    shared["scale"] = scale
                    return cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                def scale_gt(gt, **kwargs):
                    h, w = gt.shape[:2]
                    scale = shared["scale"]
                    return cv2.resize(
                        gt,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )


                #apply transformations (resizing w/ random scale + random crop)
                train_transformations.extend(
                    [
                        #resize the image with a random scale (between scale max and min)
                        A.Lambda(image=scale_image, mask=scale_gt),

                        # then random crop to desired size
                        A.PadIfNeeded(
                            min_height=None,
                            min_width=None,
                            pad_height_divisor=self.output_stride,
                            pad_width_divisor=self.output_stride,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                        ),

                        #perform random crop
                        A.RandomCrop(height=crop_h, width=crop_w),
                    ]
                )

            else:
                raise ValueError(f"Unknown depth rescaling mode: {mode}")


            # apply random horizontal flip 
            if flip_p > 0:
                train_transformations.append(A.HorizontalFlip(p=flip_p))

            # apply random grid shuffle
            if gs_cfg.get("enabled", False):
                train_transformations.append(
                    A.RandomGridShuffle(
                        grid=tuple(gs_cfg.get("grid", (4, 4))),
                        p=float(gs_cfg.get("prob", 0.0)),
                    )
                )

        # -------------------------
        # VAL / TEST (pseudo label generation at runtime)
        # -------------------------
        else:
            val_transformations.append(
                A.PadIfNeeded(
                    min_height=None,
                    min_width=None,
                    pad_height_divisor=self.output_stride,
                    pad_width_divisor=self.output_stride,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                )
            )


        #compose the transformations
        self.val_transformations = A.Compose(val_transformations)
        self.train_transformations = A.Compose(train_transformations)


        return None

    # =====================================================
    # Public API
    # =====================================================
    def apply_augmentation(self, image, gt, dataset_name=None):

        # 1) remove car hood
        image, gt = self._remove_car_hood(image, gt, dataset_name)

        # =================================================
        # PSEUDO-LABELING
        # =================================================

        if self.pseudo_labeling and self.mode == "train":
            out = self.train_transformations(image=image, mask=gt)
            image, gt = out["image"], out["mask"]
        elif self.pseudo_labeling and self.mode != "train":
            out = self.val_transformations(image=image, mask=gt)
            image, gt = out["image"], out["mask"]
        else:
            ValueError("Real GT not supported in DepthAugmentation.")


        image = self._apply_noise_img(image)
        image = self._apply_normalize(image)

        # --------------------------------------------------
        # GT CONSISTENCY 
        # --------------------------------------------------
        if gt is not None and self.pseudo_labeling == False:
            gt = cv2.resize(
                gt,
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return image, gt
