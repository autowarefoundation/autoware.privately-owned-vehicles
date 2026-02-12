# dataloader/augmentations/lanes.py
import albumentations as A
from dataloader.augmentation.BaseAugmentation import BaseAugmentation
import random
import cv2

import os
import numpy as np

import torch


class LanesAugmentation(BaseAugmentation):
    def _build(self):
        tfs = []
        mode = self.cfg.get("rescaling", {}).get("mode", "fixed_resize")


        #Albumentations Resize interpolation. Default is cv2.INTER_LINEAR for images and cv2.INTER_NEAREST for masks
        if self.mode == "train":
            if mode == "fixed_resize":
                tfs.append(
                    A.Resize(
                        height=int(self.cfg["rescaling"]["height"]),
                        width=int(self.cfg["rescaling"]["width"]),
                    )
                )

            elif mode == "random_crop":
                crop_h = int(self.cfg["rescaling"]["height"])
                crop_w = int(self.cfg["rescaling"]["width"])

                scale_min, scale_max = self.cfg["rescaling"]["scale_range"]

                # shared state per image+mask (per-sample)
                shared = {}

                def sample_valid_scale(h, w):
                    for _ in range(10):
                        scale = random.randint(
                            int(scale_min * 10),
                            int(scale_max * 10)
                        ) / 10.0
                        if h * scale >= crop_h and w * scale >= crop_w:
                            return scale
                    return max(crop_h / h, crop_w / w)

                def scale_image(img, **kwargs):
                    h, w = img.shape[:2]
                    scale = sample_valid_scale(h, w)
                    shared["scale"] = scale  # salva la scala
                    return cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                def scale_mask(mask, **kwargs):
                    h, w = mask.shape[:2]
                    scale = shared["scale"]  # RIUSA la scala
                    return cv2.resize(
                        mask,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_NEAREST,
                    )

                tfs.extend([
                    # 1) global multi-scale (SAFE)
                    A.Lambda(
                        image=scale_image,
                        mask=scale_mask,
                    ),
                    # 2) padding minimo (raramente attivo)
                    A.PadIfNeeded(
                        min_height=crop_h,
                        min_width=crop_w,
                        border_mode=0,
                        value=0,
                        mask_value=255,
                    ),
                    # 3) random crop FINALE
                    A.RandomCrop(height=crop_h, width=crop_w),
                ])


            else:
                raise ValueError(f"Unknown lane detection mode: {mode}")

            flip_p = float(self.cfg.get("flip_prob", 0.0))
            if flip_p > 0:
                tfs.append(A.HorizontalFlip(p=flip_p))

        else:
            # validation / test
            if mode == "fixed_resize":
                # explicit fixed-res validation (if you really want it)
                tfs.append(
                    A.Resize(
                        height=int(self.cfg["rescaling"]["height"]),
                        width=int(self.cfg["rescaling"]["width"]),
                    )
                )

            else:
                # NO crop, NO resize
                # only pad to make H,W divisible by output stride (e.g. 16)
                output_stride = 16      #used by deeplabv3plus

                tfs.append(
                    A.PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_height_divisor=output_stride,
                        pad_width_divisor=output_stride,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,          # image padding
                        mask_value=255,   # IGNORE label
                    )
                )

        self.transform = A.ReplayCompose(tfs)

    def apply_augmentation(self, image, label, dataset_name=None):
        # 1) remove car hood
        image, label = self._remove_car_hood(image, label, dataset_name)

        # 2) geometry (image+label)
        out = self.transform(image=image, mask=label)
        image, label = out["image"], out["mask"]

        #check if flip has been applied. if so, swap the lane labels for class0 and class1 (left -> right, right --> left)
        # --------------------------------------------------
        # Check if HorizontalFlip was applied
        # --------------------------------------------------
        flip_applied = False
        for t in out["replay"]["transforms"]:
            if t["__class_fullname__"].endswith("HorizontalFlip") and t["applied"]:
                flip_applied = True
                break

        # --------------------------------------------------
        # If flipped → swap left/right channels
        # --------------------------------------------------
        if flip_applied:
            # label shape is [H, W, 3] at this stage
            label = label.copy()
            label[..., [0, 1]] = label[..., [1, 0]]
            # print("[LanesAugmentation] Applied HorizontalFlip, swapped left/right lane channels.")


        # 3) noise + normalize (image only)
        image = self._apply_noise_img(image)
        image = self._apply_normalize(image)

        return image, label
