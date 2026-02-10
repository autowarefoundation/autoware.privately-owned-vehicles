# dataloader/augmentations/segmentation.py
import albumentations as A
from Models.data_parsing.lite_models.augmentation.BaseAugmentation import BaseAugmentation
import random
import cv2

import os
import numpy as np

import torch

# def save_img_debug(
#     batch,
#     batch_id: int,
#     out_dir: str,
#     ignore_index=255,
# ):
#     """
#     Save ALL images + masks from a batch (PRE-normalization).

#     batch:
#         {
#           "image": Tensor or np.ndarray [3, H, W],
#           "gt"   : Tensor or np.ndarray [H, W]
#         }
#     """

#     os.makedirs(out_dir, exist_ok=True)

#     img = batch["image"]
#     mask = batch["mask"]

#     # ---------- IMAGE ----------

#     # tensor → numpy
#     if isinstance(img, torch.Tensor):
#         img = img.cpu().numpy()

#     # CHW → HWC
#     if img.ndim == 3 and img.shape[0] == 3:
#         img = np.transpose(img, (1, 2, 0))

#     # ensure uint8
#     if img.dtype != np.uint8:
#         img = np.clip(img, 0, 255)
#         if img.max() <= 1.0:
#             img = img * 255.0
#         img = img.astype(np.uint8)

#     # ---------- MASK ----------
#     if isinstance(mask, torch.Tensor):
#         mask = mask.cpu().numpy()

#     mask = mask.astype(np.uint8)
#     mask[mask == ignore_index] = 0

#     # ---------- SAVE ----------
#     img_path = os.path.join(out_dir, f"batch{batch_id}_img.png")
#     mask_path = os.path.join(out_dir, f"batch{batch_id}_mask.png")

#     cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#     cv2.imwrite(mask_path, mask)

#     print(f"[DEBUG] Saved batch {batch_id} with 1 samples → {out_dir}")


class SegmentationAugmentation(BaseAugmentation):
    def _build(self):
        """
        GEOMETRY ONLY.
        (Noise + Normalize are applied in apply_augmentation via BaseAugmentation._postprocess_image)
          rescaling:
            enabled: true
            mode: "fixed_resize"    # fixed_resize | random_crop


            width: 1024       
            height: 512

            #for random cropping
            scale_range: [0.5, 2.0]   # min and max scale for random cropping

        """
        tfs = []
        mode = self.cfg.get("rescaling", {}).get("mode", "fixed_resize")


        #Albumentations Resize interpolation. Default is cv2.INTER_LINEAR for images and cv2.INTER_NEAREST for masks (so default is ok for segmentation)
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
                raise ValueError(f"Unknown segmentation mode: {mode}")

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

        self.transform = A.Compose(tfs)

    def apply_augmentation(self, image, label, dataset_name=None):
        # 1) remove car hood
        image, label = self._remove_car_hood(image, label, dataset_name)

        # 2) geometry (image+label)
        out = self.transform(image=image, mask=label)
        image, label = out["image"], out["mask"]

        #save the intermediate result for debug
        # if self.mode == "train":
        #     self.batch_id += 1
        #     save_img_debug(
        #         out,
        #         batch_id=self.batch_id,
        #         out_dir="debug_batches",
        #     )

        # 3) noise + normalize (image only)
        image = self._apply_noise_img(image)
        image = self._apply_normalize(image)

        return image, label
