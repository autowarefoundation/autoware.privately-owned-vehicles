# dataloader/augmentations/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import random
import numpy as np
import albumentations as A


class BaseAugmentation(ABC):
    def __init__(self, mode: str, aug_cfg: dict, pseudo_labeling: bool = False):

        #mode : train / val / test
        self.mode = mode or "train"

        self.cfg = aug_cfg or {}

        self.pseudo_labeling = pseudo_labeling  #used for depth supervision

        # -------------------------
        # normalization (image only)
        # -------------------------
        norm = self.cfg.get("normalize", {}) or {}
        self.norm_enabled = bool(norm.get("enabled", False))
        self.mean = np.array(norm.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std  = np.array(norm.get("std",  [0.229, 0.224, 0.225]), dtype=np.float32)

        # -------------------------
        # noise (image only)
        # -------------------------
        noise_cfg = self.cfg.get("noise", {}) or {}
        self.noise_prob = float(noise_cfg.get("prob", 0.5))
        self.noise_img = self._build_image_noise(noise_cfg)


        # output stride in case some models need multiples of it
        self.output_stride = self.cfg.get("output_stride", 16)

        # build geometry pipeline ONCE (implemented in derived classes)
        self._build()

        self.batch_id = 0

        # car hood removal for automotive datasets
        self.remove_car_hood = self.cfg.get("remove_car_hood", True)


    # =========================
    # Abstract API
    # =========================
    @abstractmethod
    def _build(self):
        """Build albumentations GEOMETRY pipeline(s)."""
        raise NotImplementedError

    @abstractmethod
    def apply_augmentation(self, image, target, dataset_name=None):
        """Apply geometry + (noise+normalize on image)"""
        raise NotImplementedError

    # =========================
    # Common: dataset-specific crop
    # =========================
    def _remove_car_hood(self, image, gt, path):

        #apply car hood removal only during training
        if self.mode != "train" or self.remove_car_hood == False:
            return image, gt
        

        p = (path or "").lower()
        cut = None
        if "acdc" in p: cut = 990
        elif "muses" in p: cut = 918
        elif "idda" in p: cut = 950
        elif "bdd100k" in p: cut = 640

        if cut is None:
            return image, gt

        image = image[:cut]
        if gt is not None:
            gt = gt[:cut]
        return image, gt

    # =========================
    # Common: noise + normalize (image only)
    # =========================
    def _build_image_noise(self, noise_cfg: dict):
        """
        Build Albumentations noise pipeline ONCE.
        """
        profile = (noise_cfg.get("profile", "none") or "none").lower()

        if profile == "none":
            return None

        if profile == "moderate":
            return A.Compose([
                A.PixelDropout(dropout_prob=0.25, per_channel=True, p=0.05),
                A.MultiplicativeNoise(multiplier=(0.2, 0.5), p=0.05),
                A.Spatter(
                    mean=(0.65, 0.65), std=(0.3, 0.3),
                    gauss_sigma=(2, 2),
                    cutout_threshold=(0.68, 0.68),
                    intensity=(0.3, 0.3),
                    mode="rain", p=0.05
                ),
                A.ToGray(num_output_channels=3, p=0.1),
                A.RandomRain(p=0.05),
                A.RandomShadow(
                    shadow_roi=(0.2, 0.2, 0.8, 0.8),
                    num_shadows_limit=(2, 4),
                    shadow_dimension=8,
                    shadow_intensity_range=(0.3, 0.7),
                    p=0.05
                ),
                A.RandomGravel(
                    gravel_roi=(0.2, 0.2, 0.8, 0.8),
                    number_of_patches=5,
                    p=0.05
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.5,
                    p=0.05
                ),
                A.ISONoise(color_shift=(0.1, 0.3), intensity=(0.5, 0.5), p=0.05),
                A.GaussNoise(noise_scale_factor=0.2, p=0.05),
            ])

        if profile == "heavy":
            return A.Compose([
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), p=0.5),
                A.PixelDropout(dropout_prob=0.025, per_channel=True, p=0.25),
                A.ColorJitter(
                    brightness=0.6, contrast=0.6,
                    saturation=0.6, hue=0.2, p=0.5
                ),
                A.GaussNoise(noise_scale_factor=0.2, p=0.5),
                A.GaussNoise(noise_scale_factor=1.0, p=0.5),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.5),
                A.RandomFog(alpha_coef=0.2, p=0.25),
                A.RandomFog(alpha_coef=0.04, p=0.25),
                A.RandomRain(p=0.1),
                A.Spatter(
                    mean=(0.65, 0.65), std=(0.3, 0.3),
                    gauss_sigma=(2, 2),
                    cutout_threshold=(0.68, 0.68),
                    intensity=(0.3, 0.3),
                    mode="rain", p=0.1
                ),
                A.ToGray(num_output_channels=3, p=0.1),
            ])

        raise ValueError(f"Unknown noise profile: {profile}")



    def _apply_noise_img(self, img):
        if (
            self.mode == "train"
            and self.noise_img is not None
            and random.random() < self.noise_prob
        ):
            return self.noise_img(image=img)["image"]
        return img


    def _apply_normalize(self, img):
        """
        Identico semantico al tuo originale:
        - input uint8 [0..255]
        - output float32 normalized
        """
        if not self.norm_enabled:
            return img
        img = img.astype(np.float32) / 255.0
        return (img - self.mean) / self.std
