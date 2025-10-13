#! /usr/bin/env python3

import random
import albumentations as A
import numpy as np
from typing import Literal, get_args

DATA_TYPES_LITERAL = Literal[
    "SEGMENTATION",
    "BINARY_SEGMENTATION",
    "DEPTH",
    "KEYPOINTS"
]
DATA_TYPES_LIST = list(get_args(DATA_TYPES_LITERAL))

class Augmentations():
    def __init__(self, is_train: bool, data_type: DATA_TYPES_LITERAL):

        # Data
        self.image = None
        self.ground_truth = None
        self.augmented_data = None

        # Train vs Test/Val mode
        self.is_train = is_train

        # Dataset type
        self.data_type = data_type
        if not (self.data_type in DATA_TYPES_LIST):
            raise ValueError('Dataset type is not correctly specified')

        # ========================== Shape transforms ========================== #

        self.transform_shape = A.Compose(
            [
                A.Resize(width = 640, height = 320),
                A.HorizontalFlip(p = 0.5),
            ]
        )

        self.transform_shape_with_shuffle = A.Compose(
            [
                A.Resize(width = 640, height = 320),
                A.HorizontalFlip(p = 0.5),
                A.RandomGridShuffle(grid=(1,2), p=0.25)
            ]
        )

        self.transform_shape_test = A.Compose(
            [
                A.Resize(width = 640, height = 320),
            ]
        )

        self.transform_shape_bev = A.Compose(
            [
                A.Resize(width = 640, height = 320),
            ]
        )

        # ========================== Noise transforms ========================== #

        self.transform_noise_ego_space = A.Compose(
            [
                A.PixelDropout(dropout_prob=0.25, per_channel=True, p=0.1),
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.3, 0.3), mode='rain', \
                    p=0.1),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.1),
                A.RandomRain(p=0.1),
                A.RandomShadow(shadow_roi=(0.2, 0.2, 0.8, 0.8), num_shadows_limit=(2, 4), shadow_dimension=8, \
                    shadow_intensity_range=(0.3, 0.7), p=0.1),
                A.RandomGravel(gravel_roi=(0.2, 0.2, 0.8, 0.8), number_of_patches=5, p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.5, p=0.1),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.1),
                A.GaussNoise(noise_scale_factor=0.2, p=0.1)
            ]
        )

        self.transform_noise = A.Compose(
            [
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=0.5),
                A.PixelDropout(dropout_prob=0.025, per_channel=True, p=0.25),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=0.5),
                A.GaussNoise(noise_scale_factor=0.2, p=0.5),
                A.GaussNoise(noise_scale_factor=1, p=0.5),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.5),
                A.RandomFog(alpha_coef=0.2, p=0.25),
                A.RandomFog(alpha_coef=0.04, p=0.25),
                A.RandomRain(p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.3, 0.3), mode='rain', \
                    p=0.1),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.1)
            ]
        )

        self.transform_noise_roadwork = A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=[-180, 180], sat_shift_limit=[-150,150], \
                    val_shift_limit=[-80, 80], p=1.0),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.5)
            ]
        )

    # ========================== Data type specific transform functions ========================== #

    # Set ground truth and image data

    def setData(self, image, ground_truth):
        self.image = image
        self.ground_truth = ground_truth

        self.augmented_data = ground_truth
        self.augmented_image = image

    def setImage(self, image):
        self.image = image
        self.augmented_image = image


    # SEMANTIC SEGMENTATION - SceneSeg
    # Apply augmentations transform
    def applyTransformSeg(self, image, ground_truth):

        if(self.data_type != 'SEGMENTATION'):
            raise ValueError('Please set dataset type to SEGMENTATION in intialization of class')

        self.setData(image, ground_truth)

        if(self.is_train):

            # Resize and random horiztonal flip
            self.adjust_shape = self.transform_shape(image=self.image, \
                masks = self.ground_truth)

            self.augmented_data = self.adjust_shape["masks"]
            self.augmented_image = self.adjust_shape["image"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):

                self.add_noise = self.transform_noise(image=self.augmented_image)
                self.augmented_image = self.add_noise["image"]
        else:
            # Only resize in test/validation mode
            self.adjust_shape = self.transform_shape_test(image=self.image, \
            masks = self.ground_truth)
            self.augmented_data = self.adjust_shape["masks"]
            self.augmented_image = self.adjust_shape["image"]

        return self.augmented_image, self.augmented_data

    # BINARY SEGMENTATION - DomainSeg, EgoSpace
    # Apply augmentations transform
    def applyTransformBinarySeg(self, image, ground_truth):

        if(self.data_type != 'BINARY_SEGMENTATION'):
            raise ValueError('Please set dataset type to BINARY_SEGMENTATION in intialization of class')

        self.setData(image, ground_truth)

        if(self.is_train):

            # Resize and random horiztonal flip
            self.adjust_shape = self.transform_shape(image=self.image, \
                mask=self.ground_truth)

            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):

                self.add_noise = self.transform_noise_ego_space(image=self.augmented_image)
                self.augmented_image = self.add_noise["image"]

        else:

            # Only resize in test/validation mode
            self.adjust_shape = self.transform_shape_test(image=self.image, \
                mask = self.ground_truth)
            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]
        return self.augmented_image, self.augmented_data

    # DEPTH ESTIMATION - Scene3D
    # Apply augmentations transform
    def applyTransformDepth(self, image, ground_truth):

        if(self.data_type != 'DEPTH'):
            raise ValueError('Please set dataset type to DEPTH in intialization of class')

        self.setData(image, ground_truth)

        if(self.is_train):

            # Resize and random horiztonal flip
            self.adjust_shape = self.transform_shape(image=self.image, \
                mask=self.ground_truth)

            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]

            # Random image augmentations
            if (random.random() >= 0.25 and self.is_train):

                self.add_noise = self.transform_noise(image=self.augmented_image)
                self.augmented_image = self.add_noise["image"]

        else:

            # Only resize in test/validation mode
            self.adjust_shape = self.transform_shape_test(image=self.image, \
                mask = self.ground_truth)
            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]
        return self.augmented_image, self.augmented_data

    # KEYPOINTS - EgoPath, EgoLanes
    # Apply augmentation transform
    def applyTransformKeypoint(self, image):

        if (self.data_type != "KEYPOINTS"):
            raise ValueError("Please set dataset type to KEYPOINTS in intialization of class")

        self.setImage(image)

        # For train set
        if (self.is_train):

            # Resize image
            self.adjust_shape = self.transform_shape_bev(image = self.image)
            self.augmented_image = self.adjust_shape["image"]

            # Add noise
            if(random.random() >= 0.25):
                self.add_noise = self.transform_noise_ego_space(image = self.augmented_image)
                self.augmented_image = self.add_noise["image"]

        # For test/val sets
        else:

            # Only resize the image without any augmentations
            self.adjust_shape = self.transform_shape_bev(image = self.image)
            self.augmented_image = self.adjust_shape["image"]

        return self.augmented_image


    # ADDITIONAL DATA SPECIFIC NOISE
    # Apply roadwork objects noise for DomainSeg
    def applyNoiseRoadWork(self):
        if(self.is_train):
            self.add_noise = self.transform_noise_roadwork(image=self.augmented_image)
            self.augmented_image = self.add_noise["image"]

        return self.augmented_image
