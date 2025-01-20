import albumentations as A
from typing import Literal
import random

class Augmentations():
    def __init__(self, input_image, ground_truth, is_train, data_type: Literal['SEGMENTATION', 'DEPTH']):
        self.is_train = is_train
        self.image = input_image
        self.ground_truth = ground_truth

        self.data_type = data_type

        if(self.data_type != 'SEGMENTATION' and self.data_type != 'DEPTH'):
            raise ValueError('Dataset type is not correctly specified')

        transform_shape = A.Compose(
            [
                A.Resize(width = 640, height = 320),   
                A.HorizontalFlip(p = 0.5),   
            ]
        )

        transform_noise = A.Compose(
            [      
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=False, p=0.5),
                A.PixelDropout(dropout_prob=0.025, per_channel=True, p=0.25),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, p=0.5),
                A.GaussNoise(var_limit=(50.0, 100.0), mean=0, noise_scale_factor=0.2, p=0.5),
                A.GaussNoise(var_limit=(250.0, 250.0), mean=0, noise_scale_factor=1, p=0.5),
                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.5, 0.5), p=0.5),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.4, p=0.25),
                A.RandomRain(p=0.1),
                A.Spatter(mean=(0.65, 0.65), std=(0.3, 0.3), gauss_sigma=(2, 2), \
                    cutout_threshold=(0.68, 0.68), intensity=(0.3, 0.3), mode='rain', \
                    p=0.1),
                A.ToGray(num_output_channels=3, method='weighted_average', p=0.1)           
            ]
        )
    
        
        self.augmented_data = ground_truth
        self.augmented_image = input_image

        if (self.data_type == 'SEGMENTATION'):

            self.adjust_shape = transform_shape(image=self.image, \
                masks = self.ground_truth)
            
            self.augmented_data = self.adjust_shape["masks"]
            self.augmented_image = self.adjust_shape["image"]

        elif(self.data_type == 'DEPTH'):

            self.adjust_shape = transform_shape(image=self.image, \
                mask = self.ground_truth)
            
            self.augmented_data = self.adjust_shape["mask"]
            self.augmented_image = self.adjust_shape["image"]

        if (random.random() >= 0.25 and self.is_train):
            
            self.add_noise = transform_noise(image=self.augmented_image)
            self.augmented_image = self.add_noise["image"]
        
        self.getAugmentedData()

    def getAugmentedData(self):
        return self.augmented_image, self.augmented_data