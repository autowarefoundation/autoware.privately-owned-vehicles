#!/usr/bin/env python3
"""
Data loader for temporal steering angle prediction network.

This loader handles sequences of images with temporal context:
- Input: 3 consecutive frames [t-2, t-1, t]
- Output: Steering angle at time t

Dataset structure:
    dataset_root/
        images/
            <timestamp1>.jpg
            <timestamp2>.jpg
            ...
        steering_angles.json
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SteeringAngleDataset(Dataset):
    """
    Dataset for temporal steering angle prediction.
    
    Returns sequences of 3 frames with corresponding steering angle.
    """
    
    def __init__(
        self,
        dataset_root,
        image_size=(320, 640),
        temporal_length=3,
        augment=False,
        split='train',
        val_cap=500,
    ):
        """
        Args:
            dataset_root: Path to dataset directory containing images/ and steering_angles.json
            image_size: Target image size (height, width)
            temporal_length: Number of consecutive frames (default: 3 for t-2, t-1, t)
            augment: Enable data augmentation
            split: 'train' or 'val'
            val_cap: Maximum validation samples
        """
        self.dataset_root = dataset_root
        self.image_dir = os.path.join(dataset_root, 'images')
        self.json_path = os.path.join(dataset_root, 'steering_angles.json')
        self.image_size = image_size
        self.temporal_length = temporal_length
        self.augment = augment
        self.split = split
        self.val_cap = val_cap
        
        # Load annotations
        self._load_annotations()
        
        # Split into train/val
        self._split_data()
        
        # Image preprocessing
        self.transform = self._get_transforms()
        
        print(f"Dataset loaded with {self.N_trains} trains and {self.N_vals} vals.")
    
    def _load_annotations(self):
        """Load steering angle annotations from JSON file."""
        with open(self.json_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Sort by timestamp
        self.annotations = sorted(self.annotations, key=lambda x: x['timestamp'])
    
    def _split_data(self):
        """Split data into train/val following AutoSteer pattern."""
        self.train_indices = []
        self.val_indices = []
        self.N_trains = 0
        self.N_vals = 0
        
        # Start from temporal_length-1 to have enough history
        for set_idx in range(self.temporal_length - 1, len(self.annotations)):
            if (
                (set_idx % 10 == 0) and
                (self.N_vals < self.val_cap)
            ):
                # Slap it to Val
                self.val_indices.append(set_idx)
                self.N_vals += 1
            else:
                # Slap it to Train
                self.train_indices.append(set_idx)
                self.N_trains += 1
        
        # Set active indices based on split
        if self.split == 'train':
            self.indices = self.train_indices
        else:
            self.indices = self.val_indices
    
    def _get_transforms(self):
        """Get image preprocessing transforms."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ]
        return transforms.Compose(transform_list)
    
    def _load_image(self, timestamp):
        """Load and preprocess image by timestamp."""
        img_path = os.path.join(self.image_dir, f"{timestamp}.jpg")
        image = Image.open(img_path).convert('RGB')
        return image
    
    def _apply_augmentation(self, images, steering_angle):
        """
        Apply data augmentation to image sequence and steering angle.
        
        Args:
            images: List of PIL images [t-2, t-1, t]
            steering_angle: Current steering angle
            
        Returns:
            Augmented images and steering angle
        """
        # Horizontal flip (with steering angle negation)
        if np.random.rand() > 0.5:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            steering_angle = -steering_angle
        
        # Brightness adjustment (apply same to all frames for consistency)
        if np.random.rand() > 0.5:
            brightness_factor = np.random.uniform(0.7, 1.3)
            from PIL import ImageEnhance
            enhancer_list = [ImageEnhance.Brightness(img) for img in images]
            images = [enh.enhance(brightness_factor) for enh in enhancer_list]
        
        return images, steering_angle
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a sample: temporal image sequence + steering angle."""
        ann_idx = self.indices[idx]
        
        # Load temporal sequence [t-2, t-1, t]
        images = []
        for offset in range(self.temporal_length):
            frame_idx = ann_idx - (self.temporal_length - 1 - offset)
            timestamp = self.annotations[frame_idx]['timestamp']
            img = self._load_image(timestamp)
            images.append(img)
        
        # Get steering angle
        current_annotation = self.annotations[ann_idx]
        steering_angle = current_annotation['steering_angle']
        zero_point = current_annotation['steering_zero_point']
        steering_angle = steering_angle - zero_point
        
        # Augmentation
        if self.augment:
            images, steering_angle = self._apply_augmentation(images, steering_angle)
        
        # Transform to tensors
        image_tensors = [self.transform(img) for img in images]
        image_sequence = torch.stack(image_tensors, dim=0)
        steering_tensor = torch.tensor([steering_angle], dtype=torch.float32)
        
        return image_sequence, steering_tensor
    
    def get_sample_info(self, idx):
        """Get metadata for a sample."""
        ann_idx = self.indices[idx]
        info = {
            'current_timestamp': self.annotations[ann_idx]['timestamp'],
            'current_angle': self.annotations[ann_idx]['steering_angle'],
            'sequence_timestamps': []
        }
        for offset in range(self.temporal_length):
            frame_idx = ann_idx - (self.temporal_length - 1 - offset)
            info['sequence_timestamps'].append(self.annotations[frame_idx]['timestamp'])
        return info


def get_steering_dataloaders(
    dataset_root,
    batch_size=16,
    num_workers=4,
    image_size=(320, 640),
    temporal_length=3,
    augment_train=True,
    val_cap=500
):
    """Create train and validation dataloaders."""
    train_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        temporal_length=temporal_length,
        augment=augment_train,
        split='train',
        val_cap=val_cap
    )
    
    val_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        temporal_length=temporal_length,
        augment=False,
        split='val',
        val_cap=val_cap
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for consistent LSTM training
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_data_steering_network.py <dataset_root>")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    
    # Create datasets
    train_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        temporal_length=3,
        augment=False,
        split='train'
    )
    
    val_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        temporal_length=3,
        augment=False,
        split='val'
    )
    
    # Test samples
    if len(train_dataset) > 0:
        images, steering = train_dataset[0]
        print(f"\nTrain sample: {images.shape}, angle={steering.item():.4f}")
        info = train_dataset.get_sample_info(0)
        print(f"Timestamps: {info['sequence_timestamps']}")
    
    if len(val_dataset) > 0:
        images, steering = val_dataset[0]
        print(f"\nVal sample: {images.shape}, angle={steering.item():.4f}")

