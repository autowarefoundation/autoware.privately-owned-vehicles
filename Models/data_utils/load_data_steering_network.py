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
        self.normalize_angles = normalize_angles
        self.angle_range = angle_range
        self.split = split
        self.val_ratio = val_ratio
        
        # Load and parse JSON
        self._load_annotations()
        
        # Create valid sample indices (skip first temporal_length-1 frames)
        self._create_valid_indices()
        
        # Apply train/val split
        self._apply_split()
        
        # Image preprocessing
        self.transform = self._get_transforms()
        
        print(f"[SteeringDataset] Split: {split}")
        print(f"[SteeringDataset] Loaded {len(self.valid_indices)} valid samples")
        print(f"[SteeringDataset] Temporal length: {temporal_length}")
        print(f"[SteeringDataset] Image size: {image_size}")
        print(f"[SteeringDataset] Augmentation: {augment}")
    
    def _load_annotations(self):
        """Load steering angle annotations from JSON file."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Annotations not found: {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Sort by timestamp to ensure temporal order
        self.annotations = sorted(self.annotations, key=lambda x: x['timestamp'])
        
        print(f"[SteeringDataset] Loaded {len(self.annotations)} annotations")
        
        # Verify all images exist
        missing_count = 0
        for ann in self.annotations:
            img_path = os.path.join(self.image_dir, f"{ann['timestamp']}.jpg")
            if not os.path.exists(img_path):
                missing_count += 1
        
        if missing_count > 0:
            print(f"[SteeringDataset] Warning: {missing_count} images not found")
    
    def _create_valid_indices(self):
        """
        Create list of valid sample indices.
        
        A sample at index i is valid if we have frames at [i-2, i-1, i].
        We skip the first (temporal_length - 1) frames.
        """
        self.valid_indices = []
        
        for i in range(self.temporal_length - 1, len(self.annotations)):
            # Check if all required images exist
            all_exist = True
            for offset in range(self.temporal_length):
                idx = i - (self.temporal_length - 1 - offset)
                timestamp = self.annotations[idx]['timestamp']
                img_path = os.path.join(self.image_dir, f"{timestamp}.jpg")
                if not os.path.exists(img_path):
                    all_exist = False
                    break
            
            if all_exist:
                self.valid_indices.append(i)
    
    def _apply_split(self):
        """
        Apply train/val split.
        
        Every 10th sample (indices 9, 19, 29, ...) goes to validation.
        Remaining samples go to training.
        """
        if self.split == 'all':
            return  # Keep all samples
        
        train_indices = []
        val_indices = []
        
        for i, idx in enumerate(self.valid_indices):
            if i % 10 == 9:  # Every 10th sample (0-indexed: 9, 19, 29, ...)
                val_indices.append(idx)
            else:
                train_indices.append(idx)
        
        if self.split == 'train':
            self.valid_indices = train_indices
            print(f"[SteeringDataset] Train split: {len(train_indices)} samples")
        elif self.split == 'val':
            self.valid_indices = val_indices
            print(f"[SteeringDataset] Val split: {len(val_indices)} samples")
    
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
    
    def _normalize_angle(self, angle):
        """Normalize steering angle to [-1, 1] range."""
        if not self.normalize_angles:
            return angle
        
        # Clip to expected range
        angle = np.clip(angle, self.angle_range[0], self.angle_range[1])
        
        # Normalize to [-1, 1]
        angle_normalized = 2.0 * (angle - self.angle_range[0]) / \
                          (self.angle_range[1] - self.angle_range[0]) - 1.0
        
        return angle_normalized
    
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
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample: temporal image sequence + steering angle.
        
        Returns:
            images: Tensor of shape [temporal_length, 3, H, W]
            steering_angle: Tensor of shape [1]
        """
        # Get actual annotation index
        ann_idx = self.valid_indices[idx]
        
        # Load temporal sequence of images [t-2, t-1, t]
        images = []
        for offset in range(self.temporal_length):
            frame_idx = ann_idx - (self.temporal_length - 1 - offset)
            timestamp = self.annotations[frame_idx]['timestamp']
            img = self._load_image(timestamp)
            images.append(img)
        
        # Get current steering angle (at time t)
        current_annotation = self.annotations[ann_idx]
        steering_angle = current_annotation['steering_angle']
        
        # Apply steering zero point calibration (always present)
        zero_point = current_annotation.get('steering_zero_point', 0.0)
        steering_angle = steering_angle - zero_point
        
        # Data augmentation
        if self.augment:
            images, steering_angle = self._apply_augmentation(images, steering_angle)
        
        # Normalize steering angle
        steering_angle = self._normalize_angle(steering_angle)
        
        # Transform images to tensors
        image_tensors = [self.transform(img) for img in images]
        
        # Stack to [temporal_length, 3, H, W]
        image_sequence = torch.stack(image_tensors, dim=0)
        
        # Convert steering angle to tensor
        steering_tensor = torch.tensor([steering_angle], dtype=torch.float32)
        
        return image_sequence, steering_tensor
    
    def get_sample_info(self, idx):
        """Get metadata for a sample (useful for debugging)."""
        ann_idx = self.valid_indices[idx]
        
        info = {
            'current_timestamp': self.annotations[ann_idx]['timestamp'],
            'current_angle': self.annotations[ann_idx]['steering_angle'],
            'sequence_timestamps': []
        }
        
        for offset in range(self.temporal_length):
            frame_idx = ann_idx - (self.temporal_length - 1 - offset)
            info['sequence_timestamps'].append(
                self.annotations[frame_idx]['timestamp']
            )
        
        return info


def get_steering_dataloaders(
    dataset_root,
    batch_size=16,
    num_workers=4,
    image_size=(320, 640),
    temporal_length=3,
    augment_train=True,
    val_ratio=0.1
):
    """
    Create train and validation dataloaders from single dataset.
    
    Automatically splits data: every 10th sample goes to validation.
    
    Args:
        dataset_root: Path to dataset directory (contains images/ and steering_angles.json)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size (H, W)
        temporal_length: Number of frames in sequence
        augment_train: Enable augmentation for training
        val_ratio: Validation ratio (default: 0.1 for every 10th sample)
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets with automatic split
    train_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        temporal_length=temporal_length,
        augment=augment_train,
        normalize_angles=True,
        split='train',
        val_ratio=val_ratio
    )
    
    val_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        image_size=image_size,
        temporal_length=temporal_length,
        augment=False,  # No augmentation for validation
        normalize_angles=True,
        split='val',
        val_ratio=val_ratio
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
    """Test the data loader."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_data_steering_network.py <dataset_root>")
        print("Example: python load_data_steering_network.py /path/to/dataset")
        sys.exit(1)
    
    dataset_root = sys.argv[1]
    
    print("Testing SteeringAngleDataset...")
    print(f"Dataset root: {dataset_root}\n")
    
    # Create datasets with split
    print("\n=== Testing Train Split ===")
    train_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        temporal_length=3,
        augment=False,
        split='train'
    )
    
    print("\n=== Testing Val Split ===")
    val_dataset = SteeringAngleDataset(
        dataset_root=dataset_root,
        temporal_length=3,
        augment=False,
        split='val'
    )
    
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")
    print(f"Val ratio: {len(val_dataset) / (len(train_dataset) + len(val_dataset)):.2%}")
    
    # Test first sample from train
    if len(train_dataset) > 0:
        print("\n=== Testing Train Samples ===")
        images, steering = train_dataset[0]
        print(f"Image sequence shape: {images.shape}")  # [3, 3, 320, 640]
        print(f"Steering angle: {steering.item():.4f}")
        
        # Get sample info
        info = train_dataset.get_sample_info(0)
        print(f"Current timestamp: {info['current_timestamp']}")
        print(f"Sequence timestamps: {info['sequence_timestamps']}")
        
        # Test a few more samples
        print("\nRandom train samples:")
        for i in np.random.randint(0, min(len(train_dataset), 100), 3):
            images, steering = train_dataset[i]
            info = train_dataset.get_sample_info(i)
            print(f"  Sample {i}: angle={steering.item():.4f}, "
                  f"timestamp={info['current_timestamp']}")
    
    # Test validation samples
    if len(val_dataset) > 0:
        print("\n=== Testing Val Samples ===")
        for i in range(min(3, len(val_dataset))):
            images, steering = val_dataset[i]
            info = val_dataset.get_sample_info(i)
            print(f"  Sample {i}: angle={steering.item():.4f}, "
                  f"timestamp={info['current_timestamp']}")
    
    print("\nDataset test complete!")

