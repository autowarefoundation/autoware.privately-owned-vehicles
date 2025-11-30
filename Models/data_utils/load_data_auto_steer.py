#!/usr/bin/env python3

import os
import json
import numpy as np
from PIL import Image
from typing import List

class LoadDataAutoSteer():
    def __init__(
        self,
        dataset_roots: List[str],
        temporal_length: int = 3,
    ):
        """
        Args:
            dataset_roots: List of dataset directories (e.g., ['60/', '70/', '80/', '100/'])
                          Each contains images/ and steering_angles.json
            temporal_length: Number of consecutive frames (default: 3 for t-2, t-1, t)
        """
        self.dataset_roots = dataset_roots if isinstance(dataset_roots, list) else [dataset_roots]
        self.temporal_length = temporal_length
        
        # Load annotations from all datasets
        self._load_annotations()
        
        # Split into train/val
        self._split_data()
        
        print(f"Dataset loaded with {self.N_trains} trains and {self.N_vals} vals.")
    
    def _load_annotations(self):
        """Load steering angle annotations from all dataset directories."""
        self.annotations = []
        
        for dataset_root in self.dataset_roots:
            json_path = os.path.join(dataset_root, 'steering_angles.json')
            image_dir = os.path.join(dataset_root, 'images')
            
            with open(json_path, 'r') as f:
                dataset_annotations = json.load(f)
            
            # Add dataset info to each annotation
            for ann in dataset_annotations:
                ann['image_dir'] = image_dir
            
            self.annotations.extend(dataset_annotations)
        
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
            if (set_idx % 10 == 0):
                # Slap it to Val
                self.val_indices.append(set_idx)
                self.N_vals += 1
            else:
                # Slap it to Train
                self.train_indices.append(set_idx)
                self.N_trains += 1
    
    def getItemCount(self):
        """Get sizes of Train/Val sets."""
        return self.N_trains, self.N_vals
    
    def getItem(self, index: int, is_train: bool):
        """
        Get item at index, returning temporal image sequence and steering angle.
        
        Args:
            index: Index in train or val set
            is_train: True for training set, False for validation set
            
        Returns:
            List containing:
                - frame_id: Current frame timestamp
                - images: List of PIL images [t-2, t-1, t]
                - steering_angle: Calibrated steering angle (float)
        """
        if is_train:
            ann_idx = self.train_indices[index]
        else:
            ann_idx = self.val_indices[index]
        
        # Load temporal sequence [t-2, t-1, t]
        images = []
        for offset in range(self.temporal_length):
            frame_idx = ann_idx - (self.temporal_length - 1 - offset)
            timestamp = self.annotations[frame_idx]['timestamp']
            image_dir = self.annotations[frame_idx]['image_dir']
            
            img_path = os.path.join(image_dir, f"{timestamp}.jpg")
            img = Image.open(img_path).convert('RGB')
            images.append(img)
        
        # Get steering angle
        current_annotation = self.annotations[ann_idx]
        frame_id = current_annotation['timestamp']
        steering_angle = current_annotation['steering_angle']
        zero_point = current_annotation['steering_zero_point']
        steering_angle = steering_angle - zero_point
        
        return [
            frame_id,
            images,
            steering_angle,
        ]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_data_steering_network.py <dataset_root1> [dataset_root2] ...")
        print("Example: python load_data_steering_network.py 60/ 70/ 80/ 100/")
        sys.exit(1)
    
    dataset_roots = sys.argv[1:]
    print(f"Loading datasets: {dataset_roots}")
    
    # Create data loader
    data_loader = LoadDataAutoSteer(
        dataset_roots=dataset_roots,
        temporal_length=3
    )
    
    # Get counts
    n_train, n_val = data_loader.getItemCount()
    print(f"\nTrain samples: {n_train}")
    print(f"Val samples: {n_val}")
    
    # Test train sample
    if n_train > 0:
        frame_id, images, steering_angle = data_loader.getItem(0, is_train=True)
        print(f"\nTrain sample:")
        print(f"  Frame ID: {frame_id}")
        print(f"  Images: {len(images)} frames")
        print(f"  Image size: {images[0].size}")
        print(f"  Steering angle: {steering_angle:.4f}")
    
    # Test val sample
    if n_val > 0:
        frame_id, images, steering_angle = data_loader.getItem(0, is_train=False)
        print(f"\nVal sample:")
        print(f"  Frame ID: {frame_id}")
        print(f"  Steering angle: {steering_angle:.4f}")
