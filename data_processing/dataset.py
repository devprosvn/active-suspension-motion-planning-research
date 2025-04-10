#!/usr/bin/env python

"""
Dataset Classes for Terrain Recognition and Depth Estimation

This module provides PyTorch Dataset classes for loading and processing
the terrain and depth data for training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2


class TerrainDepthDataset(Dataset):
    """Dataset for terrain classification and depth estimation"""
    
    def __init__(self, metadata_file, transform=None):
        """
        Initialize the dataset
        
        Args:
            metadata_file: Path to the CSV file containing dataset metadata
            transform: Optional transforms to apply to images
        """
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        
        # Verify that all files exist
        valid_samples = []
        for idx, row in self.metadata.iterrows():
            image_path = row['image_path']
            depth_path = row['depth_path']
            
            if os.path.exists(image_path) and os.path.exists(depth_path):
                valid_samples.append(row)
            else:
                print(f"Warning: Missing files for sample {row['image_id']}")
        
        self.metadata = pd.DataFrame(valid_samples)
        print(f"Loaded {len(self.metadata)} valid samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get metadata for this sample
        sample = self.metadata.iloc[idx]
        
        # Load image and depth map
        image = np.load(sample['image_path'])
        depth_map = np.load(sample['depth_path'])
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
        depth_map = torch.from_numpy(depth_map).float()
        
        # Get terrain class label
        terrain_class = sample['terrain_class']
        terrain_label = torch.tensor(terrain_class, dtype=torch.long)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'depth': depth_map,
            'terrain_label': terrain_label,
            'image_id': sample['image_id']
        }


def get_data_transforms():
    """Get data transforms for training and validation"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transforms for validation/testing (no augmentation)
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_dataloaders(base_dir, batch_size=32, num_workers=4):
    """Create DataLoaders for training, validation, and testing"""
    # Get data transforms
    train_transform, val_transform = get_data_transforms()
    
    # Metadata file paths
    metadata_dir = os.path.join(base_dir, 'processed_data', 'metadata')
    train_metadata = os.path.join(metadata_dir, 'train_processed.csv')
    val_metadata = os.path.join(metadata_dir, 'val_processed.csv')
    test_metadata = os.path.join(metadata_dir, 'test_processed.csv')
    
    # Create datasets
    train_dataset = TerrainDepthDataset(train_metadata, transform=train_transform)
    val_dataset = TerrainDepthDataset(val_metadata, transform=val_transform)
    test_dataset = TerrainDepthDataset(test_metadata, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader