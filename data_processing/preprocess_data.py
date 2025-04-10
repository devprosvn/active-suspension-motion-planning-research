#!/usr/bin/env python

"""
Data Preprocessing Script for Terrain Recognition and Depth Estimation

This script preprocesses the collected image data and depth maps for training:
- Resizes images to a standard size
- Normalizes pixel values
- Creates train/validation/test splits
- Generates dataset metadata
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')

# Create processed data directories
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images')
PROCESSED_DEPTH_DIR = os.path.join(PROCESSED_DIR, 'depth')
METADATA_DIR = os.path.join(PROCESSED_DIR, 'metadata')

# Standard image size for model input
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

# Terrain type mapping (for one-hot encoding)
TERRAIN_TYPES = ['asphalt', 'gravel', 'sand', 'snow', 'bumpy']
TERRAIN_MAPPING = {terrain: idx for idx, terrain in enumerate(TERRAIN_TYPES)}


def create_directories():
    """Create necessary directories for processed data"""
    os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DEPTH_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    # Create subdirectories for train/val/test splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(PROCESSED_IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DEPTH_DIR, split), exist_ok=True)


def preprocess_image(image_path, target_size=(TARGET_WIDTH, TARGET_HEIGHT)):
    """Preprocess a single RGB image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized


def preprocess_depth_map(depth_path, target_size=(TARGET_WIDTH, TARGET_HEIGHT)):
    """Preprocess a single depth map"""
    # Read depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        print(f"Warning: Could not read depth map {depth_path}")
        return None
    
    # Resize depth map
    depth_resized = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize depth values to [0, 1]
    # Note: This assumes depth values are already in a reasonable range
    depth_min = np.min(depth_resized)
    depth_max = np.max(depth_resized)
    if depth_max > depth_min:
        depth_normalized = (depth_resized - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_resized, dtype=np.float32)
    
    return depth_normalized


def process_dataset(args):
    """Process the entire dataset"""
    # Create necessary directories
    create_directories()
    
    # Read labels file
    labels_file = os.path.join(DATASET_DIR, 'labels.csv')
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at {labels_file}")
        return
    
    labels_df = pd.read_csv(labels_file)
    print(f"Found {len(labels_df)} samples in the dataset")
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(
        labels_df, test_size=0.3, stratify=labels_df['terrain_type'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['terrain_type'], random_state=42
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Save split metadata
    train_df.to_csv(os.path.join(METADATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(METADATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(METADATA_DIR, 'test.csv'), index=False)
    
    # Process each split
    process_split(train_df, 'train', args)
    process_split(val_df, 'val', args)
    process_split(test_df, 'test', args)
    
    print("Dataset preprocessing completed!")


def process_split(df, split_name, args):
    """Process a specific data split (train/val/test)"""
    print(f"\nProcessing {split_name} split...")
    
    # Create output directories for this split
    images_output_dir = os.path.join(PROCESSED_IMAGES_DIR, split_name)
    depth_output_dir = os.path.join(PROCESSED_DEPTH_DIR, split_name)
    
    # Process each sample
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} data"):
        image_id = row['image_id']
        terrain_type = row['terrain_type']
        depth_map_id = row['depth_map_id']
        
        # Construct file paths
        image_path = os.path.join(DATASET_DIR, 'images', f"{image_id}.png")
        depth_path = os.path.join(DATASET_DIR, 'depth', f"{depth_map_id}.png")
        
        # Skip if files don't exist
        if not os.path.exists(image_path) or not os.path.exists(depth_path):
            print(f"Warning: Missing files for sample {image_id}")
            continue
        
        # Preprocess image and depth map
        processed_image = preprocess_image(image_path)
        processed_depth = preprocess_depth_map(depth_path)
        
        if processed_image is None or processed_depth is None:
            continue
        
        # Save processed files
        output_image_path = os.path.join(images_output_dir, f"{image_id}.npy")
        output_depth_path = os.path.join(depth_output_dir, f"{depth_map_id}.npy")
        
        np.save(output_image_path, processed_image)
        np.save(output_depth_path, processed_depth)
        
        # Add to processed data list
        processed_data.append({
            'image_id': image_id,
            'terrain_type': terrain_type,
            'terrain_class': TERRAIN_MAPPING[terrain_type],
            'depth_map_id': depth_map_id,
            'image_path': output_image_path,
            'depth_path': output_depth_path
        })
    
    # Save processed metadata
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(os.path.join(METADATA_DIR, f"{split_name}_processed.csv"), index=False)
    
    print(f"Processed {len(processed_data)} samples for {split_name} split")


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for terrain recognition')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize some processed samples for verification')
    args = parser.parse_args()
    
    process_dataset(args)
    
    if args.visualize:
        # Implement visualization code here if needed
        pass


if __name__ == '__main__':
    main()