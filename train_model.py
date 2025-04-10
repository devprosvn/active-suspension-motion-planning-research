#!/usr/bin/env python
"""
Training Script for MobileViT Model

This script trains the MobileViT model for terrain classification and depth estimation
using the collected dataset.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from models.mobilevit_model import TerrainDepthMobileViT, TerrainDepthLoss
from data_processing.dataset import create_dataloaders

# Training parameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SIZE = 's'  # 's', 'xs', or 'xxs'
NUM_TERRAIN_CLASSES = 5

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_model():
    """Main training function"""
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TerrainDepthMobileViT(
        num_terrain_classes=NUM_TERRAIN_CLASSES,
        pretrained=True,
        model_size=MODEL_SIZE
    ).to(device)
    
    criterion = TerrainDepthLoss(terrain_weight=1.0, depth_weight=1.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(BASE_DIR, batch_size=BATCH_SIZE)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(LOG_DIR)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            depth_maps = batch['depth'].to(device)
            terrain_labels = batch['terrain_label'].to(device)
            
            # Forward pass
            terrain_pred, depth_pred = model(images)
            
            # Compute loss
            loss, terrain_loss, depth_loss = criterion(
                terrain_pred, terrain_labels, depth_pred, depth_maps
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Terrain Loss: {terrain_loss.item():.4f}, '
                      f'Depth Loss: {depth_loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                depth_maps = batch['depth'].to(device)
                terrain_labels = batch['terrain_label'].to(device)
                
                terrain_pred, depth_pred = model(images)
                loss, _, _ = criterion(
                    terrain_pred, terrain_labels, depth_pred, depth_maps
                )
                val_loss += loss.item()
        
        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
    
    writer.close()
    print('Training completed!')

if __name__ == '__main__':
    train_model()