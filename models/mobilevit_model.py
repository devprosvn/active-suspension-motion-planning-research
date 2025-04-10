#!/usr/bin/env python

"""
MobileViT Model with Depth Estimation

This module implements the MobileViT architecture with an additional
branch for depth estimation, using the timm library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs


class TerrainDepthMobileViT(nn.Module):
    """MobileViT model with terrain classification and depth estimation branches"""
    
    def __init__(self, num_terrain_classes=5, pretrained=True, model_size='s'):
        """
        Initialize the model
        
        Args:
            num_terrain_classes: Number of terrain classes to predict
            pretrained: Whether to use pretrained weights
            model_size: MobileViT model size ('s', 'xs', or 'xxs')
        """
        super(TerrainDepthMobileViT, self).__init__()
        
        # Select MobileViT backbone based on model_size
        if model_size == 's':
            self.backbone = mobilevit_s(pretrained=pretrained)
            feature_dim = 640
        elif model_size == 'xs':
            self.backbone = mobilevit_xs(pretrained=pretrained)
            feature_dim = 384
        elif model_size == 'xxs':
            self.backbone = mobilevit_xxs(pretrained=pretrained)
            feature_dim = 320
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from 's', 'xs', or 'xxs'")
        
        # Remove the original classification head
        self.backbone.head = nn.Identity()
        
        # Terrain classification branch
        self.terrain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_terrain_classes)
        )
        
        # Depth estimation branch
        # Using a decoder-like structure to upsample features for dense prediction
        self.depth_decoder = nn.ModuleList([
            # First upsampling block
            nn.Sequential(
                nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            # Second upsampling block
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            # Third upsampling block
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ),
            # Final prediction layer
            nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()  # Normalize depth predictions to [0, 1]
            )
        ])
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            terrain_logits: Logits for terrain classification
            depth_pred: Predicted depth map
        """
        # Extract features from the backbone
        features = self.backbone.forward_features(x)
        
        # Terrain classification branch
        terrain_logits = self.terrain_classifier(features)
        
        # Depth estimation branch
        depth_features = features
        for decoder_block in self.depth_decoder:
            depth_features = decoder_block(depth_features)
        
        # Ensure depth map is the right size (same as input)
        depth_pred = F.interpolate(depth_features, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return terrain_logits, depth_pred


class TerrainDepthLoss(nn.Module):
    """Combined loss function for terrain classification and depth estimation"""
    
    def __init__(self, terrain_weight=1.0, depth_weight=1.0):
        """
        Initialize the loss function
        
        Args:
            terrain_weight: Weight for the terrain classification loss
            depth_weight: Weight for the depth estimation loss
        """
        super(TerrainDepthLoss, self).__init__()
        self.terrain_weight = terrain_weight
        self.depth_weight = depth_weight
        
        # Loss functions
        self.terrain_loss_fn = nn.CrossEntropyLoss()
        self.depth_loss_fn = nn.L1Loss()  # Mean Absolute Error for depth
    
    def forward(self, terrain_pred, terrain_target, depth_pred, depth_target):
        """
        Compute the combined loss
        
        Args:
            terrain_pred: Predicted terrain logits
            terrain_target: Ground truth terrain labels
            depth_pred: Predicted depth maps
            depth_target: Ground truth depth maps
            
        Returns:
            total_loss: Combined weighted loss
            terrain_loss: Classification loss component
            depth_loss: Depth estimation loss component
        """
        # Compute terrain classification loss
        terrain_loss = self.terrain_loss_fn(terrain_pred, terrain_target)
        
        # Compute depth estimation loss
        # Ensure depth_pred and depth_target have the same shape
        depth_pred = depth_pred.squeeze(1)  # Remove channel dimension if present
        depth_loss = self.depth_loss_fn(depth_pred, depth_target)
        
        # Compute total loss
        total_loss = self.terrain_weight * terrain_loss + self.depth_weight * depth_loss
        
        return total_loss, terrain_loss, depth_loss


def create_model(num_terrain_classes=5, pretrained=True, model_size='s'):
    """
    Create a TerrainDepthMobileViT model
    
    Args:
        num_terrain_classes: Number of terrain classes to predict
        pretrained: Whether to use pretrained weights
        model_size: MobileViT model size ('s', 'xs', or 'xxs')
        
    Returns:
        model: Initialized model
    """
    model = TerrainDepthMobileViT(
        num_terrain_classes=num_terrain_classes,
        pretrained=pretrained,
        model_size=model_size
    )
    return model