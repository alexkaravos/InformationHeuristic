"""
General resnet backbone models for representation learning
"""

import torch
import torch.nn as nn
from torchvision import models
class ResNetSSL(nn.Module):
    """
    General ResNet architecture for self-supervised learning (SimCLR, Barlow Twins, etc.)
    Supports different ResNet variants and input channels.
    """
    def __init__(self, arch='resnet18', in_channels=3, proj_dim=128,
                small_images=False, final_batchnorm=False): 
        super(ResNetSSL, self).__init__()
        
        if arch == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            backbone_dim = 512
        elif arch == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            backbone_dim = 2048
            
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        hidden_dim = backbone_dim*2
        
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            print("Number of input channels: ", in_channels)
        if small_images:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
        
        self.backbone.fc = nn.Identity()
        
        # Modified projection head to conditionally add final BN
        projection_layers = [
            nn.Linear(backbone_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        ]
        if final_batchnorm:
            projection_layers.append(nn.BatchNorm1d(proj_dim))
        
        self.projection_head = nn.Sequential(*projection_layers)
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        projections = self.projection_head(features)
        
        if return_features:
            return features, projections
        return projections
