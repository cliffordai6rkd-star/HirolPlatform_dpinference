from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)

class ResNetEncoder(nn.Module):
    OUT_DIM_BY_NAME = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }

    def __init__(self,
        name: str = "resnet18",
        pretrained: bool = True,
        trainable: bool = True,
        in_channels: int = 3):
        super().__init__()
        if name not in self.OUT_DIM_BY_NAME:
            raise ValueError(f"Unsupported resnet name: {name}")
        self.out_dim = self.OUT_DIM_BY_NAME[name]

        # Build
        backbone_fn = getattr(models, name)
        self.backbone = backbone_fn(weights=(
            models.ResNet18_Weights.DEFAULT if name == "resnet18" and pretrained else
            models.ResNet34_Weights.DEFAULT if name == "resnet34" and pretrained else
            models.ResNet50_Weights.DEFAULT if name == "resnet50" and pretrained else
            models.ResNet101_Weights.DEFAULT if name == "resnet101" and pretrained else
            models.ResNet152_Weights.DEFAULT if name == "resnet152" and pretrained else
            None
        ))
        
        # Replace first conv if in_channels != 3
        if in_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding,
            bias=old.bias is not None)
        
        # Remove FC head -> features only
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
        )

        if not trainable:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, out_dim)"""
        return self.feature_extractor(x)
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate=0.25,
                 use_layer_norm=True,
                 hidden_dims: List[int]= [512,512,512]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hdim))
            if i+1 < len(hidden_dims):
                if dropout_rate is not None:
                    layers.append(nn.Dropout(p=dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hdim))
                # activations
                layers.append(nn.SiLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        # Action are normalized between -1 and 1
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.network(x)
        