"""
Model definitions for wildlife species identification.

Supports:
  - custom_cnn      : Lightweight CNN built from scratch
  - resnet50        : ResNet-50 with ImageNet pre-training
  - efficientnet_b3 : EfficientNet-B3 with ImageNet pre-training
  - vit_b_16        : Vision Transformer (ViT-B/16) via timm
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm


# ---------------------------------------------------------------------------
# Custom CNN
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    5-block custom CNN for 224×224 inputs.
    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # → 112
            ConvBlock(32,  64,  pool=True),   # → 56
            ConvBlock(64,  128, pool=True),   # → 28
            ConvBlock(128, 256, pool=True),   # → 14
            ConvBlock(256, 512, pool=True),   # → 7
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Transfer Learning Wrappers
# ---------------------------------------------------------------------------

class WildlifeClassifier(nn.Module):
    """
    Unified wrapper around any backbone supported by torchvision or timm.
    Replaces the classification head with a custom head for `num_classes`.
    """

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.architecture = architecture
        self.backbone, feature_dim = self._build_backbone(
            architecture, pretrained
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    # ------------------------------------------------------------------
    def _build_backbone(
        self, architecture: str, pretrained: bool
    ) -> tuple[nn.Module, int]:
        weights_flag = "IMAGENET1K_V1" if pretrained else None

        if architecture == "resnet50":
            model = tv_models.resnet50(weights=weights_flag)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim

        elif architecture == "efficientnet_b3":
            model = tv_models.efficientnet_b3(weights=weights_flag)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

        elif architecture == "vit_b_16":
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,  # remove timm head
            )
            feature_dim = model.num_features
            return model, feature_dim

        elif architecture == "custom_cnn":
            raise ValueError(
                "Use CustomCNN directly for 'custom_cnn' architecture."
            )

        else:
            # Generic timm fallback
            model = timm.create_model(
                architecture, pretrained=pretrained, num_classes=0
            )
            feature_dim = model.num_features
            return model, feature_dim

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the correct model from config."""
    arch = cfg["model"]["architecture"]
    num_classes = cfg["model"]["num_classes"]
    pretrained = cfg["model"]["pretrained"]
    dropout = cfg["model"]["dropout"]
    freeze = cfg["model"]["freeze_backbone"]

    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=dropout)

    return WildlifeClassifier(
        architecture=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze,
        dropout=dropout,
    )
