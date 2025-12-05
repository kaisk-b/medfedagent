"""
MedFedAgent Model Architectures

CNN-based chest X-ray classifiers including DenseNet-121 and ResNet-18
with support for multi-label and binary classification.

All models are designed to be Opacus-compatible for differential privacy training.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Literal, Tuple
from loguru import logger

# Try to import Opacus for model validation
try:
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logger.warning("Opacus not available. Model validation for DP will be skipped.")


def replace_batchnorm_with_groupnorm(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    Recursively replace all BatchNorm layers with GroupNorm.
    
    GroupNorm is compatible with Opacus while BatchNorm is not,
    because BatchNorm computes statistics across the batch dimension.
    
    Args:
        module: PyTorch module to modify
        num_groups: Number of groups for GroupNorm
        
    Returns:
        Modified module with GroupNorm instead of BatchNorm
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            # Ensure num_groups divides num_channels
            actual_groups = min(num_groups, num_channels)
            while num_channels % actual_groups != 0:
                actual_groups -= 1
            
            # Create GroupNorm with same number of features
            group_norm = nn.GroupNorm(
                num_groups=actual_groups,
                num_channels=num_channels,
                eps=child.eps,
                affine=child.affine
            )
            
            # Copy weights if they exist
            if child.affine and child.weight is not None:
                group_norm.weight.data = child.weight.data.clone()
                group_norm.bias.data = child.bias.data.clone()
            
            setattr(module, name, group_norm)
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)
    
    return module


def replace_inplace_relu(module: nn.Module) -> nn.Module:
    """
    Recursively replace all inplace ReLU operations with non-inplace versions.
    
    Opacus requires non-inplace operations for proper gradient computation.
    
    Args:
        module: PyTorch module to modify
        
    Returns:
        Modified module with non-inplace ReLU
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_inplace_relu(child)
    
    return module


def make_opacus_compatible(model: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    Make a model compatible with Opacus DP-SGD training.
    
    This function:
    1. Replaces BatchNorm with GroupNorm
    2. Replaces inplace ReLU with non-inplace
    3. Validates the model with Opacus (if available)
    
    Args:
        model: PyTorch model to modify
        num_groups: Number of groups for GroupNorm
        
    Returns:
        Opacus-compatible model
    """
    logger.info("Making model Opacus-compatible...")
    
    # Replace BatchNorm with GroupNorm
    model = replace_batchnorm_with_groupnorm(model, num_groups)
    
    # Replace inplace ReLU
    model = replace_inplace_relu(model)
    
    # Validate with Opacus if available
    if OPACUS_AVAILABLE:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning(f"Opacus validation warnings after fix: {errors}")
            # Apply Opacus's own fix as a fallback
            model = ModuleValidator.fix(model)
        else:
            logger.info("Model is now Opacus-compatible")
    
    return model


def validate_model_for_dp(model: nn.Module) -> Tuple[bool, list]:
    """
    Validate if a model is compatible with Opacus DP-SGD.
    
    Args:
        model: PyTorch model to validate
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    if not OPACUS_AVAILABLE:
        logger.warning("Opacus not available for validation")
        return True, []
    
    errors = ModuleValidator.validate(model, strict=False)
    return len(errors) == 0, errors


class ChestXrayClassifier(nn.Module):
    """
    Chest X-ray classifier based on pretrained CNN architectures.
    
    Supports:
    - DenseNet-121 (default, best for medical imaging)
    - ResNet-18 (faster, smaller)
    
    Can be used for:
    - Binary classification (Normal vs Abnormal)
    - Multi-label classification (14 pathologies from ChestX-ray14)
    
    Note: When opacus_compatible=True, BatchNorm layers are replaced with
    GroupNorm and inplace operations are disabled for DP-SGD compatibility.
    """
    
    def __init__(
        self,
        model_name: Literal["densenet121", "resnet18"] = "densenet121",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        opacus_compatible: bool = True
    ):
        """
        Initialize the chest X-ray classifier.
        
        Args:
            model_name: Architecture to use ('densenet121' or 'resnet18')
            num_classes: Number of output classes (2 for binary, 14 for multi-label)
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before final layer
            freeze_backbone: Whether to freeze pretrained layers
            opacus_compatible: Whether to make model compatible with Opacus DP-SGD
        """
        super(ChestXrayClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.opacus_compatible = opacus_compatible
        
        # Build the model
        if model_name == "densenet121":
            self._build_densenet(pretrained, freeze_backbone)
        elif model_name == "resnet18":
            self._build_resnet(pretrained, freeze_backbone)
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'densenet121' or 'resnet18'")
        
        # Make Opacus-compatible if requested
        if opacus_compatible:
            self.backbone = make_opacus_compatible(self.backbone)
        
        logger.info(f"Initialized {model_name} with {num_classes} classes, dropout={dropout}, opacus_compatible={opacus_compatible}")
    
    def _build_densenet(self, pretrained: bool, freeze_backbone: bool):
        """Build DenseNet-121 based classifier."""
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier.in_features  # 1024 for DenseNet121
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_features, self.num_classes)
        )
        
        self.feature_dim = num_features
    
    def _build_resnet(self, pretrained: bool, freeze_backbone: bool):
        """Build ResNet-18 based classifier."""
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        
        # Get the number of input features for the classifier
        num_features = self.backbone.fc.in_features  # 512 for ResNet18
        
        # Replace classifier with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_features, self.num_classes)
        )
        
        self.feature_dim = num_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature tensor
        """
        if self.model_name == "densenet121":
            features = self.backbone.features(x)
            features = nn.functional.relu(features, inplace=True)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        else:
            # ResNet feature extraction
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            features = torch.flatten(x, 1)
        
        return features
    
    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "total": total,
            "frozen": total - trainable
        }


class SimpleCNN(nn.Module):
    """
    A simple CNN for testing and fallback scenarios.
    Much faster than DenseNet but less accurate.
    Designed to be Opacus-compatible (no BatchNorm, no inplace operations).
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),  # GroupNorm instead of BatchNorm for Opacus
            nn.ReLU(inplace=False),  # inplace=False for Opacus compatibility
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self.feature_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(
    model_name: str = "densenet121",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    opacus_compatible: bool = True
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        freeze_backbone: Whether to freeze pretrained layers
        opacus_compatible: Whether to make model compatible with Opacus DP-SGD
            (replaces BatchNorm with GroupNorm, disables inplace operations)
        
    Returns:
        PyTorch model (Opacus-compatible if requested)
    """
    if model_name in ["densenet121", "resnet18"]:
        return ChestXrayClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            opacus_compatible=opacus_compatible
        )
    elif model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


if __name__ == "__main__":
    # Quick test
    print("Testing model architectures...")
    
    for model_name in ["densenet121", "resnet18", "simple_cnn"]:
        print(f"\n--- {model_name} ---")
        model = create_model(model_name=model_name, num_classes=2, opacus_compatible=True)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Model size: {get_model_size_mb(model):.2f} MB")
        
        if hasattr(model, "count_parameters"):
            params = model.count_parameters()
            print(f"Parameters: {params}")
        
        # Validate Opacus compatibility
        is_valid, errors = validate_model_for_dp(model)
        if is_valid:
            print("✅ Opacus-compatible")
        else:
            print(f"❌ Not Opacus-compatible: {errors}")
