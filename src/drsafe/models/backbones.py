"""
Backbone models for DR-SAFE pipeline.

Provides factory functions for creating pretrained backbones using timm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn

from drsafe.utils.logging import get_logger

logger = get_logger()


# Recommended backbones for DR classification
RECOMMENDED_BACKBONES = {
    "efficientnetv2_rw_m": {
        "description": "EfficientNetV2-M (RW variant) - Good balance of speed and accuracy",
        "input_size": 480,
        "features": 1792,
    },
    "efficientnetv2_rw_s": {
        "description": "EfficientNetV2-S (RW variant) - Faster, slightly lower accuracy",
        "input_size": 384,
        "features": 1280,
    },
    "convnext_base": {
        "description": "ConvNeXt-Base - Strong CNN baseline",
        "input_size": 224,
        "features": 1024,
    },
    "convnext_small": {
        "description": "ConvNeXt-Small - Lighter ConvNeXt variant",
        "input_size": 224,
        "features": 768,
    },
    "swin_base_patch4_window7_224": {
        "description": "Swin Transformer Base - Strong vision transformer",
        "input_size": 224,
        "features": 1024,
    },
    "vit_base_patch16_224": {
        "description": "ViT-Base - Classic vision transformer",
        "input_size": 224,
        "features": 768,
    },
    "resnet50": {
        "description": "ResNet-50 - Classic CNN baseline",
        "input_size": 224,
        "features": 2048,
    },
    "tf_efficientnet_b4": {
        "description": "EfficientNet-B4 - Efficient architecture",
        "input_size": 380,
        "features": 1792,
    },
}


def list_available_backbones() -> List[str]:
    """
    List all available backbone models.
    
    Returns:
        List of backbone model names.
    """
    return list(timm.list_models(pretrained=True))


def list_recommended_backbones() -> Dict[str, Dict[str, Any]]:
    """
    List recommended backbones for DR classification.
    
    Returns:
        Dictionary of backbone names to their info.
    """
    return RECOMMENDED_BACKBONES


class BackboneWrapper(nn.Module):
    """
    Wrapper for timm backbones that provides a consistent interface.
    """
    
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        global_pool: str = "avg",
        features_only: bool = False,
    ):
        """
        Initialize the backbone wrapper.
        
        Args:
            model_name: Name of the timm model.
            pretrained: Whether to load pretrained weights.
            drop_rate: Dropout rate.
            drop_path_rate: Drop path rate (stochastic depth).
            global_pool: Global pooling type.
            features_only: Whether to return intermediate features.
        """
        super().__init__()
        
        self.model_name = model_name
        self.features_only = features_only
        
        # Create the model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool if not features_only else "",
        )
        
        # Get the number of output features
        self.num_features = self.model.num_features
        
        logger.info(
            f"Created backbone: {model_name}, "
            f"features: {self.num_features}, "
            f"pretrained: {pretrained}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Feature tensor of shape (B, num_features).
        """
        return self.model(x)
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            List of feature maps at different scales.
        """
        return self.model.forward_features(x)


def create_backbone(
    model_name: str = "efficientnetv2_rw_m",
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    features_only: bool = False,
) -> BackboneWrapper:
    """
    Create a backbone model.
    
    Args:
        model_name: Name of the timm model.
        pretrained: Whether to load pretrained weights.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate (stochastic depth).
        features_only: Whether to return intermediate features.
    
    Returns:
        BackboneWrapper instance.
    """
    return BackboneWrapper(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        features_only=features_only,
    )


def get_backbone_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a backbone model.
    
    Args:
        model_name: Name of the timm model.
    
    Returns:
        Dictionary with model information.
    """
    if model_name in RECOMMENDED_BACKBONES:
        return RECOMMENDED_BACKBONES[model_name]
    
    # Create a dummy model to get info
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    
    return {
        "description": f"{model_name}",
        "input_size": model.default_cfg.get("input_size", (3, 224, 224))[-1],
        "features": model.num_features,
    }


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    This maintains a moving average of model parameters for more stable
    predictions during inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EMA model.
        
        Args:
            model: Model to track.
            decay: Decay rate for the moving average.
            device: Device to store EMA parameters.
        """
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create a copy of the model parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device is not None:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self) -> None:
        """Update EMA parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self) -> None:
        """Apply EMA parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """Restore original parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dictionary."""
        return {k: v.clone() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA state dictionary."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()
