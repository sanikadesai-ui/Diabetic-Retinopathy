"""
Main model class for DR-SAFE pipeline.

Provides a unified interface for single-backbone and hybrid models.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from drsafe.models.backbones import create_backbone, EMAModel
from drsafe.models.heads import MultiTaskHead
from drsafe.models.hybrid import HybridModel, create_hybrid_model
from drsafe.utils.config import Config, ModelConfig
from drsafe.utils.logging import get_logger

logger = get_logger()


class DRModel(nn.Module):
    """
    Main model for Diabetic Retinopathy classification.
    
    Supports both single-backbone and hybrid (CNN + ViT) architectures
    with multi-task heads for severity and referable classification.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnetv2_rw_m",
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        pretrained: bool = True,
        use_ordinal: bool = False,
        ordinal_method: str = "coral",
    ):
        """
        Initialize the DR model.
        
        Args:
            backbone: Name of the backbone model.
            num_classes: Number of severity classes (default 5).
            hidden_dim: Hidden dimension for the head.
            dropout: Dropout rate.
            drop_path_rate: Drop path rate (stochastic depth).
            pretrained: Whether to use pretrained backbone weights.
            use_ordinal: Whether to use ordinal regression for severity.
            ordinal_method: Ordinal method ("coral" or "corn").
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal
        
        # Create backbone
        self.backbone = create_backbone(
            model_name=backbone,
            pretrained=pretrained,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate,
        )
        
        # Create multi-task head
        self.head = MultiTaskHead(
            in_features=self.backbone.num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_ordinal=use_ordinal,
            ordinal_method=ordinal_method,
        )
        
        logger.info(
            f"Created DRModel: backbone={backbone}, features={self.backbone.num_features}, "
            f"classes={num_classes}, ordinal={use_ordinal}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Tuple of (severity_logits, referable_logits).
        """
        features = self.backbone(x)
        severity_logits, referable_logits = self.head(features)
        
        return severity_logits, referable_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get backbone features.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Features of shape (B, num_features).
        """
        return self.backbone(x)
    
    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Dictionary containing:
                - severity_logits: Raw severity logits
                - severity_probs: Severity class probabilities
                - severity_pred: Predicted severity class
                - referable_logits: Raw referable logits
                - referable_probs: Referable probability
                - referable_pred: Predicted referable status
        """
        severity_logits, referable_logits = self.forward(x)
        
        # Compute probabilities
        if self.use_ordinal:
            severity_probs = self.head.severity_head.predict_proba(severity_logits)
        else:
            severity_probs = torch.softmax(severity_logits, dim=1)
        
        referable_probs = torch.sigmoid(referable_logits)
        
        return {
            "severity_logits": severity_logits,
            "severity_probs": severity_probs,
            "severity_pred": severity_probs.argmax(dim=1),
            "referable_logits": referable_logits,
            "referable_probs": referable_probs.squeeze(-1),
            "referable_pred": (referable_probs.squeeze(-1) > 0.5).long(),
        }
    
    def enable_mc_dropout(self) -> None:
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_mc_dropout(self) -> None:
        """Disable dropout layers (return to eval mode)."""
        self.eval()


def create_model(
    config: Union[Config, ModelConfig],
    num_classes: int = 5,
) -> Union[DRModel, HybridModel]:
    """
    Create a model from configuration.
    
    Args:
        config: Configuration object (Config or ModelConfig).
        num_classes: Number of severity classes.
    
    Returns:
        Model instance (DRModel or HybridModel).
    """
    if isinstance(config, Config):
        model_config = config.model
    else:
        model_config = config
    
    if model_config.use_hybrid:
        return create_hybrid_model(
            cnn_backbone=model_config.backbone,
            vit_backbone=model_config.vit_backbone,
            fusion_method=model_config.fusion_method,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=model_config.drop_rate,
            pretrained=model_config.pretrained,
            use_ordinal=model_config.use_ordinal,
        )
    else:
        return DRModel(
            backbone=model_config.backbone,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=model_config.drop_rate,
            drop_path_rate=model_config.drop_path_rate,
            pretrained=model_config.pretrained,
            use_ordinal=model_config.use_ordinal,
        )


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Config] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Union[DRModel, HybridModel]:
    """
    Load a model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        config: Optional configuration (if not saved in checkpoint).
        map_location: Device to map tensors to.
        strict: Whether to strictly enforce state dict keys match.
    
    Returns:
        Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Try to get config from checkpoint
    if config is None:
        if "config" in checkpoint:
            config = Config.from_dict(checkpoint["config"])
        else:
            raise ValueError("Config not found in checkpoint and not provided")
    
    # Create model
    model = create_model(config)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 512, 512)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: Model instance.
        input_size: Input tensor size (B, C, H, W).
    
    Returns:
        Model summary string.
    """
    from collections import OrderedDict
    
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    total_params, trainable_params = count_parameters(model)
    
    summary_lines = [
        "=" * 60,
        f"Model: {model.__class__.__name__}",
        "=" * 60,
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable parameters: {total_params - trainable_params:,}",
        "=" * 60,
    ]
    
    return "\n".join(summary_lines)
