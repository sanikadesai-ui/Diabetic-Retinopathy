"""
Hybrid CNN + ViT model for DR-SAFE pipeline.

Implements feature fusion between CNN and Vision Transformer backbones
for improved representation learning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drsafe.models.backbones import create_backbone
from drsafe.models.heads import MultiTaskHead
from drsafe.utils.logging import get_logger

logger = get_logger()


class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion module.
    
    Uses cross-attention to fuse features from CNN and ViT backbones.
    """
    
    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize attention fusion.
        
        Args:
            cnn_dim: CNN feature dimension.
            vit_dim: ViT feature dimension.
            hidden_dim: Hidden dimension for fusion.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # Project both features to common dimension
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        
        # Cross-attention: CNN queries, ViT keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        vit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cnn_features: CNN features of shape (B, cnn_dim).
            vit_features: ViT features of shape (B, vit_dim).
        
        Returns:
            Fused features of shape (B, hidden_dim).
        """
        # Project features
        cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # (B, 1, hidden_dim)
        vit_proj = self.vit_proj(vit_features).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Concatenate for key/value
        kv = torch.cat([cnn_proj, vit_proj], dim=1)  # (B, 2, hidden_dim)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(cnn_proj, kv, kv)
        
        # Residual and norm
        x = self.norm1(cnn_proj + attn_out)
        
        # Feedforward
        x = self.norm2(x + self.ffn(x))
        
        return x.squeeze(1)  # (B, hidden_dim)


class GatedFusion(nn.Module):
    """
    Gated feature fusion module.
    
    Uses learnable gates to combine CNN and ViT features.
    """
    
    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        hidden_dim: int = 512,
    ):
        """
        Initialize gated fusion.
        
        Args:
            cnn_dim: CNN feature dimension.
            vit_dim: ViT feature dimension.
            hidden_dim: Output dimension.
        """
        super().__init__()
        
        # Project to common dimension
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        
        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        vit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cnn_features: CNN features of shape (B, cnn_dim).
            vit_features: ViT features of shape (B, vit_dim).
        
        Returns:
            Fused features of shape (B, hidden_dim).
        """
        # Project features
        cnn_proj = self.cnn_proj(cnn_features)
        vit_proj = self.vit_proj(vit_features)
        
        # Compute gate
        combined = torch.cat([cnn_features, vit_features], dim=1)
        gate = self.gate(combined)
        
        # Gated fusion
        fused = gate * cnn_proj + (1 - gate) * vit_proj
        
        return fused


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based feature fusion.
    """
    
    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize concatenation fusion.
        
        Args:
            cnn_dim: CNN feature dimension.
            vit_dim: ViT feature dimension.
            hidden_dim: Output dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        cnn_features: torch.Tensor,
        vit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            cnn_features: CNN features of shape (B, cnn_dim).
            vit_features: ViT features of shape (B, vit_dim).
        
        Returns:
            Fused features of shape (B, hidden_dim).
        """
        combined = torch.cat([cnn_features, vit_features], dim=1)
        return self.fusion(combined)


class HybridModel(nn.Module):
    """
    Hybrid CNN + ViT model for diabetic retinopathy classification.
    
    Combines features from a CNN backbone (e.g., EfficientNet) and a
    Vision Transformer (e.g., ViT) using various fusion strategies.
    """
    
    def __init__(
        self,
        cnn_backbone: str = "efficientnetv2_rw_m",
        vit_backbone: str = "vit_base_patch16_224",
        fusion_method: str = "concat",  # concat, attention, gated
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
        use_ordinal: bool = False,
    ):
        """
        Initialize the hybrid model.
        
        Args:
            cnn_backbone: Name of the CNN backbone.
            vit_backbone: Name of the ViT backbone.
            fusion_method: Feature fusion method.
            num_classes: Number of severity classes.
            hidden_dim: Hidden dimension for fusion.
            dropout: Dropout rate.
            pretrained: Whether to use pretrained weights.
            use_ordinal: Whether to use ordinal regression.
        """
        super().__init__()
        
        self.cnn_backbone_name = cnn_backbone
        self.vit_backbone_name = vit_backbone
        self.fusion_method = fusion_method
        
        # Create backbones
        self.cnn = create_backbone(
            model_name=cnn_backbone,
            pretrained=pretrained,
            drop_rate=dropout,
        )
        
        self.vit = create_backbone(
            model_name=vit_backbone,
            pretrained=pretrained,
            drop_rate=dropout,
        )
        
        cnn_dim = self.cnn.num_features
        vit_dim = self.vit.num_features
        
        logger.info(
            f"Created hybrid model: CNN ({cnn_backbone}, {cnn_dim}d) + "
            f"ViT ({vit_backbone}, {vit_dim}d), fusion: {fusion_method}"
        )
        
        # Create fusion module
        if fusion_method == "attention":
            self.fusion = AttentionFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_method == "gated":
            self.fusion = GatedFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                hidden_dim=hidden_dim,
            )
        else:  # concat
            self.fusion = ConcatFusion(
                cnn_dim=cnn_dim,
                vit_dim=vit_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        
        # Create classification head
        self.head = MultiTaskHead(
            in_features=self.fusion.output_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_ordinal=use_ordinal,
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
        # Extract features from both backbones
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        
        # Fuse features
        fused = self.fusion(cnn_features, vit_features)
        
        # Classification
        severity_logits, referable_logits = self.head(fused)
        
        return severity_logits, referable_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fused features before classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Fused features of shape (B, hidden_dim).
        """
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        return self.fusion(cnn_features, vit_features)


def create_hybrid_model(
    cnn_backbone: str = "efficientnetv2_rw_m",
    vit_backbone: str = "vit_base_patch16_224",
    fusion_method: str = "concat",
    num_classes: int = 5,
    hidden_dim: int = 512,
    dropout: float = 0.3,
    pretrained: bool = True,
    use_ordinal: bool = False,
) -> HybridModel:
    """
    Factory function to create a hybrid model.
    
    Args:
        cnn_backbone: Name of the CNN backbone.
        vit_backbone: Name of the ViT backbone.
        fusion_method: Feature fusion method.
        num_classes: Number of severity classes.
        hidden_dim: Hidden dimension for fusion.
        dropout: Dropout rate.
        pretrained: Whether to use pretrained weights.
        use_ordinal: Whether to use ordinal regression.
    
    Returns:
        HybridModel instance.
    """
    return HybridModel(
        cnn_backbone=cnn_backbone,
        vit_backbone=vit_backbone,
        fusion_method=fusion_method,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained=pretrained,
        use_ordinal=use_ordinal,
    )
