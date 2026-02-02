"""
Loss functions for DR-SAFE pipeline.

Provides multi-task loss combining severity classification and referable detection,
with support for focal loss, label smoothing, and ordinal regression.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drsafe.utils.config import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference:
        Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Per-class weights tensor of shape (num_classes,).
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction method ('none', 'mean', 'sum').
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (B, num_classes).
            targets: Target labels of shape (B,).
        
        Returns:
            Loss value.
        """
        num_classes = inputs.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_one_hot = F.one_hot(targets, num_classes).float()
                targets_smooth = targets_one_hot * (1 - self.label_smoothing) + \
                                self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()
        
        # Compute probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Compute focal weight
        focal_weight = (1 - probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_weight = alpha.unsqueeze(0).expand_as(probs)
            focal_weight = focal_weight * alpha_weight
        
        # Compute loss
        loss = -focal_weight * targets_smooth * log_probs
        loss = loss.sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize label smoothing cross-entropy.
        
        Args:
            smoothing: Label smoothing factor.
            weight: Per-class weights.
            reduction: Reduction method.
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            inputs: Logits of shape (B, num_classes).
            targets: Target labels of shape (B,).
        
        Returns:
            Loss value.
        """
        num_classes = inputs.size(1)
        
        # Smooth targets
        with torch.no_grad():
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.smoothing) + \
                            self.smoothing / num_classes
        
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Apply class weights
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            log_probs = log_probs * weight.unsqueeze(0)
        
        # Compute loss
        loss = -targets_smooth * log_probs
        loss = loss.sum(dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalLoss(nn.Module):
    """
    Loss function for ordinal regression (CORAL/CORN).
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        method: str = "coral",  # "coral" or "corn"
        reduction: str = "mean",
    ):
        """
        Initialize ordinal loss.
        
        Args:
            num_classes: Number of ordinal classes.
            method: Ordinal method ("coral" or "corn").
            reduction: Reduction method.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.method = method
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ordinal loss.
        
        Args:
            logits: Cumulative/conditional logits of shape (B, num_classes - 1).
            targets: Target labels of shape (B,).
        
        Returns:
            Loss value.
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # Create ordinal targets
        # For target k, labels are [1, 1, ..., 1, 0, 0, ..., 0]
        # where there are k ones (for Y > 0, Y > 1, ..., Y > k-1)
        ordinal_targets = torch.zeros(batch_size, self.num_thresholds, device=device)
        
        for i in range(self.num_thresholds):
            ordinal_targets[:, i] = (targets > i).float()
        
        if self.method == "coral":
            # CORAL: standard BCE loss on cumulative probabilities
            loss = F.binary_cross_entropy_with_logits(
                logits,
                ordinal_targets,
                reduction=self.reduction,
            )
        else:
            # CORN: task-conditional training
            # Only compute loss for tasks where label is relevant
            loss = 0.0
            task_count = 0
            
            for k in range(self.num_thresholds):
                # For task k, only include samples where Y >= k
                mask = (targets >= k)
                
                if mask.sum() > 0:
                    task_logits = logits[mask, k]
                    task_targets = ordinal_targets[mask, k]
                    
                    task_loss = F.binary_cross_entropy_with_logits(
                        task_logits,
                        task_targets,
                        reduction="mean",
                    )
                    loss += task_loss
                    task_count += 1
            
            if task_count > 0:
                loss = loss / task_count
        
        return loss


class DRLoss(nn.Module):
    """
    Multi-task loss for Diabetic Retinopathy classification.
    
    Combines severity classification loss and referable detection loss
    with configurable weights and loss functions.
    """
    
    def __init__(
        self,
        severity_weight: float = 1.0,
        referable_weight: float = 0.5,
        num_classes: int = 5,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        use_ordinal: bool = False,
        ordinal_method: str = "coral",
    ):
        """
        Initialize multi-task loss.
        
        Args:
            severity_weight: Weight for severity loss.
            referable_weight: Weight for referable loss.
            num_classes: Number of severity classes.
            use_focal: Whether to use focal loss for severity.
            focal_gamma: Focal loss gamma parameter.
            label_smoothing: Label smoothing factor.
            class_weights: Per-class weights for severity loss.
            use_ordinal: Whether to use ordinal loss for severity.
            ordinal_method: Ordinal method ("coral" or "corn").
        """
        super().__init__()
        
        self.severity_weight = severity_weight
        self.referable_weight = referable_weight
        self.use_ordinal = use_ordinal
        
        # Severity loss
        if use_ordinal:
            self.severity_loss_fn = OrdinalLoss(
                num_classes=num_classes,
                method=ordinal_method,
            )
        elif use_focal:
            self.severity_loss_fn = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            self.severity_loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights,
            )
        
        # Referable loss (binary cross-entropy)
        self.referable_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        severity_logits: torch.Tensor,
        referable_logits: torch.Tensor,
        severity_targets: torch.Tensor,
        referable_targets: torch.Tensor,
        mixup_lambda: Optional[float] = None,
        severity_targets_2: Optional[torch.Tensor] = None,
        referable_targets_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            severity_logits: Severity logits of shape (B, num_classes) or (B, num_classes - 1) for ordinal.
            referable_logits: Referable logits of shape (B, 1).
            severity_targets: Severity targets of shape (B,).
            referable_targets: Referable targets of shape (B,).
            mixup_lambda: MixUp/CutMix lambda for mixed labels.
            severity_targets_2: Second set of severity targets for MixUp.
            referable_targets_2: Second set of referable targets for MixUp.
        
        Returns:
            Dictionary containing individual losses and total loss.
        """
        # Handle MixUp/CutMix
        if mixup_lambda is not None and severity_targets_2 is not None:
            # Compute loss for both sets of targets and interpolate
            severity_loss_1 = self.severity_loss_fn(severity_logits, severity_targets)
            severity_loss_2 = self.severity_loss_fn(severity_logits, severity_targets_2)
            severity_loss = mixup_lambda * severity_loss_1 + (1 - mixup_lambda) * severity_loss_2
        else:
            severity_loss = self.severity_loss_fn(severity_logits, severity_targets)
        
        # Referable loss
        referable_logits_flat = referable_logits.squeeze(-1)
        referable_targets_float = referable_targets.float()
        
        if mixup_lambda is not None and referable_targets_2 is not None:
            referable_targets_2_float = referable_targets_2.float()
            mixed_referable_targets = mixup_lambda * referable_targets_float + \
                                     (1 - mixup_lambda) * referable_targets_2_float
            referable_loss = self.referable_loss_fn(referable_logits_flat, mixed_referable_targets)
        else:
            referable_loss = self.referable_loss_fn(referable_logits_flat, referable_targets_float)
        
        # Total loss
        total_loss = self.severity_weight * severity_loss + self.referable_weight * referable_loss
        
        return {
            "loss": total_loss,
            "severity_loss": severity_loss,
            "referable_loss": referable_loss,
        }


def create_loss_function(
    config: LossConfig,
    num_classes: int = 5,
    class_weights: Optional[torch.Tensor] = None,
    use_ordinal: bool = False,
) -> DRLoss:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration.
        num_classes: Number of severity classes.
        class_weights: Per-class weights.
        use_ordinal: Whether to use ordinal regression.
    
    Returns:
        DRLoss instance.
    """
    # Convert focal_alpha to tensor if provided
    if config.focal_alpha is not None:
        focal_alpha = torch.tensor(config.focal_alpha, dtype=torch.float32)
    elif class_weights is not None:
        focal_alpha = class_weights
    elif config.use_class_weights:
        # Default weights emphasizing minority classes
        focal_alpha = torch.tensor([0.5, 1.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    else:
        focal_alpha = None
    
    return DRLoss(
        severity_weight=config.severity_weight,
        referable_weight=config.referable_weight,
        num_classes=num_classes,
        use_focal=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
        class_weights=focal_alpha,
        use_ordinal=use_ordinal,
        ordinal_method=config.ordinal_method,
    )
