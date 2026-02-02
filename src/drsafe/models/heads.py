"""
Classification heads for DR-SAFE pipeline.

Provides various head architectures for multi-task learning:
- Severity classification (5-class)
- Referable DR detection (binary)
- Ordinal regression (optional)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeverityHead(nn.Module):
    """
    Classification head for 5-class severity grading.
    
    Outputs logits for classes 0-4 (No DR, Mild, Moderate, Severe, Proliferative).
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the severity head.
        
        Args:
            in_features: Number of input features from backbone.
            num_classes: Number of severity classes (default 5).
            hidden_dim: Hidden layer dimension (if None, direct projection).
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, in_features).
        
        Returns:
            Logits of shape (B, num_classes).
        """
        return self.head(x)


class ReferableHead(nn.Module):
    """
    Classification head for binary referable DR detection.
    
    Outputs a single logit for referable (DR level >= 2) vs non-referable.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the referable head.
        
        Args:
            in_features: Number of input features from backbone.
            hidden_dim: Hidden layer dimension (if None, direct projection).
            dropout: Dropout rate.
        """
        super().__init__()
        
        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, in_features).
        
        Returns:
            Logits of shape (B, 1).
        """
        return self.head(x)


class OrdinalHead(nn.Module):
    """
    Ordinal regression head using cumulative link model (CORAL).
    
    For ordinal labels (0 < 1 < 2 < 3 < 4), uses K-1 binary classifiers
    where each predicts P(Y > k).
    
    Reference:
        Cao, W., et al. (2019). Rank consistent ordinal regression for neural
        networks with application to age estimation.
        https://arxiv.org/abs/1901.07884
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the ordinal head.
        
        Args:
            in_features: Number of input features from backbone.
            num_classes: Number of ordinal classes (default 5).
            hidden_dim: Hidden layer dimension (if None, direct projection).
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        # Feature projection
        if hidden_dim is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            proj_dim = hidden_dim
        else:
            self.feature_proj = nn.Dropout(dropout)
            proj_dim = in_features
        
        # Shared linear layer (weight-sharing across thresholds)
        self.linear = nn.Linear(proj_dim, 1, bias=False)
        
        # Learnable thresholds (biases for each cumulative probability)
        self.thresholds = nn.Parameter(torch.zeros(self.num_thresholds))
        
        # Initialize thresholds to be ordered
        with torch.no_grad():
            self.thresholds.data = torch.linspace(-1, 1, self.num_thresholds)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, in_features).
        
        Returns:
            Cumulative logits of shape (B, num_classes - 1).
            Each output[i] represents logit for P(Y > i).
        """
        x = self.feature_proj(x)
        
        # Project features to single value
        proj = self.linear(x)  # (B, 1)
        
        # Compute cumulative logits for each threshold
        # logit[k] = proj - threshold[k]
        cumulative_logits = proj - self.thresholds.unsqueeze(0)  # (B, K-1)
        
        return cumulative_logits
    
    def predict_proba(self, cumulative_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert cumulative logits to class probabilities.
        
        Args:
            cumulative_logits: Cumulative logits of shape (B, num_classes - 1).
        
        Returns:
            Class probabilities of shape (B, num_classes).
        """
        # P(Y > k) = sigmoid(cumulative_logits[k])
        cum_probs = torch.sigmoid(cumulative_logits)
        
        # Prepend 1 (P(Y > -1) = 1) and append 0 (P(Y > K) = 0)
        ones = torch.ones(cum_probs.size(0), 1, device=cum_probs.device)
        zeros = torch.zeros(cum_probs.size(0), 1, device=cum_probs.device)
        
        extended_cum_probs = torch.cat([ones, cum_probs, zeros], dim=1)  # (B, K+1)
        
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        class_probs = extended_cum_probs[:, :-1] - extended_cum_probs[:, 1:]
        
        return class_probs
    
    def predict(self, cumulative_logits: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels from cumulative logits.
        
        Args:
            cumulative_logits: Cumulative logits of shape (B, num_classes - 1).
        
        Returns:
            Predicted class labels of shape (B,).
        """
        probs = self.predict_proba(cumulative_logits)
        return probs.argmax(dim=1)


class CORNHead(nn.Module):
    """
    CORN (Conditional Ordinal Regression for Neural Networks) head.
    
    Similar to CORAL but uses conditional probabilities instead of
    cumulative probabilities, which can be more stable for training.
    
    Reference:
        Shi, X., et al. (2021). Deep neural networks for rank-consistent
        ordinal regression based on conditional probabilities.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize the CORN head.
        
        Args:
            in_features: Number of input features from backbone.
            num_classes: Number of ordinal classes (default 5).
            hidden_dim: Hidden layer dimension (if None, direct projection).
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        
        # Feature projection
        if hidden_dim is not None:
            self.feature_proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            proj_dim = hidden_dim
        else:
            self.feature_proj = nn.Dropout(dropout)
            proj_dim = in_features
        
        # Independent binary classifiers for each rank
        self.classifiers = nn.ModuleList([
            nn.Linear(proj_dim, 1) for _ in range(self.num_thresholds)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, in_features).
        
        Returns:
            Conditional logits of shape (B, num_classes - 1).
            Each output[i] represents logit for P(Y > i | Y >= i).
        """
        x = self.feature_proj(x)
        
        logits = [clf(x) for clf in self.classifiers]
        return torch.cat(logits, dim=1)  # (B, K-1)
    
    def predict_proba(self, conditional_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert conditional logits to class probabilities.
        
        Args:
            conditional_logits: Conditional logits of shape (B, num_classes - 1).
        
        Returns:
            Class probabilities of shape (B, num_classes).
        """
        # P(Y > k | Y >= k) = sigmoid(conditional_logits[k])
        cond_probs = torch.sigmoid(conditional_logits)
        
        batch_size = cond_probs.size(0)
        class_probs = torch.zeros(batch_size, self.num_classes, device=cond_probs.device)
        
        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[:, 0] = 1 - cond_probs[:, 0]
        
        # P(Y = k) = P(Y >= k) * (1 - P(Y > k | Y >= k))
        #          = prod_{j<k}(P(Y > j | Y >= j)) * (1 - P(Y > k | Y >= k))
        cumulative = cond_probs[:, 0:1]
        
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cumulative[:, 0] * (1 - cond_probs[:, k])
            cumulative = cumulative * cond_probs[:, k:k+1]
        
        # P(Y = K-1) = P(Y >= K-1) = prod of all conditional probs
        class_probs[:, -1] = cumulative[:, 0]
        
        return class_probs
    
    def predict(self, conditional_logits: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels from conditional logits.
        
        Args:
            conditional_logits: Conditional logits of shape (B, num_classes - 1).
        
        Returns:
            Predicted class labels of shape (B,).
        """
        probs = self.predict_proba(conditional_logits)
        return probs.argmax(dim=1)


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head for severity and referable classification.
    
    Can optionally use ordinal regression for severity.
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = 512,
        dropout: float = 0.3,
        use_ordinal: bool = False,
        ordinal_method: str = "coral",  # "coral" or "corn"
    ):
        """
        Initialize the multi-task head.
        
        Args:
            in_features: Number of input features from backbone.
            num_classes: Number of severity classes (default 5).
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
            use_ordinal: Whether to use ordinal regression for severity.
            ordinal_method: Ordinal method ("coral" or "corn").
        """
        super().__init__()
        
        self.use_ordinal = use_ordinal
        self.ordinal_method = ordinal_method
        
        # Shared feature projection
        self.shared_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Severity head
        if use_ordinal:
            if ordinal_method == "coral":
                self.severity_head = OrdinalHead(
                    in_features=hidden_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                )
            else:
                self.severity_head = CORNHead(
                    in_features=hidden_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                )
        else:
            self.severity_head = SeverityHead(
                in_features=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        
        # Referable head
        self.referable_head = ReferableHead(
            in_features=hidden_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, in_features).
        
        Returns:
            Tuple of (severity_logits, referable_logits).
        """
        shared = self.shared_proj(x)
        
        severity_logits = self.severity_head(shared)
        referable_logits = self.referable_head(shared)
        
        return severity_logits, referable_logits
