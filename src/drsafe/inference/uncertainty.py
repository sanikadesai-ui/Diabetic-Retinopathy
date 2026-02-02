"""
Uncertainty estimation for DR-SAFE pipeline.

Provides MC Dropout for uncertainty estimation and triage categorization
based on predictive entropy.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from drsafe.utils.config import Config
from drsafe.utils.logging import get_logger

logger = get_logger()


class TriageCategory(str, Enum):
    """Triage categories based on uncertainty."""
    CERTAIN_NON_REFER = "CERTAIN_NON_REFER"
    CERTAIN_REFER = "CERTAIN_REFER"
    UNCERTAIN = "UNCERTAIN"


def compute_predictive_entropy(
    probs: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute predictive entropy from probability distribution.
    
    Higher entropy indicates higher uncertainty.
    
    Args:
        probs: Probability array of shape (B, num_classes) or (B,).
        eps: Small constant for numerical stability.
    
    Returns:
        Entropy values of shape (B,).
    """
    probs = np.clip(probs, eps, 1 - eps)
    
    if probs.ndim == 1:
        # Binary case
        entropy = -probs * np.log(probs) - (1 - probs) * np.log(1 - probs)
    else:
        # Multi-class case
        entropy = -np.sum(probs * np.log(probs), axis=1)
    
    return entropy


def compute_mutual_information(
    mc_probs: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute mutual information (epistemic uncertainty) from MC samples.
    
    MI = H(y|x) - E[H(y|x, w)]
    
    Args:
        mc_probs: MC probability samples of shape (n_samples, B, num_classes).
        eps: Small constant for numerical stability.
    
    Returns:
        Mutual information of shape (B,).
    """
    # Mean prediction entropy (total uncertainty)
    mean_probs = mc_probs.mean(axis=0)
    total_entropy = compute_predictive_entropy(mean_probs, eps)
    
    # Mean of individual entropies (aleatoric uncertainty)
    individual_entropies = np.array([
        compute_predictive_entropy(probs, eps) for probs in mc_probs
    ])
    mean_entropy = individual_entropies.mean(axis=0)
    
    # Mutual information (epistemic uncertainty)
    mi = total_entropy - mean_entropy
    
    return mi


class MCDropoutPredictor:
    """
    MC Dropout predictor for uncertainty estimation.
    
    Performs multiple forward passes with dropout enabled to estimate
    predictive uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        n_samples: int = 10,
    ):
        """
        Initialize MC Dropout predictor.
        
        Args:
            model: Trained model (must have dropout layers).
            config: Configuration object.
            device: Device to run inference on.
            n_samples: Number of MC dropout samples.
        """
        self.model = model
        self.config = config
        self.device = device
        self.n_samples = n_samples
        
        logger.info(f"MC Dropout enabled with {n_samples} samples")
    
    def _enable_dropout(self) -> None:
        """Enable dropout layers for MC inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Make MC Dropout predictions on a batch.
        
        Args:
            images: Tensor of shape (B, C, H, W).
        
        Returns:
            Dictionary containing predictions and uncertainties.
        """
        self.model.eval()
        self._enable_dropout()
        
        images = images.to(self.device)
        
        mc_severity_probs = []
        mc_referable_probs = []
        
        for _ in range(self.n_samples):
            severity_logits, referable_logits = self.model(images)
            
            if hasattr(self.model, 'use_ordinal') and self.model.use_ordinal:
                severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
            else:
                severity_probs = torch.softmax(severity_logits, dim=1)
            
            referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
            
            mc_severity_probs.append(severity_probs.cpu().numpy())
            mc_referable_probs.append(referable_probs.cpu().numpy())
        
        # Stack MC samples
        mc_severity_probs = np.array(mc_severity_probs)  # (n_samples, B, num_classes)
        mc_referable_probs = np.array(mc_referable_probs)  # (n_samples, B)
        
        # Mean predictions
        mean_severity_probs = mc_severity_probs.mean(axis=0)
        mean_referable_probs = mc_referable_probs.mean(axis=0)
        
        # Predictions
        severity_preds = mean_severity_probs.argmax(axis=1)
        referable_preds = (mean_referable_probs > self.config.inference.referable_threshold).astype(int)
        
        # Uncertainties
        severity_entropy = compute_predictive_entropy(mean_severity_probs)
        referable_entropy = compute_predictive_entropy(mean_referable_probs)
        
        # Epistemic uncertainty (mutual information)
        severity_mi = compute_mutual_information(mc_severity_probs)
        referable_std = mc_referable_probs.std(axis=0)
        
        # Combined uncertainty score
        uncertainty = (severity_entropy + referable_entropy) / 2
        
        # Triage categorization
        triage = self._compute_triage(
            referable_probs=mean_referable_probs,
            uncertainty=uncertainty,
        )
        
        self.model.eval()  # Restore eval mode
        
        return {
            "severity_probs": mean_severity_probs,
            "severity_pred": severity_preds,
            "referable_probs": mean_referable_probs,
            "referable_pred": referable_preds,
            "severity_entropy": severity_entropy,
            "referable_entropy": referable_entropy,
            "severity_epistemic": severity_mi,
            "referable_std": referable_std,
            "uncertainty": uncertainty,
            "triage": triage,
        }
    
    def _compute_triage(
        self,
        referable_probs: np.ndarray,
        uncertainty: np.ndarray,
    ) -> List[str]:
        """
        Compute triage categories based on predictions and uncertainty.
        
        Args:
            referable_probs: Referable probabilities.
            uncertainty: Uncertainty scores.
        
        Returns:
            List of triage category strings.
        """
        threshold = self.config.inference.certain_threshold
        referable_thresh = self.config.inference.referable_threshold
        
        triage = []
        for prob, unc in zip(referable_probs, uncertainty):
            if unc > threshold:
                triage.append(TriageCategory.UNCERTAIN.value)
            elif prob >= referable_thresh:
                triage.append(TriageCategory.CERTAIN_REFER.value)
            else:
                triage.append(TriageCategory.CERTAIN_NON_REFER.value)
        
        return triage


def triage_predictions(
    referable_probs: np.ndarray,
    uncertainty: np.ndarray,
    referable_threshold: float = 0.5,
    uncertainty_threshold: float = 0.15,
) -> List[str]:
    """
    Categorize predictions into triage categories.
    
    Args:
        referable_probs: Referable probabilities.
        uncertainty: Uncertainty scores (e.g., entropy).
        referable_threshold: Threshold for referable classification.
        uncertainty_threshold: Threshold for uncertainty categorization.
    
    Returns:
        List of triage category strings.
    
    Categories:
        - CERTAIN_NON_REFER: Low uncertainty, low referable probability
        - CERTAIN_REFER: Low uncertainty, high referable probability
        - UNCERTAIN: High uncertainty (needs expert review)
    """
    triage = []
    
    for prob, unc in zip(referable_probs, uncertainty):
        if unc > uncertainty_threshold:
            triage.append(TriageCategory.UNCERTAIN.value)
        elif prob >= referable_threshold:
            triage.append(TriageCategory.CERTAIN_REFER.value)
        else:
            triage.append(TriageCategory.CERTAIN_NON_REFER.value)
    
    return triage


def ensemble_uncertainty(
    predictions: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Compute ensemble predictions and uncertainty from multiple models.
    
    Args:
        predictions: List of prediction dictionaries from different models.
    
    Returns:
        Dictionary with ensemble predictions and uncertainties.
    """
    n_models = len(predictions)
    
    # Stack predictions
    severity_probs_stack = np.stack([p["severity_probs"] for p in predictions])
    referable_probs_stack = np.stack([p["referable_probs"] for p in predictions])
    
    # Mean predictions
    mean_severity_probs = severity_probs_stack.mean(axis=0)
    mean_referable_probs = referable_probs_stack.mean(axis=0)
    
    # Predictions
    severity_preds = mean_severity_probs.argmax(axis=1)
    referable_preds = (mean_referable_probs > 0.5).astype(int)
    
    # Uncertainty from ensemble disagreement
    severity_std = severity_probs_stack.std(axis=0).mean(axis=1)
    referable_std = referable_probs_stack.std(axis=0)
    
    # Predictive entropy
    severity_entropy = compute_predictive_entropy(mean_severity_probs)
    referable_entropy = compute_predictive_entropy(mean_referable_probs)
    
    return {
        "severity_probs": mean_severity_probs,
        "severity_pred": severity_preds,
        "referable_probs": mean_referable_probs,
        "referable_pred": referable_preds,
        "severity_std": severity_std,
        "referable_std": referable_std,
        "severity_entropy": severity_entropy,
        "referable_entropy": referable_entropy,
        "uncertainty": (severity_entropy + referable_entropy) / 2,
    }
