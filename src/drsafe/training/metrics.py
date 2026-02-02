"""
Metrics for DR-SAFE pipeline.

Provides comprehensive metrics for evaluating diabetic retinopathy classification:
- Quadratic Weighted Kappa (QWK) for severity grading
- ROC-AUC, PR-AUC for referable detection (correctly computed from probabilities)
- Calibration metrics (ECE, MCE, Brier Score)
- Confusion matrix and per-class metrics
- Operating point utilities (sensitivity @ specificity, threshold selection)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def quadratic_weighted_kappa(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    num_classes: int = 5,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).
    
    QWK measures agreement between raters, accounting for agreement by chance.
    It penalizes disagreements more heavily when they are further apart.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
    
    Returns:
        QWK score in range [-1, 1], where 1 is perfect agreement.
    
    Example:
        >>> y_true = [0, 1, 2, 3, 4]
        >>> y_pred = [0, 1, 2, 3, 4]
        >>> qwk = quadratic_weighted_kappa(y_true, y_pred)
        >>> print(f"QWK: {qwk:.4f}")
        QWK: 1.0000
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Build confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Build weight matrix (quadratic weights)
    weight_mat = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weight_mat[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    
    # Compute expected matrix
    hist_true = np.bincount(y_true, minlength=num_classes)
    hist_pred = np.bincount(y_pred, minlength=num_classes)
    
    n = len(y_true)
    expected = np.outer(hist_true, hist_pred).astype(float) / n
    
    # Compute QWK
    observed_weighted = np.sum(weight_mat * conf_mat)
    expected_weighted = np.sum(weight_mat * expected)
    
    if expected_weighted == 0:
        return 1.0
    
    kappa = 1 - observed_weighted / expected_weighted
    
    return float(kappa)


def expected_calibration_error(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    n_bins: int = 10,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Measures how well the predicted probabilities match empirical accuracy.
    
    Args:
        probs: Predicted probabilities.
        targets: Binary targets (0 or 1).
        n_bins: Number of bins for calibration.
    
    Returns:
        Tuple of (ECE, MCE, bin_accuracies, bin_confidences, bin_counts).
    """
    # Convert to numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    probs = np.asarray(probs).flatten()
    targets = np.asarray(targets).flatten()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        
        if in_bin.sum() > 0:
            # Compute accuracy and confidence for this bin
            bin_acc = targets[in_bin].mean()
            bin_conf = probs[in_bin].mean()
            bin_count = int(in_bin.sum())
            
            # Update ECE
            ece += np.abs(bin_acc - bin_conf) * bin_count
            
            # Update MCE
            mce = max(mce, np.abs(bin_acc - bin_conf))
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
        else:
            # Use NaN for empty bins to distinguish from bins with 0% accuracy
            bin_accuracies.append(np.nan)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    ece = ece / len(probs) if len(probs) > 0 else 0.0
    
    return float(ece), float(mce), np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def fbeta_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    beta: float = 2.0,
) -> float:
    """
    Compute F-beta score.
    
    F2 score (beta=2) weighs recall higher than precision, useful for
    medical screening where missing positive cases is costly.
    
    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        beta: Beta parameter (2 for F2 score).
    
    Returns:
        F-beta score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    return float(f_beta)


def compute_brier_score(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Brier score for binary classification.
    
    Brier score is the mean squared error between predicted probabilities
    and actual outcomes. Lower is better (0 = perfect, 0.25 = random).
    
    Args:
        probs: Predicted probabilities for positive class.
        targets: Binary targets (0 or 1).
    
    Returns:
        Brier score in [0, 1].
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    probs = np.asarray(probs).flatten()
    targets = np.asarray(targets).flatten()
    
    return float(brier_score_loss(targets, probs))


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_specificity: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute sensitivity at a target specificity level.
    
    Useful for clinical settings where you need to guarantee a minimum
    specificity (e.g., 90%) and want to know the resulting sensitivity.
    
    Args:
        y_true: True binary labels.
        y_probs: Predicted probabilities.
        target_specificity: Target specificity level (default 0.90).
    
    Returns:
        Tuple of (sensitivity, threshold) at the target specificity.
    """
    y_true = np.asarray(y_true).flatten()
    y_probs = np.asarray(y_probs).flatten()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificities = 1 - fpr
    
    # Find the threshold that achieves target specificity
    # We want the highest sensitivity (tpr) where specificity >= target
    valid_idx = specificities >= target_specificity
    
    if not valid_idx.any():
        # Cannot achieve target specificity
        return 0.0, 1.0
    
    # Among valid points, find the one with highest sensitivity
    best_idx = np.where(valid_idx)[0][np.argmax(tpr[valid_idx])]
    
    return float(tpr[best_idx]), float(thresholds[best_idx])


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_sensitivity: float = 0.90,
) -> Tuple[float, float]:
    """
    Compute specificity at a target sensitivity level.
    
    Useful for clinical settings where you need to guarantee a minimum
    sensitivity (e.g., 90%) and want to know the resulting specificity.
    
    Args:
        y_true: True binary labels.
        y_probs: Predicted probabilities.
        target_sensitivity: Target sensitivity level (default 0.90).
    
    Returns:
        Tuple of (specificity, threshold) at the target sensitivity.
    """
    y_true = np.asarray(y_true).flatten()
    y_probs = np.asarray(y_probs).flatten()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    specificities = 1 - fpr
    
    # Find the threshold that achieves target sensitivity
    valid_idx = tpr >= target_sensitivity
    
    if not valid_idx.any():
        # Cannot achieve target sensitivity
        return 0.0, 0.0
    
    # Among valid points, find the one with highest specificity
    best_idx = np.where(valid_idx)[0][np.argmax(specificities[valid_idx])]
    
    return float(specificities[best_idx]), float(thresholds[best_idx])


def find_threshold_for_sensitivity(
    probs: np.ndarray,
    targets: np.ndarray,
    target_sensitivity: float = 0.90,
) -> float:
    """
    Find threshold that achieves at least target sensitivity.
    
    Args:
        probs: Predicted probabilities.
        targets: Binary targets.
        target_sensitivity: Minimum required sensitivity.
    
    Returns:
        Threshold value.
    """
    _, threshold = specificity_at_sensitivity(targets, probs, target_sensitivity)
    return threshold


def find_threshold_for_specificity(
    probs: np.ndarray,
    targets: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """
    Find threshold that achieves at least target specificity.
    
    Args:
        probs: Predicted probabilities.
        targets: Binary targets.
        target_specificity: Minimum required specificity.
    
    Returns:
        Threshold value.
    """
    _, threshold = sensitivity_at_specificity(targets, probs, target_specificity)
    return threshold


class DRMetrics:
    """
    Comprehensive metrics calculator for DR classification.
    
    Tracks predictions and computes metrics at the end of an epoch.
    
    IMPORTANT: ROC-AUC and PR-AUC are computed from probability scores,
    NOT from hard predictions. This is critical for correct evaluation.
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes: Number of severity classes.
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset all stored predictions."""
        self.severity_preds: List[np.ndarray] = []
        self.severity_targets: List[np.ndarray] = []
        self.severity_probs: List[np.ndarray] = []
        
        self.referable_preds: List[np.ndarray] = []
        self.referable_targets: List[np.ndarray] = []
        self.referable_probs: List[np.ndarray] = []
    
    def update(
        self,
        severity_preds: Union[np.ndarray, torch.Tensor],
        severity_targets: Union[np.ndarray, torch.Tensor],
        severity_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        referable_preds: Optional[Union[np.ndarray, torch.Tensor]] = None,
        referable_targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        referable_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Update metrics with batch predictions.
        
        Args:
            severity_preds: Predicted severity classes.
            severity_targets: True severity classes.
            severity_probs: Severity class probabilities.
            referable_preds: Predicted referable status (hard labels).
            referable_targets: True referable status.
            referable_probs: Referable probabilities (REQUIRED for AUC metrics).
        """
        # Convert to numpy
        def to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        self.severity_preds.append(to_numpy(severity_preds))
        self.severity_targets.append(to_numpy(severity_targets))
        
        if severity_probs is not None:
            self.severity_probs.append(to_numpy(severity_probs))
        
        if referable_preds is not None:
            self.referable_preds.append(to_numpy(referable_preds))
        
        if referable_targets is not None:
            self.referable_targets.append(to_numpy(referable_targets))
        
        if referable_probs is not None:
            self.referable_probs.append(to_numpy(referable_probs))
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names to values.
            
        Note:
            ROC-AUC and PR-AUC are computed from referable_probs (probabilities),
            NOT from referable_preds (hard labels). This is the correct approach.
        """
        metrics = {}
        
        # Concatenate all predictions
        severity_preds = np.concatenate(self.severity_preds)
        severity_targets = np.concatenate(self.severity_targets)
        
        # Severity metrics
        metrics["severity_accuracy"] = float(accuracy_score(severity_targets, severity_preds))
        metrics["severity_qwk"] = quadratic_weighted_kappa(
            severity_targets, severity_preds, self.num_classes
        )
        metrics["severity_macro_f1"] = float(
            f1_score(severity_targets, severity_preds, average="macro", zero_division=0)
        )
        
        # Per-class accuracy
        for cls in range(self.num_classes):
            mask = severity_targets == cls
            if mask.sum() > 0:
                metrics[f"severity_accuracy_class{cls}"] = float(
                    (severity_preds[mask] == cls).mean()
                )
        
        # Referable metrics
        if self.referable_targets:
            referable_targets = np.concatenate(self.referable_targets)
            
            if self.referable_preds:
                referable_preds = np.concatenate(self.referable_preds)
                
                # Confusion matrix elements
                tn = ((referable_preds == 0) & (referable_targets == 0)).sum()
                fp = ((referable_preds == 1) & (referable_targets == 0)).sum()
                fn = ((referable_preds == 0) & (referable_targets == 1)).sum()
                tp = ((referable_preds == 1) & (referable_targets == 1)).sum()
                
                metrics["referable_accuracy"] = float(
                    accuracy_score(referable_targets, referable_preds)
                )
                metrics["referable_precision"] = float(
                    precision_score(referable_targets, referable_preds, zero_division=0)
                )
                metrics["referable_recall"] = float(
                    recall_score(referable_targets, referable_preds, zero_division=0)
                )
                # Sensitivity = Recall = TPR
                metrics["referable_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                # Specificity = TNR
                metrics["referable_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                
                metrics["referable_f1"] = float(
                    f1_score(referable_targets, referable_preds, zero_division=0)
                )
                metrics["referable_f2"] = fbeta_score(referable_targets, referable_preds, beta=2.0)
            
            # AUC metrics - MUST use probabilities, not hard predictions
            if self.referable_probs:
                referable_probs = np.concatenate(self.referable_probs)
                
                # ROC-AUC (using probabilities - CORRECT)
                try:
                    metrics["referable_roc_auc"] = float(
                        roc_auc_score(referable_targets, referable_probs)
                    )
                except ValueError:
                    metrics["referable_roc_auc"] = 0.0
                
                # PR-AUC using average_precision_score (CORRECT implementation)
                # NOTE: The old implementation used np.trapz(precision, recall) which
                # can give wrong/negative values because sklearn's recall array from
                # precision_recall_curve is not sorted in ascending order.
                # average_precision_score computes the correct area under the PR curve.
                try:
                    metrics["referable_pr_auc"] = float(
                        average_precision_score(referable_targets, referable_probs)
                    )
                except ValueError:
                    metrics["referable_pr_auc"] = 0.0
                
                # Brier score (calibration metric - lower is better)
                metrics["referable_brier_score"] = float(
                    brier_score_loss(referable_targets, referable_probs)
                )
                
                # Calibration (ECE/MCE)
                ece, mce, _, _, _ = expected_calibration_error(
                    referable_probs, referable_targets
                )
                metrics["referable_ece"] = ece
                metrics["referable_mce"] = mce
                
                # Operating point metrics for clinical use
                try:
                    sens_at_90spec, thresh_90spec = sensitivity_at_specificity(
                        referable_targets, referable_probs, 0.90
                    )
                    metrics["referable_sens_at_90spec"] = sens_at_90spec
                    metrics["referable_thresh_90spec"] = thresh_90spec
                    
                    spec_at_90sens, thresh_90sens = specificity_at_sensitivity(
                        referable_targets, referable_probs, 0.90
                    )
                    metrics["referable_spec_at_90sens"] = spec_at_90sens
                    metrics["referable_thresh_90sens"] = thresh_90sens
                except Exception:
                    pass
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for severity classification."""
        severity_preds = np.concatenate(self.severity_preds)
        severity_targets = np.concatenate(self.severity_targets)
        
        return confusion_matrix(
            severity_targets, severity_preds, labels=list(range(self.num_classes))
        )
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        severity_preds = np.concatenate(self.severity_preds)
        severity_targets = np.concatenate(self.severity_targets)
        
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        
        return classification_report(
            severity_targets,
            severity_preds,
            target_names=class_names[:self.num_classes],
            zero_division=0,
        )
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve for referable detection.
        
        Returns:
            Tuple of (fpr, tpr, thresholds).
        """
        if not self.referable_probs or not self.referable_targets:
            raise ValueError("No referable predictions stored")
        
        referable_probs = np.concatenate(self.referable_probs)
        referable_targets = np.concatenate(self.referable_targets)
        
        return roc_curve(referable_targets, referable_probs)
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve for referable detection.
        
        Returns:
            Tuple of (precision, recall, thresholds).
        """
        if not self.referable_probs or not self.referable_targets:
            raise ValueError("No referable predictions stored")
        
        referable_probs = np.concatenate(self.referable_probs)
        referable_targets = np.concatenate(self.referable_targets)
        
        return precision_recall_curve(referable_targets, referable_probs)
    
    def get_optimal_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get optimal thresholds for different metrics.
        
        Returns:
            Dictionary mapping metric name to (threshold, score) tuples.
        """
        if not self.referable_probs or not self.referable_targets:
            raise ValueError("No referable predictions stored")
        
        referable_probs = np.concatenate(self.referable_probs)
        referable_targets = np.concatenate(self.referable_targets)
        
        return {
            "f1": compute_optimal_threshold(referable_probs, referable_targets, "f1"),
            "f2": compute_optimal_threshold(referable_probs, referable_targets, "f2"),
            "youden": compute_optimal_threshold(referable_probs, referable_targets, "youden"),
        }


def compute_optimal_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        probs: Predicted probabilities.
        targets: Binary targets.
        metric: Metric to optimize ("f1", "f2", "youden").
    
    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    thresholds = np.linspace(0, 1, 101)
    best_score = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(targets, preds, zero_division=0)
        elif metric == "f2":
            score = fbeta_score(targets, preds, beta=2.0)
        elif metric == "youden":
            # Youden's J statistic = sensitivity + specificity - 1
            tn = ((preds == 0) & (targets == 0)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            tp = ((preds == 1) & (targets == 1)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return float(best_threshold), float(best_score)
