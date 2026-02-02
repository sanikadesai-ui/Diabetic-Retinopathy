"""
Calibration utilities for DR-SAFE pipeline.

Implements temperature scaling for probability calibration, particularly
important for reliable confidence estimates in medical applications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from tqdm import tqdm

from drsafe.training.metrics import expected_calibration_error
from drsafe.utils.logging import get_logger

logger = get_logger()


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    
    Learns a single temperature parameter to scale logits before softmax,
    improving probability calibration without changing predictions.
    
    Reference:
        Guo, C., et al. (2017). On Calibration of Modern Neural Networks.
        https://arxiv.org/abs/1706.04599
    """
    
    def __init__(self, init_temperature: float = 1.0):
        """
        Initialize temperature scaling.
        
        Args:
            init_temperature: Initial temperature value.
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Input logits of shape (B, num_classes) or (B, 1).
        
        Returns:
            Scaled logits.
        """
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        task: str = "binary",
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Fit temperature parameter using validation data.
        
        Args:
            logits: Validation logits.
            labels: Validation labels.
            task: "binary" or "multiclass".
            max_iter: Maximum optimization iterations.
            lr: Learning rate for optimization.
        
        Returns:
            Optimal temperature value.
        """
        self.train()
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        if task == "binary":
            criterion = nn.BCEWithLogitsLoss()
            labels = labels.float()
        else:
            criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            
            if task == "binary":
                loss = criterion(scaled_logits.squeeze(), labels)
            else:
                loss = criterion(scaled_logits, labels)
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        self.eval()
        optimal_temp = self.temperature.item()
        
        logger.info(f"Calibration complete. Optimal temperature: {optimal_temp:.4f}")
        
        return optimal_temp


def calibrate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    task: str = "binary",
) -> Tuple[float, float, float]:
    """
    Calibrate a model using temperature scaling on validation data.
    
    Args:
        model: Trained model.
        val_loader: Validation data loader.
        device: Device to run on.
        task: "binary" (referable) or "multiclass" (severity).
    
    Returns:
        Tuple of (optimal_temperature, ece_before, ece_after).
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    
    all_logits = []
    all_labels = []
    
    # Collect all logits and labels
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits"):
            images = batch["image"].to(device)
            
            if task == "binary":
                labels = batch["referable_label"]
            else:
                labels = batch["severity_label"]
            
            severity_logits, referable_logits = model(images)
            
            if task == "binary":
                logits = referable_logits.squeeze(-1)
            else:
                logits = severity_logits
            
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Compute ECE before calibration
    if task == "binary":
        probs_before = torch.sigmoid(all_logits).numpy()
        ece_before, _, _, _ = expected_calibration_error(probs_before, all_labels.numpy())
    else:
        probs_before = torch.softmax(all_logits, dim=1).numpy()
        # For multiclass, use top-1 confidence
        top_probs = probs_before.max(axis=1)
        correct = (probs_before.argmax(axis=1) == all_labels.numpy()).astype(int)
        ece_before, _, _, _ = expected_calibration_error(top_probs, correct)
    
    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(all_logits, all_labels, task=task)
    
    # Compute ECE after calibration
    with torch.no_grad():
        scaled_logits = temp_scaler(all_logits)
        
        if task == "binary":
            probs_after = torch.sigmoid(scaled_logits).numpy()
            ece_after, _, _, _ = expected_calibration_error(probs_after, all_labels.numpy())
        else:
            probs_after = torch.softmax(scaled_logits, dim=1).numpy()
            top_probs = probs_after.max(axis=1)
            correct = (probs_after.argmax(axis=1) == all_labels.numpy()).astype(int)
            ece_after, _, _, _ = expected_calibration_error(top_probs, correct)
    
    logger.info(f"ECE before calibration: {ece_before:.4f}")
    logger.info(f"ECE after calibration: {ece_after:.4f}")
    
    return optimal_temp, ece_before, ece_after


def apply_temperature_scaling(
    logits: Union[np.ndarray, torch.Tensor],
    temperature: float,
    task: str = "binary",
) -> np.ndarray:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Input logits.
        temperature: Temperature value.
        task: "binary" or "multiclass".
    
    Returns:
        Calibrated probabilities.
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    scaled_logits = logits / temperature
    
    if task == "binary":
        probs = torch.sigmoid(scaled_logits).numpy()
    else:
        probs = torch.softmax(scaled_logits, dim=-1).numpy()
    
    return probs


def fit_temperature_scipy(
    logits: np.ndarray,
    labels: np.ndarray,
    task: str = "binary",
) -> float:
    """
    Fit temperature using scipy optimization (alternative method).
    
    Args:
        logits: Validation logits.
        labels: Validation labels.
        task: "binary" or "multiclass".
    
    Returns:
        Optimal temperature.
    """
    def nll_loss(temperature: float) -> float:
        scaled_logits = logits / temperature
        
        if task == "binary":
            probs = 1 / (1 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            nll = -np.mean(
                labels * np.log(probs) + (1 - labels) * np.log(1 - probs)
            )
        else:
            # Softmax
            exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        
        return nll
    
    result = minimize(
        nll_loss,
        x0=1.0,
        method="L-BFGS-B",
        bounds=[(0.1, 10.0)],
    )
    
    return float(result.x[0])


def save_calibration(
    temperature: float,
    ece_before: float,
    ece_after: float,
    filepath: Union[str, Path],
) -> None:
    """
    Save calibration results to a file.
    
    Args:
        temperature: Optimal temperature.
        ece_before: ECE before calibration.
        ece_after: ECE after calibration.
        filepath: Path to save the results.
    """
    import json
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Calibration results saved to {filepath}")


def load_calibration(filepath: Union[str, Path]) -> Dict[str, float]:
    """
    Load calibration results from a file.
    
    Args:
        filepath: Path to the calibration file.
    
    Returns:
        Dictionary with calibration parameters.
    """
    import json
    
    with open(filepath, "r") as f:
        results = json.load(f)
    
    return results


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> "matplotlib.figure.Figure":
    """
    Create a reliability diagram for probability calibration analysis.
    
    Args:
        probs: Predicted probabilities.
        labels: True binary labels.
        n_bins: Number of bins.
        title: Plot title.
    
    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    
    ece, mce, bin_accuracies, bin_confidences = expected_calibration_error(
        probs, labels, n_bins
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    
    # Plot actual calibration
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    ax.bar(
        bin_centers,
        bin_accuracies,
        width=0.08,
        alpha=0.7,
        label=f"Model (ECE={ece:.4f})",
    )
    
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    return fig
