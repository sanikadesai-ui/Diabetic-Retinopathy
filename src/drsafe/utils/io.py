"""
I/O utilities for DR-SAFE pipeline.

Provides utilities for file handling, checkpoint management, and data serialization.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml

from drsafe.utils.logging import get_logger

logger = get_logger()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path.
    
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
    is_best: bool = False,
    best_filepath: Optional[Union[str, Path]] = None,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filepath: Path to save the checkpoint.
        is_best: Whether this is the best checkpoint so far.
        best_filepath: Path to save the best checkpoint (if is_best=True).
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    
    if is_best and best_filepath is not None:
        import shutil
        best_filepath = Path(best_filepath)
        ensure_dir(best_filepath.parent)
        shutil.copy(filepath, best_filepath)
        logger.info(f"Best checkpoint copied to {best_filepath}")


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        scaler: GradScaler to load state into.
        strict: Whether to strictly enforce state_dict keys match.
        map_location: Device to map tensors to.
    
    Returns:
        Dictionary containing additional checkpoint data (epoch, metrics, etc.).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    
    # Load EMA state if available
    if "ema_state_dict" in checkpoint:
        checkpoint["ema_state_dict"] = checkpoint["ema_state_dict"]
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")
    
    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")
    
    # Load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load scaler state: {e}")
    
    logger.info(f"Checkpoint loaded from {filepath}")
    
    return checkpoint


def save_predictions(
    predictions: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    Save predictions to a file.
    
    Args:
        predictions: Dictionary containing predictions and metadata.
        filepath: Output file path.
        format: Output format ('csv', 'json', 'pickle').
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if format == "csv":
        df = pd.DataFrame(predictions)
        df.to_csv(filepath, index=False)
    elif format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in predictions.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    json_data[key] = [v.tolist() for v in value]
                else:
                    json_data[key] = list(value)
            else:
                json_data[key] = value
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    elif format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(predictions, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Predictions saved to {filepath}")


def load_predictions(
    filepath: Union[str, Path],
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load predictions from a file.
    
    Args:
        filepath: Input file path.
        format: Input format (auto-detected from extension if None).
    
    Returns:
        Dictionary containing predictions.
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lower().lstrip(".")
    
    if format == "csv":
        df = pd.read_csv(filepath)
        return df.to_dict(orient="list")
    elif format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif format in ("pickle", "pkl"):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_experiment_dir(
    base_dir: Union[str, Path],
    experiment_name: str,
    create: bool = True,
) -> Path:
    """
    Get or create an experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        create: Whether to create the directory.
    
    Returns:
        Path to the experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    if create:
        ensure_dir(exp_dir)
        ensure_dir(exp_dir / "checkpoints")
        ensure_dir(exp_dir / "logs")
        ensure_dir(exp_dir / "predictions")
    
    return exp_dir


def find_images_in_directory(
    directory: Union[str, Path],
    extensions: List[str] = None,
) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        directory: Directory to search.
        extensions: List of valid extensions (default: common image formats).
    
    Returns:
        List of image file paths.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    
    directory = Path(directory)
    images = []
    
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(images)


def detect_image_extension(
    image_dir: Union[str, Path],
    image_id: str,
) -> Optional[str]:
    """
    Detect the file extension for an image ID.
    
    Args:
        image_dir: Directory containing images.
        image_id: Image identifier (without extension).
    
    Returns:
        File extension (including dot) or None if not found.
    """
    image_dir = Path(image_dir)
    extensions = [".jpeg", ".jpg", ".png", ".bmp", ".tiff"]
    
    for ext in extensions:
        if (image_dir / f"{image_id}{ext}").exists():
            return ext
    
    return None


class CheckpointManager:
    """
    Manages multiple checkpoints with automatic cleanup.
    
    Keeps track of the top-k checkpoints based on a monitored metric.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 3,
        monitor: str = "val_qwk",
        mode: str = "max",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            monitor: Metric to monitor for ranking checkpoints.
            mode: 'max' or 'min' for the monitored metric.
        """
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode
        
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_value = float("-inf") if mode == "max" else float("inf")
    
    def _is_better(self, value: float) -> bool:
        """Check if a value is better than the current best."""
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value
    
    def save(
        self,
        state: Dict[str, Any],
        metric_value: float,
        epoch: int,
    ) -> Optional[Path]:
        """
        Save a checkpoint if it's in the top-k.
        
        Args:
            state: Checkpoint state dictionary.
            metric_value: Value of the monitored metric.
            epoch: Current epoch number.
        
        Returns:
            Path to saved checkpoint or None if not saved.
        """
        # Check if this checkpoint should be saved
        is_best = self._is_better(metric_value)
        
        should_save = (
            len(self.checkpoints) < self.max_checkpoints
            or metric_value > min(c["value"] for c in self.checkpoints)
            if self.mode == "max"
            else metric_value < max(c["value"] for c in self.checkpoints)
        )
        
        if not should_save and not is_best:
            return None
        
        # Save checkpoint
        filename = f"checkpoint_epoch{epoch:03d}_{self.monitor}{metric_value:.4f}.pt"
        filepath = self.checkpoint_dir / filename
        
        state["epoch"] = epoch
        state[self.monitor] = metric_value
        
        torch.save(state, filepath)
        
        # Update tracking
        self.checkpoints.append({
            "path": filepath,
            "value": metric_value,
            "epoch": epoch,
        })
        
        # Update best
        if is_best:
            self.best_value = metric_value
            best_path = self.checkpoint_dir / "best_model.pt"
            import shutil
            shutil.copy(filepath, best_path)
            logger.info(f"New best model! {self.monitor}={metric_value:.4f}")
        
        # Remove old checkpoints if necessary
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by value
            if self.mode == "max":
                self.checkpoints.sort(key=lambda x: x["value"], reverse=True)
            else:
                self.checkpoints.sort(key=lambda x: x["value"])
            
            # Remove worst checkpoint
            to_remove = self.checkpoints.pop()
            if to_remove["path"].exists():
                to_remove["path"].unlink()
                logger.info(f"Removed old checkpoint: {to_remove['path'].name}")
        
        return filepath
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return best_path
        return None
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the most recent checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x["epoch"])["path"]
