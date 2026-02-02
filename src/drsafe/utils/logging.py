"""
Logging utilities for DR-SAFE pipeline.

Provides structured logging with console and file handlers,
as well as experiment tracking integration.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler

# Global console for rich output
console = Console()

# Logger registry
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "drsafe",
    level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    rich_format: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name.
        level: Logging level.
        log_dir: Directory to save log files.
        log_to_file: Whether to log to a file.
        log_to_console: Whether to log to console.
        rich_format: Whether to use rich formatting for console output.
    
    Returns:
        Configured logger.
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Console handler
    if log_to_console:
        if rich_format:
            console_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(simple_formatter)
        
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "drsafe") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name.
    
    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    
    return setup_logger(name)


class MetricLogger:
    """
    Logger for tracking and displaying training metrics.
    
    Provides smoothed averages and integration with experiment tracking.
    """
    
    def __init__(
        self,
        delimiter: str = "  ",
        window_size: int = 20,
    ):
        """
        Initialize metric logger.
        
        Args:
            delimiter: Separator between metric displays.
            window_size: Window size for moving average.
        """
        self.meters: Dict[str, SmoothedValue] = {}
        self.delimiter = delimiter
        self.window_size = window_size
    
    def update(self, **kwargs: float) -> None:
        """Update metrics with new values."""
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = SmoothedValue(window_size=self.window_size)
            self.meters[name].update(value)
    
    def __getattr__(self, name: str) -> "SmoothedValue":
        if name in self.meters:
            return self.meters[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.median:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(loss_str)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metric values as a dictionary."""
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def reset(self) -> None:
        """Reset all metrics."""
        for meter in self.meters.values():
            meter.reset()


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize smoothed value tracker.
        
        Args:
            window_size: Size of the moving average window.
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracked values."""
        self.values: list = []
        self.count = 0
        self.total = 0.0
    
    def update(self, value: float, n: int = 1) -> None:
        """
        Update with a new value.
        
        Args:
            value: New value to add.
            n: Weight of the value.
        """
        self.values.append(value)
        self.count += n
        self.total += value * n
        
        # Keep only window_size values
        if len(self.values) > self.window_size:
            self.values = self.values[-self.window_size:]
    
    @property
    def median(self) -> float:
        """Get median of recent values."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        return sorted_values[mid]
    
    @property
    def avg(self) -> float:
        """Get average of all values."""
        if self.count == 0:
            return 0.0
        return self.total / self.count
    
    @property
    def recent_avg(self) -> float:
        """Get average of recent values in the window."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    @property
    def global_avg(self) -> float:
        """Alias for avg."""
        return self.avg
    
    @property
    def value(self) -> float:
        """Get most recent value."""
        if not self.values:
            return 0.0
        return self.values[-1]


class ExperimentTracker:
    """
    Unified interface for experiment tracking backends.
    
    Supports WandB and TensorBoard.
    """
    
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        wandb_entity: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: Name of the project.
            experiment_name: Name of this experiment/run.
            config: Configuration dictionary to log.
            log_dir: Directory for TensorBoard logs.
            use_wandb: Whether to use Weights & Biases.
            use_tensorboard: Whether to use TensorBoard.
            wandb_entity: WandB entity/team name.
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self._wandb_run = None
        self._tb_writer = None
        
        # Initialize WandB
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    entity=wandb_entity,
                    reinit=True,
                )
            except Exception as e:
                get_logger().warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
        
        # Initialize TensorBoard
        if use_tensorboard and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_log_dir = Path(log_dir) / "tensorboard" / experiment_name
                tb_log_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            except Exception as e:
                get_logger().warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        Log metrics to all active backends.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Global step number.
            prefix: Prefix for metric names.
        """
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to WandB
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard and self._tb_writer is not None:
            for name, value in metrics.items():
                self._tb_writer.add_scalar(name, value, step)
    
    def log_image(
        self,
        name: str,
        image: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Log an image to all active backends.
        
        Args:
            name: Image name/tag.
            image: Image data (numpy array or PIL Image).
            step: Global step number.
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log({name: wandb.Image(image)}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            import numpy as np
            if isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
                    # HWC -> CHW
                    image = np.transpose(image, (2, 0, 1))
                self._tb_writer.add_image(name, image, step)
    
    def log_confusion_matrix(
        self,
        matrix: Any,
        class_names: list,
        step: Optional[int] = None,
        title: str = "Confusion Matrix",
    ) -> None:
        """Log confusion matrix as an image."""
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        self.log_image(title.replace(" ", "_"), img, step)
        plt.close(fig)
    
    def log_artifact(
        self,
        path: Union[str, Path],
        name: str,
        artifact_type: str = "model",
    ) -> None:
        """
        Log an artifact (file) to WandB.
        
        Args:
            path: Path to the artifact file.
            name: Artifact name.
            artifact_type: Type of artifact (model, dataset, etc.).
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish tracking and close all backends."""
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.finish()
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.close()
