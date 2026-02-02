"""
Training callbacks for DR-SAFE pipeline.

Provides callbacks for early stopping, model checkpointing, and LR scheduling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch

from drsafe.utils.io import CheckpointManager
from drsafe.utils.logging import get_logger

logger = get_logger()


class Callback:
    """Base callback class."""
    
    def on_train_start(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer: Any, batch: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, trainer: Any) -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when a metric stops improving.
    """
    
    def __init__(
        self,
        monitor: str = "val_qwk",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor.
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'max' or 'min' for the monitored metric.
            verbose: Whether to log messages.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
    
    def _is_improvement(self, value: float) -> bool:
        """Check if the value is an improvement over the best."""
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Check for improvement and update counter."""
        value = metrics.get(self.monitor)
        
        if value is None:
            logger.warning(f"Early stopping monitor '{self.monitor}' not found in metrics")
            return
        
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} improved to {value:.4f}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement in {self.monitor} for {self.counter} epochs "
                    f"(best: {self.best_value:.4f}, current: {value:.4f})"
                )
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: Stopping training after {self.patience} epochs")


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback to save the best models.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = "val_qwk",
        mode: str = "max",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            monitor: Metric to monitor for best model.
            mode: 'max' or 'min' for the monitored metric.
            save_top_k: Number of best checkpoints to keep.
            save_last: Whether to save the last checkpoint.
            verbose: Whether to log messages.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=save_top_k,
            monitor=monitor,
            mode=mode,
        )
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save checkpoint if it's one of the best."""
        value = metrics.get(self.monitor)
        
        if value is None:
            logger.warning(f"Checkpoint monitor '{self.monitor}' not found in metrics")
            return
        
        # Prepare checkpoint state
        state = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
            "config": trainer.config.to_dict(),
        }
        
        if trainer.scheduler is not None:
            state["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        if trainer.scaler is not None:
            state["scaler_state_dict"] = trainer.scaler.state_dict()
        
        if trainer.ema is not None:
            state["ema_state_dict"] = trainer.ema.state_dict()
        
        # Save checkpoint
        saved_path = self.checkpoint_manager.save(state, value, epoch)
        
        if saved_path is not None and self.verbose:
            logger.info(f"Saved checkpoint to {saved_path.name}")
        
        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            torch.save(state, last_path)


class LRSchedulerCallback(Callback):
    """
    Learning rate scheduler callback.
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        step_on_batch: bool = False,
        step_on_epoch: bool = True,
        monitor: Optional[str] = None,
    ):
        """
        Initialize LR scheduler callback.
        
        Args:
            scheduler: PyTorch learning rate scheduler.
            step_on_batch: Whether to step after each batch.
            step_on_epoch: Whether to step after each epoch.
            monitor: Metric to monitor (for ReduceLROnPlateau).
        """
        self.scheduler = scheduler
        self.step_on_batch = step_on_batch
        self.step_on_epoch = step_on_epoch
        self.monitor = monitor
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None:
        """Step scheduler after batch if configured."""
        if self.step_on_batch:
            self.scheduler.step()
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Step scheduler after epoch if configured."""
        if self.step_on_epoch:
            # Handle ReduceLROnPlateau
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.monitor and self.monitor in metrics:
                    self.scheduler.step(metrics[self.monitor])
            else:
                self.scheduler.step()


class GradientClipping(Callback):
    """
    Gradient clipping callback.
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipping.
        
        Args:
            max_norm: Maximum gradient norm.
            norm_type: Type of norm to use.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """
        Clip gradients and return the total norm.
        
        Args:
            model: Model to clip gradients for.
        
        Returns:
            Total gradient norm before clipping.
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type,
        )


class ProgressLogger(Callback):
    """
    Progress logging callback using tqdm.
    """
    
    def __init__(self, log_interval: int = 50):
        """
        Initialize progress logger.
        
        Args:
            log_interval: Number of batches between log messages.
        """
        self.log_interval = log_interval
        self.batch_losses = []
    
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Reset batch losses at epoch start."""
        self.batch_losses = []
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None:
        """Log batch progress."""
        self.batch_losses.append(loss)
        
        if (batch + 1) % self.log_interval == 0:
            avg_loss = sum(self.batch_losses[-self.log_interval:]) / self.log_interval
            lr = trainer.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Batch {batch + 1}/{trainer.num_batches} - "
                f"Loss: {avg_loss:.4f} - LR: {lr:.2e}"
            )


class CallbackHandler:
    """
    Handler for managing multiple callbacks.
    """
    
    def __init__(self, callbacks: Optional[list] = None):
        """
        Initialize callback handler.
        
        Args:
            callbacks: List of callbacks.
        """
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_start(self, trainer: Any) -> None:
        """Call all callbacks' on_train_start."""
        for callback in self.callbacks:
            callback.on_train_start(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        """Call all callbacks' on_train_end."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Call all callbacks' on_epoch_start."""
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Call all callbacks' on_epoch_end."""
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_start(self, trainer: Any, batch: int) -> None:
        """Call all callbacks' on_batch_start."""
        for callback in self.callbacks:
            callback.on_batch_start(trainer, batch)
    
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None:
        """Call all callbacks' on_batch_end."""
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, loss)
    
    def on_validation_start(self, trainer: Any) -> None:
        """Call all callbacks' on_validation_start."""
        for callback in self.callbacks:
            callback.on_validation_start(trainer)
    
    def on_validation_end(self, trainer: Any, metrics: Dict[str, float]) -> None:
        """Call all callbacks' on_validation_end."""
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)
    
    def should_stop_early(self) -> bool:
        """Check if any early stopping callback wants to stop."""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop:
                return True
        return False
