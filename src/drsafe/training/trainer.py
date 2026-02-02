"""
Trainer class for DR-SAFE pipeline.

Provides a comprehensive training loop with mixed precision, gradient accumulation,
EMA, and extensive logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from drsafe.data.transforms import MixUpCutMix, create_mixup_cutmix
from drsafe.models.backbones import EMAModel
from drsafe.training.callbacks import (
    CallbackHandler,
    EarlyStopping,
    GradientClipping,
    ModelCheckpoint,
)
from drsafe.training.losses import DRLoss, create_loss_function
from drsafe.training.metrics import DRMetrics
from drsafe.utils.config import Config
from drsafe.utils.io import ensure_dir
from drsafe.utils.logging import ExperimentTracker, MetricLogger, get_logger

logger = get_logger()


class Trainer:
    """
    Trainer for Diabetic Retinopathy classification models.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Exponential Moving Average (EMA)
    - MixUp/CutMix augmentation
    - Comprehensive metrics tracking
    - Experiment logging (WandB, TensorBoard)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        fold: int = 0,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            config: Configuration object.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            fold: Current fold number (for cross-validation).
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.fold = fold
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf") if config.training.monitor_mode == "max" else float("inf")
        
        # Components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()
        self._setup_ema()
        self._setup_mixup()
        self._setup_callbacks()
        self._setup_experiment_tracking()
        
        # Metrics
        self.metrics = DRMetrics(num_classes=config.model.num_classes)
        self.metric_logger = MetricLogger()
        
        # Number of batches for progress logging
        self.num_batches = len(train_loader)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Training batches: {self.num_batches}, Accumulation steps: {config.training.accumulation_steps}")
    
    def _setup_optimizer(self) -> None:
        """Set up optimizer."""
        cfg = self.config.training
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        if cfg.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.learning_rate,
                betas=(0.9, 0.999),
            )
        elif cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                param_groups,
                lr=cfg.learning_rate,
                betas=(0.9, 0.999),
            )
        elif cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=cfg.learning_rate,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _setup_scheduler(self) -> None:
        """Set up learning rate scheduler."""
        cfg = self.config.training
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // cfg.accumulation_steps
        total_steps = steps_per_epoch * cfg.epochs
        warmup_steps = steps_per_epoch * cfg.warmup_epochs
        
        if cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=cfg.min_lr,
            )
        elif cfg.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=steps_per_epoch * 10,
                gamma=0.1,
            )
        elif cfg.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.monitor_mode,
                factor=0.5,
                patience=5,
                min_lr=cfg.min_lr,
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        if cfg.warmup_epochs > 0 and cfg.scheduler != "plateau":
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            self.warmup_scheduler = None
    
    def _setup_loss(self) -> None:
        """Set up loss function."""
        from drsafe.data.dataset import compute_class_weights
        
        # Compute class weights from training data
        class_weights = None
        if self.config.loss.use_class_weights:
            train_labels = []
            for batch in self.train_loader:
                train_labels.extend(batch["severity_label"].numpy())
            
            class_weights = compute_class_weights(
                labels=torch.tensor(train_labels).numpy(),
                num_classes=self.config.model.num_classes,
            )
        
        self.criterion = create_loss_function(
            config=self.config.loss,
            num_classes=self.config.model.num_classes,
            class_weights=class_weights,
            use_ordinal=self.config.model.use_ordinal,
        )
    
    def _setup_amp(self) -> None:
        """Set up automatic mixed precision."""
        if self.config.training.use_amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
    
    def _setup_ema(self) -> None:
        """Set up exponential moving average."""
        if self.config.model.use_ema:
            self.ema = EMAModel(
                self.model,
                decay=self.config.model.ema_decay,
                device=self.device,
            )
            logger.info(f"EMA enabled with decay={self.config.model.ema_decay}")
        else:
            self.ema = None
    
    def _setup_mixup(self) -> None:
        """Set up MixUp/CutMix augmentation."""
        self.mixup = create_mixup_cutmix(self.config.augmentation)
        if self.mixup is not None:
            logger.info("MixUp/CutMix augmentation enabled")
    
    def _setup_callbacks(self) -> None:
        """Set up training callbacks."""
        self.callbacks = CallbackHandler()
        
        # Early stopping
        self.callbacks.add(EarlyStopping(
            monitor=f"val_{self.config.training.monitor_metric.replace('val_', '')}",
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta,
            mode=self.config.training.monitor_mode,
        ))
        
        # Model checkpoint
        checkpoint_dir = Path(self.config.logging.checkpoint_dir) / f"fold_{self.fold}"
        self.callbacks.add(ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=f"val_{self.config.training.monitor_metric.replace('val_', '')}",
            mode=self.config.training.monitor_mode,
            save_top_k=self.config.training.save_top_k,
        ))
        
        # Gradient clipping
        self.gradient_clipper = GradientClipping(
            max_norm=self.config.training.grad_clip_norm
        )
    
    def _setup_experiment_tracking(self) -> None:
        """Set up experiment tracking."""
        experiment_name = f"{self.config.logging.experiment_name}_fold{self.fold}"
        
        self.tracker = ExperimentTracker(
            project_name=self.config.logging.project_name,
            experiment_name=experiment_name,
            config=self.config.to_dict(),
            log_dir=self.config.logging.log_dir,
            use_wandb=self.config.logging.use_wandb,
            use_tensorboard=self.config.logging.use_tensorboard,
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.metrics.reset()
        self.metric_logger.reset()
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=self.num_batches,
            desc=f"Epoch {self.epoch + 1}/{self.config.training.epochs}",
        )
        
        for batch_idx, batch in pbar:
            self.callbacks.on_batch_start(self, batch_idx)
            
            # Move data to device
            images = batch["image"].to(self.device)
            severity_labels = batch["severity_label"].to(self.device)
            referable_labels = batch["referable_label"].to(self.device)
            
            # Apply MixUp/CutMix
            mixup_lambda = None
            severity_labels_2 = None
            referable_labels_2 = None
            
            if self.mixup is not None and self.model.training:
                # Convert labels to one-hot for mixing
                severity_onehot = torch.nn.functional.one_hot(
                    severity_labels, self.config.model.num_classes
                ).float()
                
                images, severity_onehot, severity_onehot_2, referable_mixed, mixup_lambda = \
                    self.mixup(images, severity_onehot, referable_labels.float())
                
                # For ordinal/CE loss, we need class indices not one-hot
                # So we'll pass the mixed info to the loss function
                severity_labels_2 = severity_labels.clone()  # Will be shuffled by mixup
            
            # Forward pass
            with autocast(enabled=self.scaler is not None):
                severity_logits, referable_logits = self.model(images)
                
                loss_dict = self.criterion(
                    severity_logits=severity_logits,
                    referable_logits=referable_logits,
                    severity_targets=severity_labels,
                    referable_targets=referable_labels,
                    mixup_lambda=mixup_lambda,
                    severity_targets_2=severity_labels_2,
                    referable_targets_2=referable_labels_2,
                )
                
                loss = loss_dict["loss"] / self.config.training.accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulation step
            if (batch_idx + 1) % self.config.training.accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = self.gradient_clipper.clip_gradients(self.model)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                # Update warmup scheduler
                if self.warmup_scheduler is not None and self.epoch < self.config.training.warmup_epochs:
                    self.warmup_scheduler.step()
                elif self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Update metrics (without mixup for accurate tracking)
            with torch.no_grad():
                if self.config.model.use_ordinal:
                    severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
                else:
                    severity_probs = torch.softmax(severity_logits, dim=1)
                
                severity_preds = severity_probs.argmax(dim=1)
                referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
                referable_preds = (referable_probs > 0.5).long()
                
                self.metrics.update(
                    severity_preds=severity_preds,
                    severity_targets=batch["severity_label"],  # Original labels
                    referable_preds=referable_preds,
                    referable_targets=batch["referable_label"],
                    referable_probs=referable_probs,
                )
            
            # Update metric logger
            self.metric_logger.update(
                loss=loss_dict["loss"].item(),
                severity_loss=loss_dict["severity_loss"].item(),
                referable_loss=loss_dict["referable_loss"].item(),
            )
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{self.metric_logger.meters['loss'].avg:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            self.callbacks.on_batch_end(self, batch_idx, loss_dict["loss"].item())
            
            # Logging
            if (batch_idx + 1) % self.config.logging.log_every_n_steps == 0:
                self.tracker.log_metrics(
                    self.metric_logger.get_metrics(),
                    step=self.global_step,
                    prefix="train",
                )
        
        # Compute epoch metrics
        epoch_metrics = self.metrics.compute()
        epoch_metrics.update(self.metric_logger.get_metrics())
        
        return {f"train_{k}": v for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        # Use EMA weights if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            severity_labels = batch["severity_label"].to(self.device)
            referable_labels = batch["referable_label"].to(self.device)
            
            # Forward pass
            severity_logits, referable_logits = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(
                severity_logits=severity_logits,
                referable_logits=referable_logits,
                severity_targets=severity_labels,
                referable_targets=referable_labels,
            )
            
            total_loss += loss_dict["loss"].item()
            num_batches += 1
            
            # Compute predictions
            if self.config.model.use_ordinal:
                severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
            else:
                severity_probs = torch.softmax(severity_logits, dim=1)
            
            severity_preds = severity_probs.argmax(dim=1)
            referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
            referable_preds = (referable_probs > 0.5).long()
            
            # Update metrics
            self.metrics.update(
                severity_preds=severity_preds,
                severity_targets=severity_labels,
                severity_probs=severity_probs,
                referable_preds=referable_preds,
                referable_targets=referable_labels,
                referable_probs=referable_probs,
            )
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        val_metrics = self.metrics.compute()
        val_metrics["loss"] = total_loss / num_batches
        
        # Add QWK as main metric
        val_metrics["qwk"] = val_metrics["severity_qwk"]
        
        return {f"val_{k}": v for k, v in val_metrics.items()}
    
    def fit(self) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Returns:
            Dictionary of metric histories.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_qwk": [],
        }
        
        self.callbacks.on_train_start(self)
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            self.callbacks.on_epoch_start(self, epoch)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.logging.val_every_n_epochs == 0:
                self.callbacks.on_validation_start(self)
                val_metrics = self.validate()
                self.callbacks.on_validation_end(self, val_metrics)
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # ReduceLROnPlateau step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor_value = val_metrics.get(f"val_{self.config.training.monitor_metric.replace('val_', '')}")
                if monitor_value is not None:
                    self.scheduler.step(monitor_value)
            
            # Logging
            self.tracker.log_metrics(all_metrics, step=epoch)
            
            # Log to console
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_metrics.get('train_loss', 0):.4f} - "
                f"Val QWK: {val_metrics.get('val_qwk', 0):.4f} - "
                f"Val ROC-AUC: {val_metrics.get('val_referable_roc_auc', 0):.4f}"
            )
            
            # Update history
            history["train_loss"].append(train_metrics.get("train_loss", 0))
            history["val_loss"].append(val_metrics.get("val_loss", 0))
            history["val_qwk"].append(val_metrics.get("val_qwk", 0))
            
            # Callbacks
            self.callbacks.on_epoch_end(self, epoch, all_metrics)
            
            # Early stopping check
            if self.callbacks.should_stop_early():
                logger.info("Early stopping triggered")
                break
            
            # Log confusion matrix
            if val_metrics:
                conf_mat = self.metrics.get_confusion_matrix()
                self.tracker.log_confusion_matrix(
                    conf_mat,
                    class_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
                    step=epoch,
                )
        
        self.callbacks.on_train_end(self)
        self.tracker.finish()
        
        return history
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        from drsafe.utils.io import load_checkpoint
        
        checkpoint = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=str(self.device),
        )
        
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        
        logger.info(f"Resumed training from epoch {self.epoch}")
