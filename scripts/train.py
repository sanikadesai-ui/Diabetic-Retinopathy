#!/usr/bin/env python
"""
Training script for DR-SAFE pipeline.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --fold 0
    python scripts/train.py --config configs/strong_aug.yaml --epochs 100
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
import yaml
from typing import Optional

app = typer.Typer(help="Train DR-SAFE models for diabetic retinopathy grading", invoke_without_command=True)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config(config: dict, **kwargs) -> dict:
    """Update config with command line arguments."""
    for key, value in kwargs.items():
        if value is None:
            continue
        
        # Handle nested keys
        if key in config:
            config[key] = value
        elif key in config.get("data", {}):
            config["data"][key] = value
        elif key in config.get("model", {}):
            config["model"][key] = value
        elif key in config.get("training", {}):
            config["training"][key] = value
        elif key in config.get("logging", {}):
            config["logging"][key] = value
    
    return config


@app.command(name="train")
@app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration YAML file",
    ),
    fold: Optional[int] = typer.Option(
        None,
        "--fold", "-f",
        help="Fold number for cross-validation (0-4)",
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs", "-e",
        help="Number of training epochs",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Learning rate",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size", "-b",
        help="Training batch size",
    ),
    backbone: Optional[str] = typer.Option(
        None,
        "--backbone",
        help="Model backbone (e.g., efficientnetv2_rw_m, convnext_base)",
    ),
    experiment_name: Optional[str] = typer.Option(
        None,
        "--name", "-n",
        help="Experiment name for logging",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-s",
        help="Random seed for reproducibility",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device", "-d",
        help="Device to train on (cuda, cpu)",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume", "-r",
        help="Path to checkpoint to resume training from",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode (single batch, no logging)",
    ),
):
    """
    Train a DR-SAFE model for diabetic retinopathy severity grading.
    
    Example:
        python scripts/train.py --config configs/default.yaml --fold 0
    """
    # If a subcommand was invoked, skip the default train behavior
    if ctx.invoked_subcommand is not None:
        return
    
    import torch
    
    from drsafe.utils.config import (
        Config, DataConfig, AugmentationConfig, ModelConfig, 
        LossConfig, TrainingConfig, LoggingConfig
    )
    from drsafe.utils.seed import set_seed
    from drsafe.utils.logging import setup_logger, ExperimentTracker
    from drsafe.data.dataset import get_dataloaders, compute_class_weights
    from drsafe.data.splits import PatientSplitter
    from drsafe.data.transforms import get_train_transforms, get_val_transforms
    from drsafe.models.model import create_model
    from drsafe.training.trainer import Trainer
    
    # Load and update configuration
    cfg = load_config(config)
    cfg = update_config(
        cfg,
        fold=fold,
        epochs=epochs,
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        backbone=backbone,
        experiment_name=experiment_name,
        seed=seed,
        device=device,
    )
    
    # Set seed
    seed_val = cfg.get("seed", 42)
    set_seed(seed_val)
    
    # Setup logger
    logger = setup_logger("drsafe", level="DEBUG" if debug else "INFO")
    
    # Create config objects
    data_cfg = DataConfig(**cfg.get("data", {}))
    aug_cfg = AugmentationConfig(**cfg.get("augmentation", {}))
    model_cfg = ModelConfig(**cfg.get("model", {}))
    loss_cfg = LossConfig(**cfg.get("loss", {}))
    train_cfg = TrainingConfig(**cfg.get("training", {}))
    log_cfg = LoggingConfig(**cfg.get("logging", {}))
    
    # Update experiment name with fold
    if fold is not None:
        log_cfg.experiment_name = f"{log_cfg.experiment_name}_fold{fold}"
    
    logger.info(f"Starting training: {log_cfg.experiment_name}")
    logger.info(f"Config: {config}")
    logger.info(f"Fold: {data_cfg.fold}")
    logger.info(f"Backbone: {model_cfg.backbone}")
    logger.info(f"Epochs: {train_cfg.epochs}")
    
    # Setup device
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data splits
    import pandas as pd
    labels_df = pd.read_csv(data_cfg.labels_file)
    
    splitter = PatientSplitter(
        n_folds=data_cfg.n_folds,
        stratify_by="severity",
        random_state=seed_val,
    )
    folds = splitter.split(labels_df)
    train_indices, val_indices = folds[fold]
    train_df = labels_df.iloc[train_indices].reset_index(drop=True)
    val_df = labels_df.iloc[val_indices].reset_index(drop=True)
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_df["level"].values)
    logger.info(f"Class weights: {class_weights}")
    
    # Create transforms
    train_transform = get_train_transforms(
        config=aug_cfg,
        image_size=data_cfg.image_size,
    )
    val_transform = get_val_transforms(
        config=aug_cfg,
        image_size=data_cfg.image_size,
    )
    
    # Create data loaders
    from drsafe.data.dataset import DRDataset, create_weighted_sampler
    from torch.utils.data import DataLoader
    from drsafe.utils.seed import get_worker_seed_fn
    
    # Create datasets
    train_dataset = DRDataset(
        df=train_df,
        image_dir=data_cfg.data_dir,
        transform=train_transform,
    )
    val_dataset = DRDataset(
        df=val_df,
        image_dir=data_cfg.data_dir,
        transform=val_transform,
    )
    
    # Create sampler for class imbalance
    sampler = None
    if data_cfg.use_weighted_sampler:
        sampler = create_weighted_sampler(
            labels=train_df["level"].values,
            num_classes=model_cfg.num_classes,
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=True,
        worker_init_fn=get_worker_seed_fn(seed_val),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
        worker_init_fn=get_worker_seed_fn(seed_val),
    )
    
    # Create model
    model = create_model(
        config=model_cfg,
        num_classes=model_cfg.num_classes,
    )
    
    # Create experiment tracker
    if not debug:
        tracker = ExperimentTracker(
            experiment_name=log_cfg.experiment_name,
            project_name=log_cfg.project_name,
            log_dir=log_cfg.log_dir,
            use_wandb=log_cfg.use_wandb,
            use_tensorboard=log_cfg.use_tensorboard,
            config=cfg,
        )
    else:
        tracker = None
    
    # Create unified Config object for Trainer
    full_config = Config(
        data=data_cfg,
        augmentation=aug_cfg,
        model=model_cfg,
        loss=loss_cfg,
        training=train_cfg,
        logging=log_cfg,
        seed=seed_val,
        device=str(device),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=full_config,
        train_loader=train_loader,
        val_loader=val_loader,
        fold=fold,
    )
    
    # Resume from checkpoint if specified
    if resume:
        trainer.load_checkpoint(resume)
        logger.info(f"Resumed from checkpoint: {resume}")
    
    # Train
    if debug:
        logger.info("Running in debug mode (1 epoch, 1 batch)")
        train_cfg.epochs = 1
    
    trainer.fit()
    
    logger.info("Training complete!")
    
    # Close tracker
    if tracker:
        tracker.close()


@app.command()
def cross_validate(
    config: str = typer.Option(
        "configs/default.yaml",
        "--config", "-c",
        help="Path to configuration YAML file",
    ),
    n_folds: int = typer.Option(
        5,
        "--n-folds",
        help="Number of folds for cross-validation",
    ),
    start_fold: int = typer.Option(
        0,
        "--start-fold",
        help="Starting fold (for resuming)",
    ),
):
    """
    Run k-fold cross-validation training.
    
    Example:
        python scripts/train.py cross-validate --config configs/default.yaml --n-folds 5
    """
    import subprocess
    
    print(f"Running {n_folds}-fold cross-validation")
    
    for fold in range(start_fold, n_folds):
        print(f"\n{'='*60}")
        print(f"Training fold {fold + 1}/{n_folds}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            __file__,
            "train",
            "--config", config,
            "--fold", str(fold),
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"Fold {fold} failed with return code {result.returncode}")
            sys.exit(result.returncode)
    
    print(f"\n{'='*60}")
    print("Cross-validation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    app()
