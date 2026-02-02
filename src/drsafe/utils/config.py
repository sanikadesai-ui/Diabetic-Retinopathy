"""
Configuration management for DR-SAFE pipeline.

Provides a hierarchical configuration system using YAML files with support for
overrides and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    root: str = "./archive"
    image_folder: str = "resized_train/resized_train"  # or "resized_train_cropped/resized_train_cropped"
    labels_file: str = "trainLabels.csv"  # or "trainLabels_cropped.csv"
    use_cropped: bool = False
    image_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    
    # Preprocessing
    use_btgraham: bool = False
    btgraham_radius: int = 300
    btgraham_sigma_ratio: float = 0.1
    
    def __post_init__(self):
        """Auto-select cropped folder/labels if use_cropped is True."""
        if self.use_cropped:
            self.image_folder = "resized_train_cropped/resized_train_cropped"
            self.labels_file = "trainLabels_cropped.csv"


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    # Basic augmentations
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_limit: int = 180
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Advanced augmentations
    use_coarse_dropout: bool = True
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_size: float = 0.1
    
    use_grid_distortion: bool = True
    use_elastic_transform: bool = False
    
    # MixUp/CutMix
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5  # Probability of applying mixup or cutmix
    
    # Normalization
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model configuration."""
    # Backbone
    backbone: str = "efficientnetv2_rw_m"  # timm model name
    pretrained: bool = True
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2
    
    # Hybrid mode (CNN + ViT fusion)
    use_hybrid: bool = False
    vit_backbone: str = "vit_base_patch16_224"
    fusion_method: str = "concat"  # concat, attention, gated
    
    # Heads
    num_classes: int = 5  # 0-4 severity levels
    use_ordinal: bool = False  # Use ordinal regression instead of classification
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class LossConfig:
    """Loss configuration."""
    # Multi-task weights
    severity_weight: float = 1.0
    referable_weight: float = 0.5
    
    # Classification loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # Per-class weights
    label_smoothing: float = 0.1
    
    # Class imbalance handling
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    
    # Ordinal regression (if enabled)
    ordinal_method: str = "coral"  # coral, corn, cumulative_link


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    epochs: int = 50
    batch_size: int = 16
    accumulation_steps: int = 2  # Gradient accumulation
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Mixed precision
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Cross-validation
    n_folds: int = 5
    train_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    val_fold: int = 4
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_qwk"
    monitor_mode: str = "max"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # TTA
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: ["hflip", "vflip", "rotate90"])
    
    # Uncertainty
    use_mc_dropout: bool = False
    mc_samples: int = 10
    
    # Triage thresholds
    certain_threshold: float = 0.15  # Entropy threshold for certainty
    referable_threshold: float = 0.5  # Probability threshold for referable
    
    # Calibration
    use_temperature_scaling: bool = True
    temperature: float = 1.0  # Will be updated after calibration


@dataclass
class LoggingConfig:
    """Logging configuration."""
    project_name: str = "dr-safe"
    experiment_name: str = "baseline"
    
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()
        
        # Update each sub-config
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
            config.data.__post_init__()
        
        if "augmentation" in config_dict:
            for key, value in config_dict["augmentation"].items():
                if hasattr(config.augmentation, key):
                    setattr(config.augmentation, key, value)
        
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "loss" in config_dict:
            for key, value in config_dict["loss"].items():
                if hasattr(config.loss, key):
                    setattr(config.loss, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if "inference" in config_dict:
            for key, value in config_dict["inference"].items():
                if hasattr(config.inference, key):
                    setattr(config.inference, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        if "seed" in config_dict:
            config.seed = config_dict["seed"]
        
        if "device" in config_dict:
            config.device = config_dict["device"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "data": self.data.__dict__,
            "augmentation": self.augmentation.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "logging": self.logging.__dict__,
            "seed": self.seed,
            "device": self.device,
        }
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Merge overrides into the current configuration."""
        config_dict = self.to_dict()
        
        def deep_update(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_update(config_dict, overrides)
        return Config.from_dict(config_dict)
    
    def get_image_dir(self) -> Path:
        """Get the full path to the image directory."""
        return Path(self.data.root) / self.data.image_folder
    
    def get_labels_path(self) -> Path:
        """Get the full path to the labels CSV file."""
        return Path(self.data.root) / self.data.labels_file
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Check paths exist
        image_dir = self.get_image_dir()
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        labels_path = self.get_labels_path()
        if not labels_path.exists():
            raise ValueError(f"Labels file not found: {labels_path}")
        
        # Check model parameters
        if self.model.num_classes != 5:
            raise ValueError("num_classes must be 5 for DR severity grading (0-4)")
        
        # Check training parameters
        if self.training.val_fold not in range(self.training.n_folds):
            raise ValueError(f"val_fold must be in range [0, {self.training.n_folds})")
        
        # Check augmentation parameters
        if self.augmentation.mixup_alpha <= 0:
            raise ValueError("mixup_alpha must be positive")
        
        if self.augmentation.cutmix_alpha <= 0:
            raise ValueError("cutmix_alpha must be positive")


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from file with optional overrides.
    
    Args:
        config_path: Path to the YAML configuration file.
        overrides: Dictionary of override values.
    
    Returns:
        Configuration object.
    """
    config = Config.from_yaml(config_path)
    
    if overrides:
        config = config.merge(overrides)
    
    return config
