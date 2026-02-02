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
    """Data configuration - matches YAML structure."""
    data_dir: str = "archive/resized_train_cropped/resized_train_cropped"
    labels_file: str = "archive/trainLabels_cropped.csv"
    image_size: int = 512
    num_workers: int = 4
    train_batch_size: int = 16
    val_batch_size: int = 32
    test_batch_size: int = 32
    n_folds: int = 5
    fold: int = 0
    use_weighted_sampler: bool = True
    btgraham_preprocessing: bool = True
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    """Augmentation configuration - matches transforms.py expected interface."""
    # MixUp/CutMix
    use_mixup: bool = False
    use_cutmix: bool = False
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5
    
    # Basic augmentations (boolean toggles)
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_limit: int = 30
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    # Additional augmentations
    use_grid_distortion: bool = False
    use_elastic_transform: bool = False
    use_coarse_dropout: bool = True
    coarse_dropout_max_holes: int = 8
    coarse_dropout_max_size: float = 0.1  # As fraction of image size
    
    # Normalization
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model configuration - matches YAML structure."""
    # Backbone
    backbone: str = "efficientnetv2_rw_m"
    pretrained: bool = True
    num_classes: int = 5
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2
    global_pool: str = "avg"
    
    # Hybrid mode (CNN + ViT fusion)
    use_hybrid: bool = False
    hybrid_cnn_backbone: str = "convnext_base"
    hybrid_vit_backbone: str = "vit_base_patch16_224"
    hybrid_fusion: str = "attention"  # concat, attention, gated
    
    # Ordinal regression
    use_ordinal: bool = False
    ordinal_method: str = "coral"  # coral, corn
    
    # EMA settings (expected by Trainer)
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class LossConfig:
    """Loss configuration - matches YAML structure."""
    severity_loss: str = "focal"  # focal, cross_entropy, ordinal
    referable_loss: str = "bce"  # bce, focal
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None
    label_smoothing: float = 0.1
    severity_weight: float = 1.0
    referable_weight: float = 0.5
    # Expected by Trainer
    use_class_weights: bool = True


@dataclass
class TrainingConfig:
    """Training configuration - matches Trainer expected interface."""
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_qwk"
    early_stopping_mode: str = "max"
    save_top_k: int = 3
    # Fields expected by Trainer
    monitor_mode: str = "max"  # "max" for metrics like qwk/accuracy, "min" for loss
    monitor_metric: str = "qwk"  # Metric to monitor for early stopping
    accumulation_steps: int = 1  # Gradient accumulation
    patience: int = 10  # Early stopping patience
    min_delta: float = 0.0  # Minimum improvement delta
    grad_clip_norm: float = 1.0  # Gradient clipping
    use_amp: bool = True  # Automatic mixed precision


@dataclass
class InferenceConfig:
    """Inference configuration - matches YAML structure."""
    use_tta: bool = True
    tta_transforms: List[str] = field(default_factory=lambda: ["hflip", "vflip"])
    mc_dropout_samples: int = 0  # 0 = disabled
    temperature: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration - matches YAML structure."""
    project_name: str = "dr-safe"
    experiment_name: str = "default"
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_dir: str = "outputs/logs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_interval: int = 50
    save_predictions: bool = True


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
            config.data = DataConfig(**config_dict["data"])
        
        if "augmentation" in config_dict:
            config.augmentation = AugmentationConfig(**config_dict["augmentation"])
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        
        if "loss" in config_dict:
            config.loss = LossConfig(**config_dict["loss"])
        
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        
        if "inference" in config_dict:
            config.inference = InferenceConfig(**config_dict["inference"])
        
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])
        
        if "seed" in config_dict:
            config.seed = config_dict["seed"]
        
        if "device" in config_dict:
            config.device = config_dict["device"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return {
            "data": asdict(self.data),
            "augmentation": asdict(self.augmentation),
            "model": asdict(self.model),
            "loss": asdict(self.loss),
            "training": asdict(self.training),
            "inference": asdict(self.inference),
            "logging": asdict(self.logging),
            "seed": self.seed,
            "device": self.device,
        }
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


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
    return config
