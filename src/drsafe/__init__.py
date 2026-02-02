"""
DR-SAFE: Diabetic Retinopathy Severity Assessment Framework for Evaluation

A production-quality pipeline for diabetic retinopathy grading using deep learning.
"""

__version__ = "1.0.0"
__author__ = "DR-SAFE Team"

from drsafe.utils.config import (
    Config,
    DataConfig,
    AugmentationConfig,
    ModelConfig,
    LossConfig,
    TrainingConfig,
    InferenceConfig,
    LoggingConfig,
)
from drsafe.utils.seed import set_seed
from drsafe.models.model import DRModel, create_model, load_model_from_checkpoint

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "Config",
    "DataConfig",
    "AugmentationConfig",
    "ModelConfig",
    "LossConfig",
    "TrainingConfig",
    "InferenceConfig",
    "LoggingConfig",
    # Core
    "set_seed",
    "DRModel",
    "create_model",
    "load_model_from_checkpoint",
]
