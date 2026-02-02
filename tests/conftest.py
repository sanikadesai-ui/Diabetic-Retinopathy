"""
Pytest configuration and fixtures for DR-SAFE tests.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import tempfile
from PIL import Image


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_labels_df():
    """Create a sample labels dataframe."""
    data = {
        "image": [f"{i}_{side}.png" for i in range(1, 11) for side in ["left", "right"]],
        "level": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(sample_labels_df):
    """Create a temporary directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy images
        for img_name in sample_labels_df["image"]:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmpdir / img_name)
        
        yield tmpdir


@pytest.fixture
def model_config():
    """Create a model config for testing."""
    from drsafe.utils.config import ModelConfig
    
    return ModelConfig(
        backbone="efficientnet_b0",
        pretrained=False,
        num_classes=5,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )


@pytest.fixture
def small_model(model_config):
    """Create a small model for testing."""
    from drsafe.models.model import create_model
    
    return create_model(model_config, pretrained=False)


@pytest.fixture
def dummy_model():
    """Create a dummy model that returns consistent outputs."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            severity = torch.randn(batch_size, 5)
            referable = torch.randn(batch_size, 1)
            return severity, referable
    
    return DummyModel()


@pytest.fixture
def training_config():
    """Create a training config for testing."""
    from drsafe.utils.config import TrainingConfig
    
    return TrainingConfig(
        epochs=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer="adamw",
        scheduler="cosine",
        warmup_epochs=0,
        mixed_precision=False,
        use_ema=False,
    )


@pytest.fixture
def loss_config():
    """Create a loss config for testing."""
    from drsafe.utils.config import LossConfig
    
    return LossConfig(
        severity_loss="cross_entropy",
        referable_loss="bce",
        severity_weight=1.0,
        referable_weight=0.5,
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
