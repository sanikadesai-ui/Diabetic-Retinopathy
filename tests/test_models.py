"""
Tests for model module.

Tests model creation, forward pass, and output shapes.
"""

import pytest
import torch
import numpy as np


class TestDRModel:
    """Tests for DRModel class."""
    
    @pytest.fixture
    def model_config(self):
        """Create a model config for testing."""
        from drsafe.utils.config import ModelConfig
        
        return ModelConfig(
            backbone="efficientnet_b0",  # Smaller for testing
            pretrained=False,
            num_classes=5,
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
    
    def test_model_output_shapes(self, model_config):
        """Test that model produces correct output shapes."""
        from drsafe.models.model import create_model
        
        model = create_model(model_config, pretrained=False)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            severity_logits, referable_logits = model(x)
        
        assert severity_logits.shape == (batch_size, 5)
        assert referable_logits.shape == (batch_size, 1)
    
    def test_model_gradient_flow(self, model_config):
        """Test that gradients flow through the model."""
        from drsafe.models.model import create_model
        
        model = create_model(model_config, pretrained=False)
        model.train()
        
        x = torch.randn(2, 3, 224, 224)
        severity_logits, referable_logits = model(x)
        
        # Compute loss and backward
        loss = severity_logits.mean() + referable_logits.mean()
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestBackbones:
    """Tests for backbone wrappers."""
    
    @pytest.mark.parametrize("backbone_name", [
        "efficientnet_b0",
        "resnet18",
    ])
    def test_backbone_feature_extraction(self, backbone_name):
        """Test that backbones extract features correctly."""
        from drsafe.models.backbones import create_backbone
        
        backbone, num_features = create_backbone(
            backbone_name,
            pretrained=False,
            drop_rate=0.0,
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            features = backbone(x)
        
        assert features.shape[0] == 2
        assert features.shape[1] == num_features


class TestHeads:
    """Tests for classification heads."""
    
    def test_severity_head(self):
        """Test severity classification head."""
        from drsafe.models.heads import SeverityHead
        
        head = SeverityHead(in_features=512, num_classes=5, drop_rate=0.0)
        
        features = torch.randn(4, 512)
        
        with torch.no_grad():
            logits = head(features)
        
        assert logits.shape == (4, 5)
    
    def test_referable_head(self):
        """Test referable classification head."""
        from drsafe.models.heads import ReferableHead
        
        head = ReferableHead(in_features=512, drop_rate=0.0)
        
        features = torch.randn(4, 512)
        
        with torch.no_grad():
            logits = head(features)
        
        assert logits.shape == (4, 1)
    
    def test_ordinal_head_coral(self):
        """Test CORAL ordinal head."""
        from drsafe.models.heads import OrdinalHead
        
        head = OrdinalHead(
            in_features=512,
            num_classes=5,
            drop_rate=0.0,
            method="coral",
        )
        
        features = torch.randn(4, 512)
        
        with torch.no_grad():
            logits = head(features)
        
        # CORAL returns K-1 cumulative logits
        assert logits.shape == (4, 4)
    
    def test_ordinal_to_class_prediction(self):
        """Test ordinal logits to class prediction conversion."""
        from drsafe.models.heads import OrdinalHead
        
        head = OrdinalHead(
            in_features=512,
            num_classes=5,
            drop_rate=0.0,
            method="coral",
        )
        
        # Create logits that should predict class 2
        # Classes: 0, 1, 2, 3, 4 (5 classes, 4 thresholds)
        # P(Y > k) for k = 0, 1, 2, 3
        logits = torch.tensor([[5.0, 5.0, -5.0, -5.0]])  # High prob for >0, >1, low for >2, >3
        
        probs = torch.sigmoid(logits)
        predicted_class = head.to_class_prediction(logits)
        
        # Should predict class 2 (high prob of >0, >1, low prob of >2)
        assert predicted_class.item() == 2


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    def test_ema_updates(self):
        """Test that EMA updates model parameters."""
        from drsafe.models.backbones import EMAModel
        
        # Create simple model
        model = torch.nn.Linear(10, 5)
        ema = EMAModel(model, decay=0.99)
        
        # Store initial EMA weights
        initial_weight = ema.ema_model.weight.clone()
        
        # Modify original model
        model.weight.data += 1.0
        
        # Update EMA
        ema.update()
        
        # EMA should have moved toward new weights
        assert not torch.allclose(ema.ema_model.weight, initial_weight)
        # But should not equal new weights (due to decay)
        assert not torch.allclose(ema.ema_model.weight, model.weight)


class TestHybridModel:
    """Tests for hybrid CNN+ViT model."""
    
    @pytest.mark.slow
    def test_hybrid_model_forward(self):
        """Test hybrid model forward pass."""
        from drsafe.models.hybrid import HybridModel
        
        model = HybridModel(
            cnn_backbone="efficientnet_b0",
            vit_backbone="vit_tiny_patch16_224",
            num_classes=5,
            pretrained=False,
            fusion_method="concat",
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            severity_logits, referable_logits = model(x)
        
        assert severity_logits.shape == (2, 5)
        assert referable_logits.shape == (2, 1)
