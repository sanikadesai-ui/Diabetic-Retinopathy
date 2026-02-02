"""
Tests for inference module.

Tests prediction, TTA, uncertainty estimation, and calibration.
"""

import pytest
import torch
import numpy as np


class TestPredictor:
    """Tests for Predictor class."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                severity = torch.randn(batch_size, 5)
                referable = torch.randn(batch_size, 1)
                return severity, referable
        
        return DummyModel()
    
    def test_predictor_output_format(self, dummy_model):
        """Test that predictor returns correct format."""
        from drsafe.inference.predict import Predictor
        from drsafe.data.transforms import get_val_transforms
        
        transform = get_val_transforms(image_size=224)
        predictor = Predictor(dummy_model, transform=transform, device="cpu")
        
        # Create dummy image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        result = predictor.predict_single(image)
        
        assert "severity_prediction" in result
        assert "severity_probabilities" in result
        assert "referable_prediction" in result
        assert "referable_probability" in result
        assert 0 <= result["severity_prediction"] <= 4


class TestTTA:
    """Tests for Test-Time Augmentation."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model."""
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Return consistent predictions based on input mean
                # to verify TTA averaging works
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                severity = mean.expand(-1, 5)
                referable = mean[:, :, 0, 0].unsqueeze(-1)
                return severity, referable
        
        return DummyModel()
    
    def test_tta_averaging(self, dummy_model):
        """Test that TTA averages predictions."""
        from drsafe.inference.tta import TTAPredictor
        
        tta = TTAPredictor(
            dummy_model,
            transforms=["hflip", "vflip"],
            device="cpu",
        )
        
        x = torch.randn(1, 3, 224, 224)
        
        result = tta.predict(x)
        
        assert "severity_probs" in result
        assert "referable_probs" in result
        assert result["severity_probs"].shape == (1, 5)
    
    def test_tta_produces_valid_probabilities(self, dummy_model):
        """Test that TTA produces valid probability distributions."""
        from drsafe.inference.tta import TTAPredictor
        
        tta = TTAPredictor(
            dummy_model,
            transforms=["hflip", "vflip", "rotate90"],
            device="cpu",
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        result = tta.predict(x)
        
        # Probabilities should sum to 1
        prob_sums = result["severity_probs"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-5)


class TestMCDropout:
    """Tests for MC Dropout uncertainty estimation."""
    
    @pytest.fixture
    def dropout_model(self):
        """Create a model with dropout."""
        class DropoutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)
                self.fc = torch.nn.Linear(224 * 224 * 3, 5)
                self.ref = torch.nn.Linear(224 * 224 * 3, 1)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                return self.fc(x), self.ref(x)
        
        return DropoutModel()
    
    def test_mc_dropout_produces_uncertainty(self, dropout_model):
        """Test that MC Dropout produces uncertainty estimates."""
        from drsafe.inference.uncertainty import MCDropoutPredictor
        
        mc_predictor = MCDropoutPredictor(
            dropout_model,
            n_samples=10,
            device="cpu",
        )
        
        x = torch.randn(1, 3, 224, 224)
        
        result = mc_predictor.predict_with_uncertainty(x)
        
        assert "severity_probs" in result
        assert "severity_uncertainty" in result
        assert "referable_uncertainty" in result
        
        # Uncertainty should be non-negative
        assert result["severity_uncertainty"].item() >= 0
        assert result["referable_uncertainty"].item() >= 0


class TestEntropy:
    """Tests for entropy computation."""
    
    def test_entropy_uniform_distribution(self):
        """Test entropy is maximum for uniform distribution."""
        from drsafe.inference.uncertainty import compute_predictive_entropy
        
        # Uniform distribution
        probs = torch.ones(1, 5) / 5
        
        entropy = compute_predictive_entropy(probs)
        max_entropy = np.log(5)
        
        assert entropy.item() == pytest.approx(max_entropy, rel=1e-5)
    
    def test_entropy_deterministic_distribution(self):
        """Test entropy is zero for deterministic distribution."""
        from drsafe.inference.uncertainty import compute_predictive_entropy
        
        # Deterministic distribution
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        
        entropy = compute_predictive_entropy(probs)
        
        assert entropy.item() == pytest.approx(0.0, abs=1e-5)


class TestTriage:
    """Tests for triage categorization."""
    
    def test_triage_certain_non_refer(self):
        """Test triage categorization for certain non-referable."""
        from drsafe.inference.uncertainty import triage_predictions, TriageCategory
        
        # Low probability, low uncertainty
        probs = torch.tensor([[0.1]])
        uncertainties = torch.tensor([[0.01]])
        
        categories = triage_predictions(probs, uncertainties)
        
        assert categories[0] == TriageCategory.CERTAIN_NON_REFER
    
    def test_triage_certain_refer(self):
        """Test triage categorization for certain referable."""
        from drsafe.inference.uncertainty import triage_predictions, TriageCategory
        
        # High probability, low uncertainty
        probs = torch.tensor([[0.95]])
        uncertainties = torch.tensor([[0.01]])
        
        categories = triage_predictions(probs, uncertainties)
        
        assert categories[0] == TriageCategory.CERTAIN_REFER
    
    def test_triage_uncertain(self):
        """Test triage categorization for uncertain cases."""
        from drsafe.inference.uncertainty import triage_predictions, TriageCategory
        
        # Medium probability, high uncertainty
        probs = torch.tensor([[0.5]])
        uncertainties = torch.tensor([[0.3]])
        
        categories = triage_predictions(probs, uncertainties)
        
        assert categories[0] == TriageCategory.UNCERTAIN


class TestCalibration:
    """Tests for calibration utilities."""
    
    def test_temperature_scaling_identity(self):
        """Test temperature scaling with T=1 is identity."""
        from drsafe.inference.calibration import TemperatureScaling
        
        temp_scaler = TemperatureScaling(init_temperature=1.0)
        
        logits = torch.randn(10, 5)
        
        scaled = temp_scaler(logits)
        
        assert torch.allclose(logits, scaled)
    
    def test_temperature_scaling_reduces_confidence(self):
        """Test that T>1 reduces confidence."""
        from drsafe.inference.calibration import apply_temperature_scaling
        
        # High confidence logits
        logits = np.array([[5.0, -5.0, -5.0, -5.0, -5.0]])
        
        probs_t1 = apply_temperature_scaling(logits, temperature=1.0, task="multiclass")
        probs_t2 = apply_temperature_scaling(logits, temperature=2.0, task="multiclass")
        
        # Higher temperature should reduce max probability
        assert probs_t1.max() > probs_t2.max()
    
    def test_temperature_scaling_increases_confidence(self):
        """Test that T<1 increases confidence."""
        from drsafe.inference.calibration import apply_temperature_scaling
        
        # Medium confidence logits
        logits = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
        
        probs_t1 = apply_temperature_scaling(logits, temperature=1.0, task="multiclass")
        probs_t05 = apply_temperature_scaling(logits, temperature=0.5, task="multiclass")
        
        # Lower temperature should increase max probability
        assert probs_t1.max() < probs_t05.max()
