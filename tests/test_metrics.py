"""
Tests for metrics module.

Tests QWK, ECE, and other evaluation metrics.
"""

import pytest
import numpy as np
import torch


class TestQWK:
    """Tests for Quadratic Weighted Kappa."""
    
    def test_perfect_agreement(self):
        """Test QWK with perfect agreement."""
        from drsafe.training.metrics import quadratic_weighted_kappa
        
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        
        assert qwk == pytest.approx(1.0)
    
    def test_random_agreement(self):
        """Test QWK with random predictions (should be ~0)."""
        from drsafe.training.metrics import quadratic_weighted_kappa
        
        np.random.seed(42)
        y_true = np.random.randint(0, 5, size=1000)
        y_pred = np.random.randint(0, 5, size=1000)
        
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        
        # Random should be close to 0
        assert abs(qwk) < 0.1
    
    def test_qwk_penalizes_large_errors(self):
        """Test that QWK penalizes larger errors more."""
        from drsafe.training.metrics import quadratic_weighted_kappa
        
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred_small = np.array([1, 1, 1, 1, 1])  # Off by 1
        y_pred_large = np.array([4, 4, 4, 4, 4])  # Off by 4
        
        qwk_small = quadratic_weighted_kappa(y_true, y_pred_small)
        qwk_large = quadratic_weighted_kappa(y_true, y_pred_large)
        
        # Larger errors should result in lower QWK
        assert qwk_small > qwk_large


class TestECE:
    """Tests for Expected Calibration Error."""
    
    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        from drsafe.training.metrics import expected_calibration_error
        
        # Perfectly calibrated: all predictions are 0.7, 70% are positive
        probs = np.array([0.7] * 100)
        labels = np.array([1] * 70 + [0] * 30)
        
        ece, mce, _, _ = expected_calibration_error(probs, labels, n_bins=10)
        
        # Should be very small
        assert ece < 0.05
    
    def test_poor_calibration(self):
        """Test ECE with poorly calibrated predictions."""
        from drsafe.training.metrics import expected_calibration_error
        
        # Overconfident: predicts 0.9 but only 50% accuracy
        probs = np.array([0.9] * 100)
        labels = np.array([1] * 50 + [0] * 50)
        
        ece, mce, _, _ = expected_calibration_error(probs, labels, n_bins=10)
        
        # Should be high (~0.4)
        assert ece > 0.3


class TestDRMetrics:
    """Tests for DRMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create DRMetrics instance."""
        from drsafe.training.metrics import DRMetrics
        return DRMetrics(num_classes=5)
    
    def test_update_and_compute(self, metrics):
        """Test metric update and computation."""
        # Add some predictions
        severity_preds = torch.tensor([0, 1, 2, 3, 4])
        severity_labels = torch.tensor([0, 1, 2, 3, 4])
        severity_probs = torch.eye(5)
        
        referable_preds = torch.tensor([0, 0, 1, 1, 1])
        referable_labels = torch.tensor([0, 0, 1, 1, 1])
        referable_probs = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.95])
        
        metrics.update(
            severity_preds=severity_preds,
            severity_labels=severity_labels,
            severity_probs=severity_probs,
            referable_preds=referable_preds,
            referable_labels=referable_labels,
            referable_probs=referable_probs,
        )
        
        results = metrics.compute()
        
        assert "severity_qwk" in results
        assert "severity_accuracy" in results
        assert "referable_accuracy" in results
        assert results["severity_qwk"] == pytest.approx(1.0)
        assert results["severity_accuracy"] == pytest.approx(1.0)
    
    def test_reset(self, metrics):
        """Test metric reset."""
        metrics.update(
            severity_preds=torch.tensor([0]),
            severity_labels=torch.tensor([0]),
            severity_probs=torch.tensor([[1.0, 0, 0, 0, 0]]),
            referable_preds=torch.tensor([0]),
            referable_labels=torch.tensor([0]),
            referable_probs=torch.tensor([0.1]),
        )
        
        metrics.reset()
        
        # Should be empty
        assert len(metrics.severity_preds) == 0


class TestFocalLoss:
    """Tests for Focal Loss."""
    
    def test_focal_loss_reduces_easy_examples(self):
        """Test that focal loss reduces weight of easy examples."""
        from drsafe.training.losses import FocalLoss
        
        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
        focal_loss = FocalLoss(gamma=2.0, reduction="none")
        
        # Easy example (high confidence correct prediction)
        logits_easy = torch.tensor([[10.0, -10.0, -10.0]])
        labels_easy = torch.tensor([0])
        
        # Hard example (low confidence)
        logits_hard = torch.tensor([[0.1, 0.0, -0.1]])
        labels_hard = torch.tensor([0])
        
        fl_easy = focal_loss(logits_easy, labels_easy)
        fl_hard = focal_loss(logits_hard, labels_hard)
        
        ce_easy = ce_loss(logits_easy, labels_easy)
        ce_hard = ce_loss(logits_hard, labels_hard)
        
        # Focal loss should reduce easy more than hard
        fl_ratio = fl_easy / fl_hard
        ce_ratio = ce_easy / ce_hard
        
        assert fl_ratio < ce_ratio


class TestOrdinalLoss:
    """Tests for ordinal regression loss."""
    
    def test_coral_loss_computation(self):
        """Test CORAL loss computation."""
        from drsafe.training.losses import OrdinalLoss
        
        loss_fn = OrdinalLoss(num_classes=5, method="coral")
        
        # Cumulative logits (K-1 = 4 values)
        logits = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(logits, labels)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive
    
    def test_ordinal_loss_penalizes_distant_predictions(self):
        """Test that ordinal loss penalizes distant predictions more."""
        from drsafe.training.losses import OrdinalLoss
        
        loss_fn = OrdinalLoss(num_classes=5, method="coral")
        
        # True label is 0
        labels = torch.tensor([0])
        
        # Prediction close to 0 (all negative logits = high P(Y > k) is low)
        logits_close = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])
        
        # Prediction far from 0 (high positive logits = P(Y > k) is high)
        logits_far = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
        
        loss_close = loss_fn(logits_close, labels)
        loss_far = loss_fn(logits_far, labels)
        
        assert loss_close < loss_far
