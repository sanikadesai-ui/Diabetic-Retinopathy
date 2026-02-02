"""
Tests for data module.

Tests patient-level splitting, dataset loading, and transforms.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path


class TestPatientSplitter:
    """Tests for PatientSplitter class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        data = {
            "image": [
                "1_left.png", "1_right.png",
                "2_left.png", "2_right.png",
                "3_left.png", "3_right.png",
                "4_left.png", "4_right.png",
                "5_left.png", "5_right.png",
                "10_left.png", "10_right.png",
                "20_left.png", "20_right.png",
                "30_left.png", "30_right.png",
                "40_left.png", "40_right.png",
                "50_left.png", "50_right.png",
            ],
            "level": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        }
        return pd.DataFrame(data)
    
    def test_extract_patient_id(self, sample_dataframe):
        """Test patient ID extraction from image names."""
        from drsafe.data.splits import extract_patient_id
        
        assert extract_patient_id("1_left.png") == "1"
        assert extract_patient_id("1_right.png") == "1"
        assert extract_patient_id("123_left.jpeg") == "123"
        assert extract_patient_id("456_right.jpg") == "456"
    
    def test_patient_splitter_no_leakage(self, sample_dataframe):
        """Test that patient-level splits have no leakage."""
        from drsafe.data.splits import PatientSplitter, extract_patient_id
        
        splitter = PatientSplitter(
            labels_df=sample_dataframe,
            n_folds=5,
            seed=42,
        )
        
        for fold in range(5):
            train_df, val_df = splitter.get_fold(fold)
            
            # Extract patient IDs
            train_patients = set(train_df["image"].apply(extract_patient_id))
            val_patients = set(val_df["image"].apply(extract_patient_id))
            
            # Check no overlap
            overlap = train_patients & val_patients
            assert len(overlap) == 0, f"Patient leakage in fold {fold}: {overlap}"
    
    def test_patient_splitter_all_samples_used(self, sample_dataframe):
        """Test that all samples are used across folds."""
        from drsafe.data.splits import PatientSplitter
        
        splitter = PatientSplitter(
            labels_df=sample_dataframe,
            n_folds=5,
            seed=42,
        )
        
        all_val_indices = set()
        for fold in range(5):
            train_df, val_df = splitter.get_fold(fold)
            all_val_indices.update(val_df.index.tolist())
        
        # All samples should appear in validation at least once
        assert len(all_val_indices) == len(sample_dataframe)


class TestDataset:
    """Tests for DRDataset class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe."""
        return pd.DataFrame({
            "image": ["test1.png", "test2.png"],
            "level": [0, 2],
        })
    
    def test_dataset_returns_correct_keys(self, sample_dataframe, tmp_path):
        """Test that dataset returns expected keys."""
        # Create dummy images
        from PIL import Image
        
        for img_name in sample_dataframe["image"]:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp_path / img_name)
        
        from drsafe.data.dataset import DRDataset
        from drsafe.data.transforms import get_val_transforms
        
        transform = get_val_transforms(image_size=64)
        
        dataset = DRDataset(
            dataframe=sample_dataframe,
            data_dir=str(tmp_path),
            transform=transform,
            is_train=False,
        )
        
        sample = dataset[0]
        
        assert "image" in sample
        assert "severity_label" in sample
        assert "referable_label" in sample
        assert "image_id" in sample
    
    def test_referable_label_threshold(self, sample_dataframe, tmp_path):
        """Test that referable label is correctly computed."""
        from PIL import Image
        
        for img_name in sample_dataframe["image"]:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp_path / img_name)
        
        from drsafe.data.dataset import DRDataset
        from drsafe.data.transforms import get_val_transforms
        
        transform = get_val_transforms(image_size=64)
        
        dataset = DRDataset(
            dataframe=sample_dataframe,
            data_dir=str(tmp_path),
            transform=transform,
            is_train=False,
        )
        
        # Level 0 should be non-referable (0)
        assert dataset[0]["referable_label"] == 0
        
        # Level 2 should be referable (1)
        assert dataset[1]["referable_label"] == 1


class TestTransforms:
    """Tests for augmentation transforms."""
    
    def test_train_transforms_shape(self):
        """Test that train transforms produce correct shape."""
        from drsafe.data.transforms import get_train_transforms
        
        transform = get_train_transforms(image_size=224)
        
        # Create dummy image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        result = transform(image=image)
        
        assert result["image"].shape == (3, 224, 224)
        assert result["image"].dtype == torch.float32
    
    def test_val_transforms_shape(self):
        """Test that validation transforms produce correct shape."""
        from drsafe.data.transforms import get_val_transforms
        
        transform = get_val_transforms(image_size=224)
        
        # Create dummy image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        result = transform(image=image)
        
        assert result["image"].shape == (3, 224, 224)
        assert result["image"].dtype == torch.float32
    
    def test_mixup_produces_mixed_images(self):
        """Test that MixUp produces interpolated images."""
        from drsafe.data.transforms import MixUp
        
        mixup = MixUp(alpha=1.0, p=1.0)  # p=1.0 to always apply
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 3])
        
        mixed_images, mixed_labels_a, mixed_labels_b, lam = mixup(images, labels)
        
        assert mixed_images.shape == images.shape
        assert 0 <= lam <= 1
    
    def test_cutmix_produces_cut_images(self):
        """Test that CutMix produces images with cut regions."""
        from drsafe.data.transforms import CutMix
        
        cutmix = CutMix(alpha=1.0, p=1.0)
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        labels = torch.tensor([0, 1, 2, 3])
        
        mixed_images, mixed_labels_a, mixed_labels_b, lam = cutmix(images, labels)
        
        assert mixed_images.shape == images.shape


class TestClassWeights:
    """Tests for class weight computation."""
    
    def test_compute_class_weights(self):
        """Test class weight computation."""
        from drsafe.data.dataset import compute_class_weights
        
        # Imbalanced labels
        labels = np.array([0, 0, 0, 0, 1, 2, 2, 3, 4])
        
        weights = compute_class_weights(labels, num_classes=5)
        
        assert len(weights) == 5
        # Class 0 has most samples, should have lowest weight
        assert weights[0] < weights[4]  # Class 4 has fewer samples
