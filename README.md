# DR-SAFE: Diabetic Retinopathy Severity Assessment and Flagging Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality deep learning pipeline for automated diabetic retinopathy (DR) severity grading from fundus images. This repository implements a competition-ready solution with strong engineering practices, reproducibility, and clinical safety features.

## 🎯 Key Features

### Multi-Task Classification
- **5-Class Severity Grading**: No DR, Mild, Moderate, Severe, Proliferative DR
- **Binary Referable DR**: Flags cases requiring specialist referral (level ≥ 2)
- **Ordinal Regression**: CORAL/CORN methods for respecting ordinal nature of DR grades

### Advanced Model Architectures
- **CNN Backbones**: EfficientNetV2, ConvNeXt, ResNet variants via `timm`
- **Vision Transformers**: ViT, DeiT, Swin Transformer support
- **Hybrid CNN+ViT**: Optional fusion model with attention-based feature combination

### Robust Training Pipeline
- **Patient-Level Splits**: Prevents data leakage between left/right eye images
- **Ben Graham Preprocessing**: Circle crop and local average subtraction
- **Strong Augmentations**: MixUp, CutMix, CoarseDropout, color jitter
- **Mixed Precision**: FP16 training with automatic gradient scaling
- **EMA**: Exponential moving average for improved generalization
- **Focal Loss**: Handles severe class imbalance

### Clinical Safety Features
- **Uncertainty Quantification**: MC Dropout for prediction confidence
- **Probability Calibration**: Temperature scaling for reliable confidence scores
- **Triage Categories**: Automatic categorization (CERTAIN_REFER, CERTAIN_NON_REFER, UNCERTAIN)
- **Explainability**: Grad-CAM visualizations for clinical interpretability

### Production Ready
- **ONNX/TorchScript Export**: Deploy to production environments
- **Comprehensive Metrics**: QWK, ROC-AUC, ECE, sensitivity/specificity
- **Experiment Tracking**: WandB and TensorBoard integration
- **Configurable**: YAML-based configuration system

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/dr-safe.git
cd dr-safe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
pip install dr-safe
```

## 🗂️ Dataset Setup

This pipeline is designed for the [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) Kaggle dataset.

### Download and Prepare Data

1. Download the dataset from Kaggle:
   - `trainLabels.csv` - Labels file
   - `resized_train_cropped/` - Preprocessed images (recommended)
   
   Or use the [tanlikesmath/diabetic-retinopathy-resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) preprocessed version.

2. Organize the data:
```
archive/
├── trainLabels.csv
├── trainLabels_cropped.csv
├── resized_train/
│   └── resized_train/
│       ├── 10_left.png
│       ├── 10_right.png
│       └── ...
└── resized_train_cropped/
    └── resized_train_cropped/
        ├── 10_left.png
        ├── 10_right.png
        └── ...
```

## 🚀 Quick Start

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml --fold 0

# Train with strong augmentation
python scripts/train.py --config configs/strong_aug.yaml --fold 0

# Train hybrid model
python scripts/train.py --config configs/hybrid.yaml --fold 0

# Run 5-fold cross-validation
python scripts/train.py cross-validate --config configs/default.yaml --n-folds 5
```

### Validation

```bash
# Validate a trained model
python scripts/validate.py --checkpoint outputs/checkpoints/best.pth --fold 0

# Validate with TTA
python scripts/validate.py --checkpoint outputs/checkpoints/best.pth --use-tta

# Validate with MC Dropout uncertainty
python scripts/validate.py --checkpoint outputs/checkpoints/best.pth --use-mc-dropout --mc-samples 10
```

### Inference

```bash
# Predict on a single image
python scripts/predict.py --checkpoint outputs/checkpoints/best.pth --input image.png

# Predict on a directory
python scripts/predict.py --checkpoint outputs/checkpoints/best.pth --input images/ --output predictions.csv

# Predict with uncertainty estimation
python scripts/predict.py --checkpoint outputs/checkpoints/best.pth --input images/ --use-mc-dropout
```

### Calibration

```bash
# Calibrate model probabilities
python scripts/calibrate.py --checkpoint outputs/checkpoints/best.pth --fold 0 --task binary

# Apply calibration to predictions
python scripts/calibrate.py apply --input predictions.csv --calibration calibration.json --output calibrated.csv
```

### Explainability

```bash
# Generate Grad-CAM explanations
python scripts/explain.py --checkpoint outputs/checkpoints/best.pth --input image.png

# Compare explanations across all classes
python scripts/explain.py compare-classes --checkpoint outputs/checkpoints/best.pth --input image.png
```

### Export for Deployment

```bash
# Export to ONNX
python scripts/export.py --checkpoint outputs/checkpoints/best.pth --format onnx

# Export to TorchScript
python scripts/export.py --checkpoint outputs/checkpoints/best.pth --format torchscript

# Benchmark exported model
python scripts/export.py benchmark --model outputs/checkpoints/best.onnx --batch-size 1
```

## 📊 Expected Results

When trained on the full dataset with the default configuration:

| Metric | Validation |
|--------|------------|
| Quadratic Weighted Kappa | 0.82-0.85 |
| Severity Accuracy | 75-80% |
| Referable ROC-AUC | 0.95-0.97 |
| Referable F1 | 0.85-0.88 |

**Note**: Actual results depend on training configuration, random seed, and fold selection.

## 📁 Project Structure

```
dr-safe/
├── configs/                 # YAML configuration files
│   ├── default.yaml        # Balanced default config
│   ├── strong_aug.yaml     # Aggressive augmentation
│   └── hybrid.yaml         # CNN+ViT hybrid model
├── scripts/                 # CLI entry points
│   ├── train.py            # Training script
│   ├── validate.py         # Validation script
│   ├── predict.py          # Inference script
│   ├── calibrate.py        # Calibration script
│   ├── export.py           # Model export script
│   └── explain.py          # Explainability script
├── src/drsafe/             # Main package
│   ├── data/               # Data loading and transforms
│   │   ├── dataset.py      # Dataset classes
│   │   ├── transforms.py   # Augmentations (MixUp, CutMix)
│   │   ├── splits.py       # Patient-level splitting
│   │   └── preprocess_btgraham.py  # Ben Graham preprocessing
│   ├── models/             # Model architectures
│   │   ├── model.py        # Main DRModel class
│   │   ├── backbones.py    # CNN/ViT backbones
│   │   ├── heads.py        # Classification heads
│   │   └── hybrid.py       # Hybrid CNN+ViT model
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # Training loop
│   │   ├── losses.py       # Loss functions
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── callbacks.py    # Training callbacks
│   ├── inference/          # Inference utilities
│   │   ├── predict.py      # Prediction classes
│   │   ├── tta.py          # Test-time augmentation
│   │   ├── uncertainty.py  # MC Dropout uncertainty
│   │   └── calibration.py  # Temperature scaling
│   ├── explain/            # Explainability
│   │   └── gradcam.py      # Grad-CAM implementation
│   └── utils/              # Utilities
│       ├── config.py       # Configuration dataclasses
│       ├── seed.py         # Reproducibility
│       ├── logging.py      # Experiment tracking
│       └── io.py           # Checkpoint management
└── tests/                  # Unit tests
```

## ⚙️ Configuration

All hyperparameters are controlled via YAML configuration files. Key sections:

```yaml
# Data configuration
data:
  image_size: 512
  train_batch_size: 16
  n_folds: 5
  btgraham_preprocessing: true

# Model configuration
model:
  backbone: "efficientnetv2_rw_m"
  drop_rate: 0.3
  use_hybrid: false
  use_ordinal: false

# Training configuration
training:
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.01
  mixed_precision: true
  use_ema: true
  early_stopping_patience: 10
```

See `configs/` for complete examples.

## 🔬 Technical Details

### Patient-Level Splitting

Images are named `{patient_id}_{left|right}.png`. We use `StratifiedGroupKFold` to ensure:
1. Both eyes from the same patient are in the same split
2. Class distribution is preserved across folds

### Ben Graham Preprocessing

Following the winning solution approach:
1. Estimate fundus radius via thresholding
2. Subtract local average color (background normalization)
3. Apply circular crop

### Ordinal Regression

DR severity is ordinal (0 < 1 < 2 < 3 < 4). We implement:
- **CORAL**: Cumulative odds ratio approach
- **CORN**: Conditional ordinal regression

### Uncertainty Estimation

MC Dropout provides epistemic uncertainty by:
1. Enabling dropout at inference time
2. Running N forward passes
3. Computing variance across predictions

### Triage System

Based on referable probability and uncertainty:
- **CERTAIN_NON_REFER**: prob < 0.3, uncertainty < 0.1
- **CERTAIN_REFER**: prob > 0.7, uncertainty < 0.1
- **UNCERTAIN**: All other cases → Flag for expert review

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=drsafe --cov-report=html

# Run specific test file
pytest tests/test_metrics.py -v

# Skip slow tests
pytest -m "not slow"
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{dr_safe,
  title = {DR-SAFE: Diabetic Retinopathy Severity Assessment and Flagging Engine},
  year = {2024},
  url = {https://github.com/yourusername/dr-safe}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is NOT approved for clinical use. Always consult qualified medical professionals for healthcare decisions.

## 🙏 Acknowledgments

- [Kaggle Diabetic Retinopathy Detection Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [albumentations](https://albumentations.ai/) - Image augmentation library
- Ben Graham's winning solution for preprocessing inspiration
