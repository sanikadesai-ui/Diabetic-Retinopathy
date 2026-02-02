"""Inference modules for DR-SAFE pipeline."""

from drsafe.inference.predict import Predictor, predict_batch, predict_folder
from drsafe.inference.tta import TTAPredictor, MultiScaleTTA
from drsafe.inference.uncertainty import (
    MCDropoutPredictor,
    compute_predictive_entropy,
    triage_predictions,
    TriageCategory,
)
from drsafe.inference.calibration import (
    TemperatureScaling,
    calibrate_model,
    apply_temperature_scaling,
    fit_temperature_scipy,
    save_calibration,
    load_calibration,
    reliability_diagram,
)

__all__ = [
    # Prediction
    "Predictor",
    "predict_batch",
    "predict_folder",
    # Test-Time Augmentation
    "TTAPredictor",
    "MultiScaleTTA",
    # Uncertainty Estimation
    "MCDropoutPredictor",
    "compute_predictive_entropy",
    "triage_predictions",
    "TriageCategory",
    # Calibration
    "TemperatureScaling",
    "calibrate_model",
    "apply_temperature_scaling",
    "fit_temperature_scipy",
    "save_calibration",
    "load_calibration",
    "reliability_diagram",
]
