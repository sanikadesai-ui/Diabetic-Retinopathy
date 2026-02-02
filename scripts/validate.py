#!/usr/bin/env python
"""
Validation script for DR-SAFE pipeline.

Evaluate a trained model on validation or test data.

Usage:
    python -m drsafe.scripts.validate --checkpoint outputs/checkpoints/best.pth --data-dir archive/resized_train_cropped
    python -m drsafe.scripts.validate --checkpoint outputs/checkpoints/best.pth --use-tta
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

app = typer.Typer(help="Validate DR-SAFE models")


@app.command()
def validate(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to configuration YAML (optional, uses checkpoint config if not provided)",
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Path to data directory",
    ),
    labels_file: Optional[str] = typer.Option(
        None,
        "--labels-file",
        help="Path to labels CSV file",
    ),
    fold: Optional[int] = typer.Option(
        None,
        "--fold", "-f",
        help="Fold number for validation split",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for validation",
    ),
    use_tta: bool = typer.Option(
        False,
        "--use-tta",
        help="Use test-time augmentation",
    ),
    use_mc_dropout: bool = typer.Option(
        False,
        "--use-mc-dropout",
        help="Use MC Dropout for uncertainty estimation",
    ),
    mc_samples: int = typer.Option(
        10,
        "--mc-samples",
        help="Number of MC Dropout samples",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save validation results",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to run on (cuda, cpu)",
    ),
):
    """
    Validate a trained model and compute metrics.
    
    Example:
        python -m drsafe.scripts.validate --checkpoint outputs/checkpoints/best.pth --use-tta
    """
    from drsafe.data.dataset import DRDataset
    from drsafe.data.splits import PatientSplitter
    from drsafe.data.transforms import get_val_transforms
    from drsafe.inference.tta import TTAPredictor
    from drsafe.inference.uncertainty import MCDropoutPredictor
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.training.metrics import (
        DRMetrics,
        expected_calibration_error,
        quadratic_weighted_kappa,
        sensitivity_at_specificity,
        specificity_at_sensitivity,
    )
    from drsafe.utils.io import load_checkpoint
    from drsafe.utils.logging import setup_logger
    from drsafe.utils.seed import set_seed
    
    logger = setup_logger("drsafe")
    set_seed(42)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint}")
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    
    # Get config from checkpoint or file
    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        raise ValueError("No config found. Provide --config argument.")
    
    # Override with command line args
    data_cfg = cfg.get("data", {})
    if data_dir:
        data_cfg["data_dir"] = data_dir
    if labels_file:
        data_cfg["labels_file"] = labels_file
    if fold is not None:
        data_cfg["fold"] = fold
    
    model_cfg = cfg.get("model", {})
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Get validation data
    labels_df = pd.read_csv(data_cfg.get("labels_file", "archive/trainLabels_cropped.csv"))
    
    if fold is not None or data_cfg.get("fold") is not None:
        fold_num = fold if fold is not None else data_cfg.get("fold", 0)
        splitter = PatientSplitter(
            n_folds=data_cfg.get("n_folds", 5),
            stratify_by="severity",
            random_state=42,
        )
        folds = splitter.split(labels_df)
        train_idx, val_idx = folds[fold_num]
        # Reset index to ensure proper indexing
        labels_df = labels_df.reset_index(drop=True)
        val_df = labels_df.iloc[val_idx].reset_index(drop=True)
    else:
        val_df = labels_df
    
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Create dataset and loader
    val_transform = get_val_transforms(
        image_size=data_cfg.get("image_size", 512)
    )
    
    val_dataset = DRDataset(
        dataframe=val_df,
        data_dir=data_cfg.get("data_dir", "archive/resized_train_cropped/resized_train_cropped"),
        transform=val_transform,
        is_train=False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Setup predictor based on options
    if use_mc_dropout:
        logger.info(f"Using MC Dropout with {mc_samples} samples")
        predictor = MCDropoutPredictor(model, n_samples=mc_samples, device=device)
    elif use_tta:
        logger.info("Using test-time augmentation")
        predictor = TTAPredictor(model, device=device)
    else:
        predictor = None
    
    # Run validation
    metrics = DRMetrics(num_classes=5)
    
    all_severity_preds = []
    all_severity_labels = []
    all_referable_preds = []
    all_referable_labels = []
    all_uncertainties = []
    all_image_ids = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            severity_labels = batch["severity_label"]
            referable_labels = batch["referable_label"]
            image_ids = batch["image_id"]
            
            if use_mc_dropout:
                results = predictor.predict_with_uncertainty(images)
                severity_probs = results["severity_probs"]
                referable_probs = results["referable_probs"]
                severity_preds = severity_probs.argmax(dim=1)
                referable_preds = (referable_probs > 0.5).int()
                uncertainties = results["severity_uncertainty"]
            elif use_tta:
                results = predictor.predict(images)
                severity_probs = results["severity_probs"]
                referable_probs = results["referable_probs"]
                severity_preds = severity_probs.argmax(dim=1)
                referable_preds = (referable_probs > 0.5).int()
                uncertainties = torch.zeros(len(images))
            else:
                severity_logits, referable_logits = model(images)
                severity_probs = torch.softmax(severity_logits, dim=1)
                referable_probs = torch.sigmoid(referable_logits).squeeze()
                severity_preds = severity_probs.argmax(dim=1)
                referable_preds = (referable_probs > 0.5).int()
                uncertainties = torch.zeros(len(images))
            
            all_severity_preds.append(severity_preds.cpu())
            all_severity_labels.append(severity_labels)
            all_referable_preds.append(referable_preds.cpu())
            all_referable_labels.append(referable_labels)
            all_uncertainties.append(uncertainties.cpu() if torch.is_tensor(uncertainties) else torch.tensor(uncertainties))
            all_image_ids.extend(image_ids)
    
    # Concatenate all predictions
    all_severity_preds = torch.cat(all_severity_preds).numpy()
    all_severity_labels = torch.cat(all_severity_labels).numpy()
    all_referable_preds = torch.cat(all_referable_preds).numpy()
    all_referable_labels = torch.cat(all_referable_labels).numpy()
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    
    # Compute metrics
    from drsafe.training.metrics import (
        quadratic_weighted_kappa,
        expected_calibration_error,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )
    
    # Severity metrics
    qwk = quadratic_weighted_kappa(all_severity_labels, all_severity_preds)
    severity_acc = accuracy_score(all_severity_labels, all_severity_preds)
    severity_f1 = f1_score(all_severity_labels, all_severity_preds, average="macro")
    
    # Referable metrics
    referable_acc = accuracy_score(all_referable_labels, all_referable_preds)
    referable_f1 = f1_score(all_referable_labels, all_referable_preds)
    
    try:
        referable_auc = roc_auc_score(all_referable_labels, all_referable_preds)
    except:
        referable_auc = 0.0
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"\nSeverity Classification (5-class):")
    print(f"  Quadratic Weighted Kappa: {qwk:.4f}")
    print(f"  Accuracy: {severity_acc:.4f}")
    print(f"  Macro F1: {severity_f1:.4f}")
    
    print(f"\nReferable DR (binary):")
    print(f"  Accuracy: {referable_acc:.4f}")
    print(f"  F1 Score: {referable_f1:.4f}")
    print(f"  ROC-AUC: {referable_auc:.4f}")
    
    print(f"\nSeverity Confusion Matrix:")
    cm = confusion_matrix(all_severity_labels, all_severity_preds)
    print(cm)
    
    print(f"\nSeverity Classification Report:")
    print(classification_report(
        all_severity_labels, 
        all_severity_preds,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
    ))
    
    # Save results
    if output:
        results = {
            "metrics": {
                "qwk": float(qwk),
                "severity_accuracy": float(severity_acc),
                "severity_f1": float(severity_f1),
                "referable_accuracy": float(referable_acc),
                "referable_f1": float(referable_f1),
                "referable_auc": float(referable_auc),
            },
            "predictions": {
                "image_ids": all_image_ids,
                "severity_predictions": all_severity_preds.tolist(),
                "severity_labels": all_severity_labels.tolist(),
                "referable_predictions": all_referable_preds.tolist(),
                "referable_labels": all_referable_labels.tolist(),
                "uncertainties": all_uncertainties.tolist(),
            },
            "confusion_matrix": cm.tolist(),
        }
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return {
        "qwk": qwk,
        "severity_accuracy": severity_acc,
        "referable_accuracy": referable_acc,
    }


if __name__ == "__main__":
    app()
