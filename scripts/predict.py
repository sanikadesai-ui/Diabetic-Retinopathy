#!/usr/bin/env python
"""
Prediction script for DR-SAFE pipeline.

Run inference on new images or directories.

Usage:
    python scripts/predict.py --checkpoint outputs/checkpoints/best.pth --input image.png
    python scripts/predict.py --checkpoint outputs/checkpoints/best.pth --input images_dir/ --output predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
import yaml
import json
import torch
import pandas as pd
import numpy as np
from typing import Optional, List
from tqdm import tqdm

app = typer.Typer(help="Run inference with DR-SAFE models")


SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate", 
    3: "Severe",
    4: "Proliferative",
}


@app.command()
def predict(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help="Path to input image or directory",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save predictions (CSV or JSON)",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size for inference",
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
    image_size: int = typer.Option(
        512,
        "--image-size",
        help="Image size for inference",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        help="Temperature for probability calibration",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Threshold for referable DR classification",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to run on (cuda, cpu)",
    ),
    save_probabilities: bool = typer.Option(
        False,
        "--save-probs",
        help="Save full probability distributions",
    ),
):
    """
    Run inference on images and output predictions.
    
    Example:
        python scripts/predict.py --checkpoint model.pth --input images/ --output preds.csv
    """
    from drsafe.utils.seed import set_seed
    from drsafe.utils.logging import setup_logger
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.inference.predict import Predictor, predict_folder
    from drsafe.inference.tta import TTAPredictor
    from drsafe.inference.uncertainty import MCDropoutPredictor, triage_predictions
    from drsafe.inference.calibration import apply_temperature_scaling
    from drsafe.data.transforms import get_val_transforms
    
    logger = setup_logger("drsafe")
    set_seed(42)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {checkpoint}")
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Get input path(s)
    input_path = Path(input)
    
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpg")) + \
                     list(input_path.glob("*.jpeg"))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        logger.warning("No images found!")
        return
    
    # Create transform
    transform = get_val_transforms(image_size=image_size)
    
    # Setup predictor
    if use_mc_dropout:
        logger.info(f"Using MC Dropout with {mc_samples} samples")
        predictor = MCDropoutPredictor(model, n_samples=mc_samples, device=device)
    elif use_tta:
        logger.info("Using test-time augmentation")
        predictor = TTAPredictor(model, device=device)
    else:
        predictor = Predictor(model, transform=transform, device=device)
    
    # Run predictions
    results = []
    
    for image_path in tqdm(image_paths, desc="Predicting"):
        from PIL import Image
        import cv2
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply transform
        transformed = transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            if use_mc_dropout:
                pred_results = predictor.predict_with_uncertainty(image_tensor)
                severity_probs = pred_results["severity_probs"].cpu().numpy()[0]
                referable_prob = pred_results["referable_probs"].cpu().numpy()[0]
                severity_uncertainty = pred_results["severity_uncertainty"].cpu().numpy()[0]
                referable_uncertainty = pred_results["referable_uncertainty"].cpu().numpy()[0]
            elif use_tta:
                pred_results = predictor.predict(image_tensor)
                severity_probs = pred_results["severity_probs"].cpu().numpy()[0]
                referable_prob = pred_results["referable_probs"].cpu().numpy()[0]
                severity_uncertainty = 0.0
                referable_uncertainty = 0.0
            else:
                severity_logits, referable_logits = model(image_tensor)
                
                # Apply temperature scaling
                if temperature != 1.0:
                    severity_probs = apply_temperature_scaling(
                        severity_logits.cpu().numpy(), temperature, task="multiclass"
                    )[0]
                    referable_prob = apply_temperature_scaling(
                        referable_logits.cpu().numpy(), temperature, task="binary"
                    )[0]
                else:
                    severity_probs = torch.softmax(severity_logits, dim=1).cpu().numpy()[0]
                    referable_prob = torch.sigmoid(referable_logits).cpu().numpy()[0, 0]
                
                severity_uncertainty = 0.0
                referable_uncertainty = 0.0
        
        # Get predictions
        severity_pred = int(np.argmax(severity_probs))
        severity_label = SEVERITY_LABELS[severity_pred]
        referable_pred = int(referable_prob > threshold)
        referable_label = "Referable" if referable_pred else "Non-referable"
        
        result = {
            "image": image_path.name,
            "severity_prediction": severity_pred,
            "severity_label": severity_label,
            "severity_confidence": float(severity_probs[severity_pred]),
            "referable_prediction": referable_pred,
            "referable_label": referable_label,
            "referable_probability": float(referable_prob),
        }
        
        if use_mc_dropout:
            result["severity_uncertainty"] = float(severity_uncertainty)
            result["referable_uncertainty"] = float(referable_uncertainty)
            
            # Add triage category
            triage = triage_predictions(
                torch.tensor([[referable_prob]]),
                torch.tensor([[referable_uncertainty]]),
            )
            result["triage_category"] = triage[0].name
        
        if save_probabilities:
            for i, prob in enumerate(severity_probs):
                result[f"prob_class_{i}"] = float(prob)
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nTotal images: {len(results)}")
    print(f"\nSeverity distribution:")
    for i in range(5):
        count = (df["severity_prediction"] == i).sum()
        pct = count / len(df) * 100
        print(f"  {SEVERITY_LABELS[i]}: {count} ({pct:.1f}%)")
    
    print(f"\nReferable DR:")
    referable_count = df["referable_prediction"].sum()
    non_referable_count = len(df) - referable_count
    print(f"  Referable: {referable_count} ({referable_count/len(df)*100:.1f}%)")
    print(f"  Non-referable: {non_referable_count} ({non_referable_count/len(df)*100:.1f}%)")
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
    else:
        # Print detailed results for single image
        if len(results) == 1:
            print("\nDetailed prediction:")
            for key, value in results[0].items():
                print(f"  {key}: {value}")
        else:
            print("\nFirst 10 predictions:")
            print(df.head(10).to_string())
    
    return df


@app.command()
def batch_predict(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    csv_file: str = typer.Option(
        ...,
        "--csv",
        help="Path to CSV with image paths",
    ),
    image_column: str = typer.Option(
        "image",
        "--image-col",
        help="Column name containing image paths/names",
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Base directory for images",
    ),
    output: str = typer.Option(
        "predictions.csv",
        "--output", "-o",
        help="Output CSV path",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Batch size",
    ),
    use_tta: bool = typer.Option(
        False,
        "--use-tta",
        help="Use test-time augmentation",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device",
    ),
):
    """
    Run batch predictions from a CSV file.
    
    Example:
        python scripts/predict.py batch-predict --checkpoint model.pth --csv test.csv
    """
    from drsafe.utils.logging import setup_logger
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.inference.predict import predict_folder
    from drsafe.data.transforms import get_val_transforms
    
    logger = setup_logger("drsafe")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} entries from {csv_file}")
    
    # Get image paths
    if data_dir:
        image_paths = [Path(data_dir) / name for name in df[image_column]]
    else:
        image_paths = [Path(name) for name in df[image_column]]
    
    # Filter existing images
    existing_paths = [p for p in image_paths if p.exists()]
    logger.info(f"Found {len(existing_paths)} existing images")
    
    if len(existing_paths) == 0:
        logger.error("No images found!")
        return
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint, device=device)
    
    # Get transform
    transform = get_val_transforms(image_size=512)
    
    # Run predictions
    results = predict_folder(
        model=model,
        folder_path=existing_paths[0].parent,
        transform=transform,
        batch_size=batch_size,
        device=device,
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output, index=False)
    logger.info(f"Predictions saved to {output}")


if __name__ == "__main__":
    app()
