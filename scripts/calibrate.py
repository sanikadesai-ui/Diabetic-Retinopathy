#!/usr/bin/env python
"""
Calibration script for DR-SAFE pipeline.

Calibrate model probabilities using temperature scaling.

Usage:
    python scripts/calibrate.py --checkpoint outputs/checkpoints/best.pth --fold 0
    python scripts/calibrate.py --checkpoint outputs/checkpoints/best.pth --output calibration.json
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
from typing import Optional
from tqdm import tqdm

app = typer.Typer(help="Calibrate DR-SAFE models")


@app.command()
def calibrate(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to configuration YAML",
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
        help="Batch size for calibration",
    ),
    task: str = typer.Option(
        "binary",
        "--task", "-t",
        help="Task to calibrate: 'binary' (referable) or 'multiclass' (severity)",
    ),
    output: str = typer.Option(
        "outputs/calibration.json",
        "--output", "-o",
        help="Path to save calibration results",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to run on",
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate reliability diagram",
    ),
):
    """
    Calibrate model probabilities using temperature scaling.
    
    Example:
        python scripts/calibrate.py --checkpoint model.pth --fold 0 --task binary
    """
    from drsafe.utils.seed import set_seed
    from drsafe.utils.logging import setup_logger
    from drsafe.utils.io import load_checkpoint
    from drsafe.data.dataset import DRDataset
    from drsafe.data.splits import PatientSplitter
    from drsafe.data.transforms import get_val_transforms
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.inference.calibration import (
        calibrate_model,
        save_calibration,
        reliability_diagram,
    )
    
    logger = setup_logger("drsafe")
    set_seed(42)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint and config
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    
    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        cfg = {}
    
    # Get data config
    data_cfg = cfg.get("data", {})
    data_dir = data_dir or data_cfg.get("data_dir", "archive/resized_train_cropped/resized_train_cropped")
    labels_file = labels_file or data_cfg.get("labels_file", "archive/trainLabels_cropped.csv")
    fold_num = fold if fold is not None else data_cfg.get("fold", 0)
    image_size = data_cfg.get("image_size", 512)
    
    # Load model
    logger.info(f"Loading model from: {checkpoint}")
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Get validation data
    labels_df = pd.read_csv(labels_file)
    
    splitter = PatientSplitter(
        labels_df=labels_df,
        n_folds=data_cfg.get("n_folds", 5),
        seed=42,
    )
    _, val_df = splitter.get_fold(fold_num)
    
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Create dataset and loader
    val_transform = get_val_transforms(image_size=image_size)
    
    val_dataset = DRDataset(
        dataframe=val_df,
        data_dir=data_dir,
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
    
    # Run calibration
    logger.info(f"Running temperature scaling calibration for task: {task}")
    
    optimal_temp, ece_before, ece_after = calibrate_model(
        model=model,
        val_loader=val_loader,
        device=str(device),
        task=task,
    )
    
    # Print results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"\nTask: {task}")
    print(f"Optimal temperature: {optimal_temp:.4f}")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration: {ece_after:.4f}")
    print(f"ECE improvement: {(ece_before - ece_after) / ece_before * 100:.1f}%")
    
    # Save results
    output_path = Path(output)
    save_calibration(
        temperature=optimal_temp,
        ece_before=ece_before,
        ece_after=ece_after,
        filepath=output_path,
    )
    
    # Generate reliability diagram
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # Collect predictions for plotting
            all_probs = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Collecting predictions"):
                    images = batch["image"].to(device)
                    
                    if task == "binary":
                        labels = batch["referable_label"]
                    else:
                        labels = batch["severity_label"]
                    
                    severity_logits, referable_logits = model(images)
                    
                    if task == "binary":
                        probs = torch.sigmoid(referable_logits).squeeze()
                    else:
                        probs = torch.softmax(severity_logits, dim=1)
                        probs = probs.max(dim=1).values
                    
                    all_probs.append(probs.cpu())
                    all_labels.append(labels)
            
            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()
            
            if task != "binary":
                # For multiclass, convert to binary correct/incorrect
                severity_logits, _ = model(batch["image"].to(device))
                predictions = severity_logits.argmax(dim=1).cpu().numpy()
                all_labels = (predictions == all_labels).astype(int)
            
            # Generate diagram
            fig = reliability_diagram(
                all_probs,
                all_labels,
                n_bins=10,
                title=f"Reliability Diagram - {task.capitalize()} Task",
            )
            
            plot_path = output_path.parent / f"reliability_diagram_{task}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            logger.info(f"Reliability diagram saved to {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping reliability diagram")
    
    return {
        "temperature": optimal_temp,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


@app.command()
def apply(
    input_file: str = typer.Option(
        ...,
        "--input", "-i",
        help="Path to predictions file (CSV/JSON)",
    ),
    calibration_file: str = typer.Option(
        ...,
        "--calibration",
        help="Path to calibration results JSON",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help="Path to save calibrated predictions",
    ),
    prob_column: str = typer.Option(
        "referable_probability",
        "--prob-col",
        help="Column name for probabilities",
    ),
):
    """
    Apply temperature scaling to existing predictions.
    
    Example:
        python scripts/calibrate.py apply --input preds.csv --calibration calib.json
    """
    from drsafe.inference.calibration import load_calibration
    
    # Load calibration
    calib = load_calibration(calibration_file)
    temperature = calib["temperature"]
    
    print(f"Applying temperature scaling with T={temperature:.4f}")
    
    # Load predictions
    input_path = Path(input_file)
    if input_path.suffix == ".json":
        df = pd.read_json(input_file)
    else:
        df = pd.read_csv(input_file)
    
    # Apply temperature scaling (for binary probabilities)
    # Convert probability back to logit, scale, then back to probability
    probs = df[prob_column].values
    probs = np.clip(probs, 1e-7, 1 - 1e-7)  # Avoid log(0)
    logits = np.log(probs / (1 - probs))  # Inverse sigmoid
    scaled_logits = logits / temperature
    calibrated_probs = 1 / (1 + np.exp(-scaled_logits))  # Sigmoid
    
    df[f"{prob_column}_calibrated"] = calibrated_probs
    
    # Save
    output_path = Path(output)
    if output_path.suffix == ".json":
        df.to_json(output, orient="records", indent=2)
    else:
        df.to_csv(output, index=False)
    
    print(f"Calibrated predictions saved to {output}")


if __name__ == "__main__":
    app()
