#!/usr/bin/env python
"""
Explainability script for DR-SAFE pipeline.

Generate Grad-CAM visualizations to explain model predictions.

Usage:
    python scripts/explain.py --checkpoint outputs/checkpoints/best.pth --input image.png
    python scripts/explain.py --checkpoint outputs/checkpoints/best.pth --input images_dir/ --output explanations/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
import torch
import numpy as np
from PIL import Image
from typing import Optional, List
from tqdm import tqdm

app = typer.Typer(help="Generate visual explanations for DR-SAFE predictions")


SEVERITY_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}


@app.command()
def explain(
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
        help="Output directory for explanations",
    ),
    task: str = typer.Option(
        "severity",
        "--task", "-t",
        help="Task to explain: 'severity' or 'referable'",
    ),
    target_class: Optional[int] = typer.Option(
        None,
        "--target-class",
        help="Target class to explain (default: predicted class)",
    ),
    method: str = typer.Option(
        "gradcam",
        "--method", "-m",
        help="Explanation method: 'gradcam' or 'gradcam++'",
    ),
    layer_name: Optional[str] = typer.Option(
        None,
        "--layer",
        help="Target layer name (default: auto-detect)",
    ),
    image_size: int = typer.Option(
        512,
        "--image-size",
        help="Image size for processing",
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        help="Blending alpha for overlay (0-1)",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to run on",
    ),
    save_separate: bool = typer.Option(
        False,
        "--save-separate",
        help="Save original, heatmap, and overlay as separate files",
    ),
    side_by_side: bool = typer.Option(
        True,
        "--side-by-side/--overlay-only",
        help="Create side-by-side comparison image",
    ),
):
    """
    Generate Grad-CAM explanations for model predictions.
    
    Example:
        python scripts/explain.py --checkpoint model.pth --input image.png --output explanations/
    """
    from drsafe.utils.seed import set_seed
    from drsafe.utils.logging import setup_logger
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.data.transforms import get_val_transforms
    from drsafe.explain.gradcam import (
        GradCAM,
        GradCAMPlusPlus,
        get_target_layer,
        generate_heatmap,
        overlay_heatmap,
        create_side_by_side,
    )
    
    import cv2
    
    logger = setup_logger("drsafe")
    set_seed(42)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {checkpoint}")
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Get target layer
    if layer_name:
        target_layer = get_target_layer(model, layer_name)
    else:
        target_layer = get_target_layer(model)
    
    logger.info(f"Using target layer: {target_layer}")
    
    # Create Grad-CAM object
    if method == "gradcam++":
        cam_extractor = GradCAMPlusPlus(model, target_layer, use_cuda=(device.type == "cuda"))
    else:
        cam_extractor = GradCAM(model, target_layer, use_cuda=(device.type == "cuda"))
    
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
    
    # Setup output directory
    if output is None:
        output = "outputs/explanations"
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create transform
    transform = get_val_transforms(image_size=image_size)
    
    # Process images
    results = []
    
    for image_path in tqdm(image_paths, desc="Generating explanations"):
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image.resize((image_size, image_size)))
        
        # Transform for model
        transformed = transform(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(device)
        
        # Generate CAM
        output_logits, cam = cam_extractor(image_tensor, target_class, task)
        
        # Get prediction
        if task == "severity":
            probs = torch.softmax(output_logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            pred_label = SEVERITY_LABELS[pred_class]
        else:
            prob = torch.sigmoid(output_logits).item()
            pred_class = int(prob > 0.5)
            confidence = prob if pred_class == 1 else 1 - prob
            pred_label = "Referable" if pred_class == 1 else "Non-referable"
        
        # Generate visualizations
        heatmap = generate_heatmap(cam)
        overlay = overlay_heatmap(image_np, heatmap, alpha)
        
        # Save outputs
        stem = image_path.stem
        
        if save_separate:
            Image.fromarray(image_np).save(output_dir / f"{stem}_original.png")
            Image.fromarray(heatmap).save(output_dir / f"{stem}_heatmap.png")
        
        if side_by_side:
            prediction_text = f"{pred_label} ({confidence:.2f})"
            combined = create_side_by_side(image_np, heatmap, overlay, prediction_text)
            Image.fromarray(combined).save(output_dir / f"{stem}_explanation.png")
        else:
            Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
        
        results.append({
            "image": image_path.name,
            "prediction": pred_label,
            "confidence": confidence,
            "target_class": target_class if target_class is not None else pred_class,
        })
    
    # Print summary
    print("\n" + "="*60)
    print("EXPLANATION SUMMARY")
    print("="*60)
    print(f"\nProcessed {len(results)} images")
    print(f"Output directory: {output_dir}")
    print(f"Method: {method}")
    print(f"Task: {task}")
    
    if len(results) <= 10:
        print("\nResults:")
        for r in results:
            print(f"  {r['image']}: {r['prediction']} ({r['confidence']:.2f})")
    
    # Save results summary
    import json
    summary_path = output_dir / "explanation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")


@app.command()
def compare_classes(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint",
    ),
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help="Path to input image",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for comparison image",
    ),
    image_size: int = typer.Option(
        512,
        "--image-size",
        help="Image size",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to run on",
    ),
):
    """
    Generate Grad-CAM explanations for all severity classes.
    
    Example:
        python scripts/explain.py compare-classes --checkpoint model.pth --input image.png
    """
    from drsafe.utils.logging import setup_logger
    from drsafe.models.model import load_model_from_checkpoint
    from drsafe.data.transforms import get_val_transforms
    from drsafe.explain.gradcam import (
        GradCAM,
        get_target_layer,
        generate_heatmap,
        overlay_heatmap,
    )
    
    logger = setup_logger("drsafe")
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    
    # Get target layer
    target_layer = get_target_layer(model)
    cam_extractor = GradCAM(model, target_layer, use_cuda=(device.type == "cuda"))
    
    # Load image
    image = Image.open(input).convert("RGB")
    image_np = np.array(image.resize((image_size, image_size)))
    
    # Transform
    transform = get_val_transforms(image_size=image_size)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        severity_logits, _ = model(image_tensor)
        probs = torch.softmax(severity_logits, dim=1)[0]
        pred_class = probs.argmax().item()
    
    # Generate CAM for each class
    class_cams = []
    for class_idx in range(5):
        _, cam = cam_extractor(image_tensor, target_class=class_idx, task="severity")
        heatmap = generate_heatmap(cam)
        overlay = overlay_heatmap(image_np, heatmap, alpha=0.5)
        
        # Add label
        import cv2
        label = f"{SEVERITY_LABELS[class_idx]}: {probs[class_idx]:.2f}"
        if class_idx == pred_class:
            label += " *"
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        class_cams.append(overlay)
    
    # Combine into grid
    top_row = np.concatenate([image_np, class_cams[0], class_cams[1]], axis=1)
    bottom_row = np.concatenate([class_cams[2], class_cams[3], class_cams[4]], axis=1)
    combined = np.concatenate([top_row, bottom_row], axis=0)
    
    # Save
    if output is None:
        output = Path(input).stem + "_class_comparison.png"
    
    Image.fromarray(combined).save(output)
    
    print(f"Class comparison saved to {output}")
    print(f"Predicted class: {SEVERITY_LABELS[pred_class]} ({probs[pred_class]:.2f})")


if __name__ == "__main__":
    app()
