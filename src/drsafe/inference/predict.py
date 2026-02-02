"""
Prediction utilities for DR-SAFE pipeline.

Provides inference functionality with support for batch processing,
TTA, and uncertainty estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from drsafe.data.dataset import DRInferenceDataset
from drsafe.data.transforms import get_val_transforms
from drsafe.data.preprocess_btgraham import BTGrahamPreprocessor
from drsafe.utils.config import Config
from drsafe.utils.io import find_images_in_directory, save_predictions
from drsafe.utils.logging import get_logger

logger = get_logger()


class Predictor:
    """
    Predictor class for DR classification inference.
    
    Provides a high-level interface for making predictions on images
    with optional TTA and uncertainty estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[str] = None,
        use_tta: bool = False,
        use_mc_dropout: bool = False,
        mc_samples: int = 10,
        temperature: float = 1.0,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained model.
            config: Configuration object.
            device: Device to run inference on.
            use_tta: Whether to use test-time augmentation.
            use_mc_dropout: Whether to use MC Dropout for uncertainty.
            mc_samples: Number of MC Dropout samples.
            temperature: Temperature for calibrated probabilities.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device or config.device)
        self.use_tta = use_tta
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.temperature = temperature
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up transforms
        self.transform = get_val_transforms(config.augmentation, config.data.image_size)
        
        # Set up Ben Graham preprocessor if needed
        self.btgraham = None
        if config.data.use_btgraham:
            self.btgraham = BTGrahamPreprocessor(
                target_radius=config.data.btgraham_radius,
                sigma_ratio=config.data.btgraham_sigma_ratio,
            )
        
        # TTA predictor
        self.tta_predictor = None
        if use_tta:
            from drsafe.inference.tta import TTAPredictor
            self.tta_predictor = TTAPredictor(
                model=model,
                config=config,
                device=self.device,
                tta_transforms=config.inference.tta_transforms,
            )
        
        # MC Dropout predictor
        self.mc_predictor = None
        if use_mc_dropout:
            from drsafe.inference.uncertainty import MCDropoutPredictor
            self.mc_predictor = MCDropoutPredictor(
                model=model,
                config=config,
                device=self.device,
                n_samples=mc_samples,
            )
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Tensor of shape (B, C, H, W).
        
        Returns:
            Dictionary containing predictions and probabilities.
        """
        images = images.to(self.device)
        
        if self.use_tta and self.tta_predictor is not None:
            return self.tta_predictor.predict_batch(images)
        
        if self.use_mc_dropout and self.mc_predictor is not None:
            return self.mc_predictor.predict_batch(images)
        
        # Standard inference
        severity_logits, referable_logits = self.model(images)
        
        # Apply temperature scaling
        severity_logits = severity_logits / self.temperature
        referable_logits = referable_logits / self.temperature
        
        # Compute probabilities
        if hasattr(self.model, 'use_ordinal') and self.model.use_ordinal:
            severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
        else:
            severity_probs = torch.softmax(severity_logits, dim=1)
        
        severity_preds = severity_probs.argmax(dim=1)
        referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
        referable_preds = (referable_probs > self.config.inference.referable_threshold).long()
        
        return {
            "severity_probs": severity_probs.cpu().numpy(),
            "severity_pred": severity_preds.cpu().numpy(),
            "referable_probs": referable_probs.cpu().numpy(),
            "referable_pred": referable_preds.cpu().numpy(),
        }
    
    def predict_folder(
        self,
        image_dir: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> pd.DataFrame:
        """
        Make predictions on all images in a folder.
        
        Args:
            image_dir: Directory containing images.
            output_path: Optional path to save predictions.
            batch_size: Batch size for inference.
            num_workers: Number of data loader workers.
        
        Returns:
            DataFrame with predictions.
        """
        image_dir = Path(image_dir)
        image_paths = find_images_in_directory(image_dir)
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(image_paths)} images for prediction")
        
        # Create dataset and loader
        dataset = DRInferenceDataset(
            image_paths=image_paths,
            transform=self.transform,
            btgraham_preprocessor=self.btgraham,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Collect predictions
        all_results = []
        
        for batch in tqdm(loader, desc="Predicting"):
            images = batch["image"]
            image_ids = batch["image_id"]
            image_paths_batch = batch["image_path"]
            
            # Get predictions
            preds = self.predict_batch(images)
            
            # Collect results
            for i in range(len(image_ids)):
                result = {
                    "image_id": image_ids[i],
                    "image_path": image_paths_batch[i],
                    "severity_pred": int(preds["severity_pred"][i]),
                    "referable_prob": float(preds["referable_probs"][i]),
                    "referable_pred": int(preds["referable_pred"][i]),
                }
                
                # Add severity probabilities
                for cls in range(5):
                    result[f"severity_prob_{cls}"] = float(preds["severity_probs"][i, cls])
                
                # Add uncertainty if available
                if "uncertainty" in preds:
                    result["uncertainty"] = float(preds["uncertainty"][i])
                    result["triage"] = preds.get("triage", ["UNKNOWN"])[i]
                
                all_results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save if output path provided
        if output_path is not None:
            save_predictions(df.to_dict(orient="list"), output_path, format="csv")
            logger.info(f"Predictions saved to {output_path}")
        
        return df


def predict_batch(
    model: nn.Module,
    images: torch.Tensor,
    device: str = "cuda",
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simple batch prediction function.
    
    Args:
        model: Trained model.
        images: Tensor of shape (B, C, H, W).
        device: Device to run on.
        temperature: Temperature for calibration.
    
    Returns:
        Dictionary of predictions.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)
    images = images.to(device)
    
    with torch.no_grad():
        severity_logits, referable_logits = model(images)
        
        severity_logits = severity_logits / temperature
        referable_logits = referable_logits / temperature
        
        severity_probs = torch.softmax(severity_logits, dim=1)
        severity_preds = severity_probs.argmax(dim=1)
        referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
        referable_preds = (referable_probs > 0.5).long()
    
    return {
        "severity_probs": severity_probs.cpu().numpy(),
        "severity_pred": severity_preds.cpu().numpy(),
        "referable_probs": referable_probs.cpu().numpy(),
        "referable_pred": referable_preds.cpu().numpy(),
    }


def predict_folder(
    model: nn.Module,
    config: Config,
    image_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to predict on a folder of images.
    
    Args:
        model: Trained model.
        config: Configuration object.
        image_dir: Directory containing images.
        output_path: Optional path to save predictions.
        **kwargs: Additional arguments for Predictor.
    
    Returns:
        DataFrame with predictions.
    """
    predictor = Predictor(model=model, config=config, **kwargs)
    return predictor.predict_folder(image_dir, output_path)
