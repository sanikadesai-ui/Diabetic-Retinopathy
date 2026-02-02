"""
Test-Time Augmentation (TTA) for DR-SAFE pipeline.

Provides TTA functionality to improve prediction robustness by
averaging predictions over multiple augmented versions of input images.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from drsafe.data.transforms import get_tta_transforms
from drsafe.utils.config import Config
from drsafe.utils.logging import get_logger

logger = get_logger()


class TTAPredictor:
    """
    Test-Time Augmentation predictor.
    
    Generates multiple augmented versions of each input image and
    averages the predictions for more robust results.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        tta_transforms: Optional[List[str]] = None,
    ):
        """
        Initialize TTA predictor.
        
        Args:
            model: Trained model.
            config: Configuration object.
            device: Device to run inference on.
            tta_transforms: List of TTA transform names.
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Default TTA transforms
        if tta_transforms is None:
            tta_transforms = ["original", "hflip", "vflip", "rotate90"]
        
        self.tta_transforms = tta_transforms
        
        # Create transform functions
        self.transforms = {}
        for transform_name in tta_transforms:
            self.transforms[transform_name] = get_tta_transforms(
                config.augmentation,
                config.data.image_size,
                transform_name,
            )
        
        logger.info(f"TTA enabled with transforms: {tta_transforms}")
    
    def _apply_tta_transform(
        self,
        images: torch.Tensor,
        transform_name: str,
    ) -> torch.Tensor:
        """
        Apply a TTA transform to a batch of images.
        
        Note: Images should already be normalized tensors.
        For proper TTA, we apply geometric transforms only.
        
        Args:
            images: Tensor of shape (B, C, H, W).
            transform_name: Name of the transform.
        
        Returns:
            Transformed tensor.
        """
        if transform_name == "original":
            return images
        elif transform_name == "hflip":
            return torch.flip(images, dims=[3])  # Flip along width
        elif transform_name == "vflip":
            return torch.flip(images, dims=[2])  # Flip along height
        elif transform_name == "rotate90":
            return torch.rot90(images, k=1, dims=[2, 3])
        elif transform_name == "rotate180":
            return torch.rot90(images, k=2, dims=[2, 3])
        elif transform_name == "rotate270":
            return torch.rot90(images, k=3, dims=[2, 3])
        elif transform_name == "transpose":
            return images.transpose(2, 3)
        else:
            return images
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Make TTA predictions on a batch of images.
        
        Args:
            images: Tensor of shape (B, C, H, W).
        
        Returns:
            Dictionary containing averaged predictions.
        """
        self.model.eval()
        images = images.to(self.device)
        
        all_severity_probs = []
        all_referable_probs = []
        
        for transform_name in self.tta_transforms:
            # Apply transform
            transformed = self._apply_tta_transform(images, transform_name)
            
            # Forward pass
            severity_logits, referable_logits = self.model(transformed)
            
            # Compute probabilities
            if hasattr(self.model, 'use_ordinal') and self.model.use_ordinal:
                severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
            else:
                severity_probs = torch.softmax(severity_logits, dim=1)
            
            referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
            
            all_severity_probs.append(severity_probs)
            all_referable_probs.append(referable_probs)
        
        # Average predictions
        avg_severity_probs = torch.stack(all_severity_probs).mean(dim=0)
        avg_referable_probs = torch.stack(all_referable_probs).mean(dim=0)
        
        # Compute final predictions
        severity_preds = avg_severity_probs.argmax(dim=1)
        referable_preds = (avg_referable_probs > self.config.inference.referable_threshold).long()
        
        # Compute uncertainty from TTA variance
        severity_std = torch.stack(all_severity_probs).std(dim=0).mean(dim=1)
        referable_std = torch.stack(all_referable_probs).std(dim=0)
        
        return {
            "severity_probs": avg_severity_probs.cpu().numpy(),
            "severity_pred": severity_preds.cpu().numpy(),
            "referable_probs": avg_referable_probs.cpu().numpy(),
            "referable_pred": referable_preds.cpu().numpy(),
            "severity_uncertainty": severity_std.cpu().numpy(),
            "referable_uncertainty": referable_std.cpu().numpy(),
        }
    
    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Make TTA prediction on a single image.
        
        Args:
            image: Tensor of shape (C, H, W) or (1, C, H, W).
        
        Returns:
            Dictionary containing prediction for single image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        result = self.predict_batch(image)
        
        return {
            "severity_probs": result["severity_probs"][0].tolist(),
            "severity_pred": int(result["severity_pred"][0]),
            "referable_prob": float(result["referable_probs"][0]),
            "referable_pred": int(result["referable_pred"][0]),
            "severity_uncertainty": float(result["severity_uncertainty"][0]),
            "referable_uncertainty": float(result["referable_uncertainty"][0]),
        }


class MultiScaleTTA(TTAPredictor):
    """
    Multi-scale TTA with different input resolutions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: torch.device,
        scales: Optional[List[float]] = None,
        tta_transforms: Optional[List[str]] = None,
    ):
        """
        Initialize multi-scale TTA predictor.
        
        Args:
            model: Trained model.
            config: Configuration object.
            device: Device to run inference on.
            scales: List of scale factors (e.g., [0.8, 1.0, 1.2]).
            tta_transforms: List of TTA transform names.
        """
        super().__init__(model, config, device, tta_transforms)
        
        if scales is None:
            scales = [1.0]  # Default to single scale
        
        self.scales = scales
        logger.info(f"Multi-scale TTA enabled with scales: {scales}")
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Make multi-scale TTA predictions on a batch.
        
        Args:
            images: Tensor of shape (B, C, H, W).
        
        Returns:
            Dictionary containing averaged predictions.
        """
        self.model.eval()
        images = images.to(self.device)
        
        all_severity_probs = []
        all_referable_probs = []
        
        original_size = images.shape[-2:]
        
        for scale in self.scales:
            if scale != 1.0:
                scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_images = torch.nn.functional.interpolate(
                    images,
                    size=scaled_size,
                    mode="bilinear",
                    align_corners=False,
                )
                # Resize back to original for model input
                scaled_images = torch.nn.functional.interpolate(
                    scaled_images,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scaled_images = images
            
            # Apply TTA transforms at this scale
            for transform_name in self.tta_transforms:
                transformed = self._apply_tta_transform(scaled_images, transform_name)
                
                severity_logits, referable_logits = self.model(transformed)
                
                if hasattr(self.model, 'use_ordinal') and self.model.use_ordinal:
                    severity_probs = self.model.head.severity_head.predict_proba(severity_logits)
                else:
                    severity_probs = torch.softmax(severity_logits, dim=1)
                
                referable_probs = torch.sigmoid(referable_logits.squeeze(-1))
                
                all_severity_probs.append(severity_probs)
                all_referable_probs.append(referable_probs)
        
        # Average predictions
        avg_severity_probs = torch.stack(all_severity_probs).mean(dim=0)
        avg_referable_probs = torch.stack(all_referable_probs).mean(dim=0)
        
        severity_preds = avg_severity_probs.argmax(dim=1)
        referable_preds = (avg_referable_probs > self.config.inference.referable_threshold).long()
        
        # Compute uncertainty
        severity_std = torch.stack(all_severity_probs).std(dim=0).mean(dim=1)
        referable_std = torch.stack(all_referable_probs).std(dim=0)
        
        return {
            "severity_probs": avg_severity_probs.cpu().numpy(),
            "severity_pred": severity_preds.cpu().numpy(),
            "referable_probs": avg_referable_probs.cpu().numpy(),
            "referable_pred": referable_preds.cpu().numpy(),
            "severity_uncertainty": severity_std.cpu().numpy(),
            "referable_uncertainty": referable_std.cpu().numpy(),
        }
