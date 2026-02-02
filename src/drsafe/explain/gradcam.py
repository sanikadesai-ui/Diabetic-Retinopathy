"""
Grad-CAM explainability for DR-SAFE pipeline.

Implements Grad-CAM and Grad-CAM++ for generating visual explanations
of model predictions, essential for clinical interpretability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from drsafe.utils.logging import get_logger

logger = get_logger()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations for CNN predictions by using the gradients
    flowing into the final convolutional layer to produce a coarse localization
    map highlighting important regions.
    
    Reference:
        Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from
        Deep Networks via Gradient-based Localization.
        https://arxiv.org/abs/1610.02391
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: bool = True,
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model.
            target_layer: The layer to compute Grad-CAM for (typically last conv layer).
            use_cuda: Whether to use CUDA.
        """
        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "severity",
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Class index to generate explanation for.
                If None, uses the predicted class.
            task: "severity" or "referable" to select which output to explain.
        
        Returns:
            Tuple of (model_output, heatmap).
            Heatmap is a numpy array of shape (H, W) with values in [0, 1].
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        input_tensor.requires_grad_(True)
        
        # Forward pass
        severity_logits, referable_logits = self.model(input_tensor)
        
        if task == "severity":
            output = severity_logits
        else:
            output = referable_logits
        
        # Determine target class
        if target_class is None:
            if task == "severity":
                target_class = output.argmax(dim=1).item()
            else:
                target_class = (output.sigmoid() > 0.5).int().item()
        
        # Backward pass
        self.model.zero_grad()
        
        if task == "severity":
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            output.backward(retain_graph=True)
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU to only keep positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze()
        if cam.dim() == 3:
            cam = cam[0]
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return output.detach().cpu(), cam


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ for improved visual explanations.
    
    Uses a weighted combination of positive partial derivatives to generate
    better localization, especially for multiple instances of objects.
    
    Reference:
        Chattopadhay, A., et al. (2018). Grad-CAM++: Generalized Gradient-based
        Visual Explanations for Deep Convolutional Networks.
        https://arxiv.org/abs/1710.11063
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "severity",
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Class index to generate explanation for.
            task: "severity" or "referable" to select which output to explain.
        
        Returns:
            Tuple of (model_output, heatmap).
        """
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
        
        input_tensor.requires_grad_(True)
        
        # Forward pass
        severity_logits, referable_logits = self.model(input_tensor)
        
        if task == "severity":
            output = severity_logits
        else:
            output = referable_logits
        
        # Determine target class
        if target_class is None:
            if task == "severity":
                target_class = output.argmax(dim=1).item()
            else:
                target_class = (output.sigmoid() > 0.5).int().item()
        
        # Backward pass
        self.model.zero_grad()
        
        if task == "severity":
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            output.backward(retain_graph=True)
        
        # Compute Grad-CAM++
        gradients = self.gradients
        activations = self.activations
        
        # Second and third order gradients for weighting
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Compute alpha weights
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Weight positive gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze()
        if cam.dim() == 3:
            cam = cam[0]
        
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return output.detach().cpu(), cam


def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Get the target layer for Grad-CAM.
    
    Args:
        model: The neural network model.
        layer_name: Name of the layer. If None, tries to find the last conv layer.
    
    Returns:
        Target layer module.
    """
    if layer_name is not None:
        # Navigate to the specified layer
        layers = layer_name.split(".")
        target = model
        for layer in layers:
            target = getattr(target, layer)
        return target
    
    # Try to automatically find the last convolutional layer
    # This works for most timm models
    if hasattr(model, "backbone"):
        backbone = model.backbone
        
        # For EfficientNet-style models
        if hasattr(backbone, "conv_head"):
            return backbone.conv_head
        
        # For ConvNeXt-style models
        if hasattr(backbone, "stages"):
            return backbone.stages[-1]
        
        # For ResNet-style models
        if hasattr(backbone, "layer4"):
            return backbone.layer4
        
        # Try getting the last feature extractor layer
        if hasattr(backbone, "features"):
            return backbone.features[-1]
    
    raise ValueError(
        "Could not automatically determine target layer. "
        "Please specify layer_name explicitly."
    )


def generate_heatmap(
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Convert Grad-CAM output to a colorized heatmap.
    
    Args:
        cam: Grad-CAM output of shape (H, W) with values in [0, 1].
        colormap: OpenCV colormap to use.
    
    Returns:
        Colorized heatmap as RGB array of shape (H, W, 3).
    """
    # Convert to uint8
    heatmap = np.uint8(255 * cam)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay heatmap on the original image.
    
    Args:
        image: Original image of shape (H, W, 3), uint8.
        heatmap: Colorized heatmap of shape (H, W, 3), uint8.
        alpha: Blending factor for heatmap (0 = only image, 1 = only heatmap).
    
    Returns:
        Blended image of shape (H, W, 3), uint8.
    """
    # Ensure same size
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Blend
    overlay = ((1 - alpha) * image + alpha * heatmap).astype(np.uint8)
    
    return overlay


def generate_explanation(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    task: str = "severity",
    method: str = "gradcam",
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
    use_cuda: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate a complete visual explanation for a prediction.
    
    Args:
        model: The neural network model.
        image: Input image (numpy array, tensor, or PIL Image).
        target_layer: Layer to compute gradients for.
        target_class: Class to explain (None = predicted class).
        task: "severity" or "referable".
        method: "gradcam" or "gradcam++".
        colormap: OpenCV colormap for heatmap.
        alpha: Blending factor for overlay.
        use_cuda: Whether to use CUDA.
    
    Returns:
        Dictionary containing:
            - "original": Original image (RGB, uint8)
            - "heatmap": Colorized heatmap (RGB, uint8)
            - "overlay": Image with heatmap overlay (RGB, uint8)
            - "cam": Raw CAM values (float, 0-1)
            - "prediction": Model prediction
    """
    # Convert image to tensor if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif isinstance(image, np.ndarray):
        image_np = image.copy()
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        # Assume tensor
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
    
    # Create tensor for model
    if isinstance(image, torch.Tensor):
        input_tensor = image.unsqueeze(0) if image.dim() == 3 else image
    else:
        # Normalize and convert to tensor
        input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0)
    
    # Create Grad-CAM object
    if method == "gradcam++":
        cam_extractor = GradCAMPlusPlus(model, target_layer, use_cuda)
    else:
        cam_extractor = GradCAM(model, target_layer, use_cuda)
    
    # Generate CAM
    output, cam = cam_extractor(input_tensor, target_class, task)
    
    # Generate visualizations
    heatmap = generate_heatmap(cam, colormap)
    overlay = overlay_heatmap(image_np, heatmap, alpha)
    
    return {
        "original": image_np,
        "heatmap": heatmap,
        "overlay": overlay,
        "cam": cam,
        "prediction": output,
    }


def batch_explanations(
    model: nn.Module,
    image_paths: List[Union[str, Path]],
    target_layer: nn.Module,
    output_dir: Union[str, Path],
    task: str = "severity",
    method: str = "gradcam",
    image_size: Tuple[int, int] = (512, 512),
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
    use_cuda: bool = True,
    save_separate: bool = True,
) -> List[Dict]:
    """
    Generate explanations for multiple images.
    
    Args:
        model: The neural network model.
        image_paths: List of image paths.
        target_layer: Layer to compute gradients for.
        output_dir: Directory to save outputs.
        task: "severity" or "referable".
        method: "gradcam" or "gradcam++".
        image_size: Size to resize images to.
        colormap: OpenCV colormap for heatmap.
        alpha: Blending factor for overlay.
        use_cuda: Whether to use CUDA.
        save_separate: Whether to save separate files for each component.
    
    Returns:
        List of result dictionaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Generating explanations"):
        image_path = Path(image_path)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(image_size)
        
        # Generate explanation
        result = generate_explanation(
            model=model,
            image=image,
            target_layer=target_layer,
            task=task,
            method=method,
            colormap=colormap,
            alpha=alpha,
            use_cuda=use_cuda,
        )
        
        # Save outputs
        stem = image_path.stem
        
        if save_separate:
            Image.fromarray(result["original"]).save(
                output_dir / f"{stem}_original.png"
            )
            Image.fromarray(result["heatmap"]).save(
                output_dir / f"{stem}_heatmap.png"
            )
        
        Image.fromarray(result["overlay"]).save(
            output_dir / f"{stem}_overlay.png"
        )
        
        # Add metadata
        result["image_path"] = str(image_path)
        results.append(result)
    
    logger.info(f"Saved {len(results)} explanations to {output_dir}")
    
    return results


def create_side_by_side(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    prediction_text: str = "",
) -> np.ndarray:
    """
    Create a side-by-side comparison image.
    
    Args:
        original: Original image.
        heatmap: Heatmap image.
        overlay: Overlay image.
        prediction_text: Optional text annotation.
    
    Returns:
        Combined image.
    """
    # Ensure same size
    h, w = original.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    overlay = cv2.resize(overlay, (w, h))
    
    # Concatenate horizontally
    combined = np.concatenate([original, heatmap, overlay], axis=1)
    
    # Add text if provided
    if prediction_text:
        combined = cv2.putText(
            combined,
            prediction_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    
    return combined


def attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Attention rollout for Vision Transformers.
    
    Aggregates attention across all layers to visualize
    what the model attends to.
    
    Args:
        model: Vision Transformer model with attention hooks.
        input_tensor: Input image tensor.
        head_fusion: How to combine attention heads ("mean", "max", "min").
        discard_ratio: Ratio of lowest attention values to discard.
    
    Returns:
        Attention map of shape (H, W).
    """
    # This is a placeholder implementation
    # Full implementation requires access to attention weights from ViT
    raise NotImplementedError(
        "Attention rollout requires model with attention hooks. "
        "Use Grad-CAM for CNN-based models."
    )
