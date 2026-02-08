"""ControlNet processor for structure-preserving image generation.

Standalone implementation that doesn't require ComfyUI imports.
Provides Canny edge detection and ControlNet model loading using LightDiffusion's infrastructure.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Any, Dict, List
from PIL import Image
import logging

from src.Utilities import util
from src.Device import Device

logger = logging.getLogger(__name__)


class CannyPreprocessor:
    """Canny edge detection preprocessor for ControlNet."""
    
    @staticmethod
    def detect(image: torch.Tensor, low_threshold: int = 100, high_threshold: int = 200) -> torch.Tensor:
        """Detect edges in an image using Canny algorithm.
        
        Args:
            image: Input image tensor [B, H, W, C] in range [0, 1]
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge map tensor [B, H, W, C] in range [0, 1]
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for Canny edge detection. Install with: pip install opencv-python")
        
        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]
        results = []
        
        for i in range(batch_size):
            # Convert to numpy [H, W, C] in range [0, 255]
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Convert to grayscale if color
            if img_np.shape[-1] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np[..., 0]
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            # Convert back to [H, W, C] format with 3 channels
            edges_rgb = np.stack([edges, edges, edges], axis=-1)
            
            # Normalize to [0, 1]
            edges_tensor = torch.from_numpy(edges_rgb.astype(np.float32) / 255.0)
            results.append(edges_tensor)
        
        return torch.stack(results)


class ControlNetConditioner:
    """Lightweight ControlNet conditioner that applies preprocessing to conditioning.
    
    This implementation doesn't load a full ControlNet model - instead it prepares
    the control image for use with img2img at high denoise, which achieves similar
    structure-preserving effects.
    """
    
    def __init__(self, control_image: torch.Tensor, strength: float = 1.0):
        """Initialize the conditioner.
        
        Args:
            control_image: Preprocessed control image [B, H, W, C] or [B, C, H, W]
            strength: Control strength (0-2)
        """
        self.control_image = control_image
        self.strength = strength
        self._models = []
        
    def get_models(self) -> List:
        """Return list of models to load (for cond_util compatibility)."""
        return self._models
    
    def inference_memory_requirements(self, dtype: torch.dtype) -> int:
        """Return memory requirements (for cond_util compatibility)."""
        return 0
    
    def cleanup(self):
        """Clean up resources."""
        self.control_image = None


class ControlNetProcessor:
    """ControlNet processor using img2img with preprocessed edges.
    
    Since full ControlNet model loading requires ComfyUI dependencies,
    this implementation uses a hybrid approach: Canny edge detection + 
    high-denoise img2img, which achieves similar structure-preserving results.
    """
    
    @classmethod
    def preprocess_image(
        cls,
        image: torch.Tensor,
        preprocessor: str = "canny",
        **kwargs
    ) -> torch.Tensor:
        """Preprocess an image for structure guidance.
        
        Args:
            image: Input image tensor [B, H, W, C]
            preprocessor: Preprocessor type ("canny", "none")
            **kwargs: Preprocessor-specific arguments
            
        Returns:
            Preprocessed image tensor
        """
        if preprocessor == "canny":
            low = kwargs.get("low_threshold", 100)
            high = kwargs.get("high_threshold", 200)
            return CannyPreprocessor.detect(image, low, high)
        elif preprocessor == "none":
            return image
        else:
            logger.warning(f"Unknown preprocessor '{preprocessor}', returning original image")
            return image

    @classmethod
    def create_conditioner(
        cls,
        control_image: torch.Tensor,
        strength: float = 1.0,
    ) -> ControlNetConditioner:
        """Create a ControlNet conditioner for the control image.
        
        Args:
            control_image: Preprocessed control image
            strength: Control strength
            
        Returns:
            ControlNetConditioner object
        """
        return ControlNetConditioner(control_image, strength)


def apply_controlnet_to_img2img(
    ctx,
    model,
    positive,
    negative,
    control_image: torch.Tensor,
    strength: float = 1.0,
    original_image: Optional[torch.Tensor] = None,
    last_step: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Any]:
    """Apply ControlNet-style generation using img2img with edge guidance.
    
    This simplified ControlNet uses edge detection + img2img with controlled denoise
    to preserve input structure while allowing content changes.
    
    Key insight: We need LOW denoise to preserve structure, and blend edges with
    original image to provide both structure AND content guidance.
    
    Args:
        ctx: Pipeline context
        model: Loaded model
        positive: Positive conditioning
        negative: Negative conditioning
        control_image: Preprocessed control image (e.g., Canny edges)
        strength: How much to preserve structure (higher = more preservation)
        original_image: Original input image (required for proper guidance)
        last_step: Optional step to stop at (for refiner handoff)
        callback: Optional callback for live previews
        
    Returns:
        Generated latents and context
    """
    from src.Processors.Img2Img import Img2Img
    
    # Detect model type
    is_flux2 = getattr(model.capabilities, "is_flux2", False)
    is_flux = getattr(model.capabilities, "is_flux", False)
    
    # CRITICAL: Use LOW denoise to preserve input structure
    # ControlNet should modify the image, not regenerate from scratch
    if is_flux2 or is_flux:
        # Flux: Don't use edges at all - they cause artifacts
        # Just use original image with moderate denoise for structure preservation
        denoise = 0.55 + (strength * 0.15)  # Range: 0.55-0.7
        edge_blend = 0.0  # No edges for Flux - use original image only
    else:
        # SD1.5/SDXL: Balanced denoise - preserve structure but allow prompt changes
        denoise = 0.45 + (strength * 0.2)  # Range: 0.45-0.65
        # Blend: Balanced mix allowing both structure and color changes
        edge_blend = strength * 0.3  # Range: 0.0-0.3 for edges
    
    # Always blend edges with original for proper guidance
    if original_image is not None:
        # Blend: edges provide structure, original provides content/color reference
        input_image = control_image * edge_blend + original_image * (1.0 - edge_blend)
        logger.info(
            f"ControlNet {'Flux' if is_flux or is_flux2 else 'SD'}: "
            f"strength={strength:.2f}, denoise={denoise:.2f}, edge_blend={edge_blend:.2f}"
            + (f", last_step={last_step}" if last_step else "")
        )
    else:
        # Fallback: use edges only (not recommended)
        input_image = control_image
        logger.warning("ControlNet: No original image provided, using edges only (may not work well)")
        logger.info(
            f"ControlNet {'Flux' if is_flux or is_flux2 else 'SD'}: "
            f"strength={strength:.2f}, denoise={denoise:.2f}, edges only"
        )
    
    # Run img2img with moderate denoise to preserve structure
    latents = Img2Img.simple_img2img(
        ctx, model, positive, negative,
        image_tensor=input_image,
        denoise=denoise,
        last_step=last_step,
        callback=callback,
    )
    
    return latents, ctx


def find_controlnet_models(search_dir: str = None) -> list:
    """Find ControlNet models in the specified directory.
    
    Args:
        search_dir: Directory to search (default: ./include/controlnets)
        
    Returns:
        List of ControlNet model paths
    """
    if search_dir is None:
        search_dir = "./include/controlnets"
    
    if not os.path.exists(search_dir):
        return []
    
    models = []
    for f in os.listdir(search_dir):
        if f.endswith((".safetensors", ".pth", ".pt")):
            models.append(os.path.join(search_dir, f))
    
    return sorted(models)
