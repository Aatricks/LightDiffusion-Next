"""AutoHDR processor for LightDiffusion-Next.

Applies HDR-like effects to enhance image dynamic range and color.
"""

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from src.Core.Context import Context
    from src.Core.AbstractModel import AbstractModel


class AutoHDRProcessor:
    """Automatic HDR effect processor.
    
    Wraps src/AutoHDR/ahdr.py as a standardized processor.
    """
    
    @classmethod
    def is_enabled(cls, ctx: "Context") -> bool:
        """Check if AutoHDR should be applied."""
        return getattr(ctx.generation, "autohdr", True)
    
    @classmethod
    def apply(
        cls,
        image: torch.Tensor,
        ctx: "Context" = None,
        intensity: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Apply HDR effects to an image.
        
        Args:
            image: Input image tensor
            ctx: Optional pipeline context (used for autohdr flag)
            intensity: HDR effect intensity
            **kwargs: Additional HDR parameters
            
        Returns:
            Enhanced image tensor
        """
        logger = logging.getLogger(__name__)
        
        try:
            from src.AutoHDR import ahdr
            
            hdr = ahdr.HDREffects()
            result = hdr.apply_hdr2(image, hdr_intensity=intensity)
            
            # Handle tuple/list return
            if isinstance(result, (tuple, list)):
                return result[0]
            return result
            
        except Exception as e:
            logger.warning(f"AutoHDR failed: {e}")
            return image
    
    @classmethod
    def process(
        cls,
        ctx: "Context",
        model: "AbstractModel" = None,
        **kwargs,
    ) -> "Context":
        """Process context, applying HDR to current_image.
        
        Args:
            ctx: Pipeline context with current_image
            model: Not used, included for interface compatibility
            **kwargs: Additional parameters
            
        Returns:
            Context with enhanced current_image
        """
        if not cls.is_enabled(ctx):
            return ctx
        
        if ctx.current_image is not None:
            ctx.current_image = cls.apply(ctx.current_image, ctx)
        
        return ctx
