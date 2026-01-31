"""Processors module for LightDiffusion-Next.

This module provides post-processing components that can be applied
to generated images in a clean, modular fashion:
- HiresFix: High-resolution upscaling with latent re-diffusion
- Adetailer: Automatic face/body enhancement
- Img2Img: Image-to-image generation and upscaling

Usage:
    from src.Processors import HiresFix, Adetailer
    
    image = model.generate(ctx, positive, negative)
    if ctx.features.hires_fix:
        image = HiresFix.apply(image, ctx, model)
    if ctx.features.adetailer:
        image = Adetailer.apply(image, ctx, model)
"""

from src.Processors.HiresFix import HiresFix
from src.Processors.Adetailer import Adetailer
from src.Processors.Img2Img import Img2Img

__all__ = ["HiresFix", "Adetailer", "Img2Img"]
