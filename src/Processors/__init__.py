"""Processors module for LightDiffusion-Next.

This module provides post-processing components that can be applied
to generated images in a clean, modular fashion:
- HiresFix: High-resolution upscaling with latent re-diffusion
- Adetailer: Automatic face/body enhancement
- Img2Img: Image-to-image generation and upscaling
- AutoHDRProcessor: HDR effect enhancement
- StableFastProcessor: Model compilation optimization
- DeepCacheProcessor: U-Net feature caching

All processors follow the same interface:
    - is_enabled(ctx) -> bool: Check if processor should run
    - process(ctx, model, **kwargs) -> ctx: Apply processing

Usage:
    from src.Processors import HiresFix, Adetailer, AutoHDRProcessor
    
    # Check and apply
    if HiresFix.is_enabled(ctx):
        latents = HiresFix.apply(latents, ctx, model, positive, negative)
    
    # Or use the unified process method
    ctx = AutoHDRProcessor.process(ctx, model)
"""

from src.Processors.HiresFix import HiresFix
from src.Processors.Adetailer import Adetailer
from src.Processors.Img2Img import Img2Img
from src.Processors.AutoHDRProcessor import AutoHDRProcessor
from src.Processors.StableFastProcessor import StableFastProcessor, DeepCacheProcessor

__all__ = [
    "HiresFix",
    "Adetailer", 
    "Img2Img",
    "AutoHDRProcessor",
    "StableFastProcessor",
    "DeepCacheProcessor",
]
