"""High-resolution fix processor for LightDiffusion-Next.

This processor upscales latents and runs an additional diffusion pass
to enhance detail at higher resolutions.
"""

import logging
import random
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from src.Core.PipelineContext import PipelineContext
    from src.Core.AbstractModel import AbstractModel


class HiresFix:
    """High-resolution fix processor.
    
    Upscales latents in latent space and runs additional sampling
    to enhance details at the higher resolution.
    """
    
    # Default settings
    DEFAULT_SCALE = 2.0
    DEFAULT_DENOISE = 0.45
    DEFAULT_STEPS_RATIO = 0.5
    DEFAULT_CFG = 8
    
    @classmethod
    def apply(
        cls,
        latents: dict,
        ctx: "PipelineContext",
        model: "AbstractModel",
        positive: Any,
        negative: Any,
        scale: float = None,
        denoise: float = None,
        steps: int = None,
    ) -> dict:
        """Apply high-resolution fix to latents.
        
        Args:
            latents: Dictionary containing 'samples' key with latent tensor
            ctx: Pipeline context with configuration
            model: The loaded model instance
            positive: Positive conditioning
            negative: Negative conditioning
            scale: Upscale factor (default: 2.0)
            denoise: Denoising strength (default: 0.45)
            steps: Number of sampling steps (default: 50% of original)
            
        Returns:
            Dictionary with upscaled and refined latents
        """
        logger = logging.getLogger(__name__)
        
        # Check if model supports hires fix
        if not model.capabilities.supports_hires_fix:
            logger.warning("Model does not support HiresFix, returning original latents")
            return latents
        
        # Determine model flags
        is_flux = getattr(model.capabilities, "is_flux", False)
        is_flux2 = getattr(model.capabilities, "is_flux2", False)
        
        # Use defaults if not specified
        scale = scale or cls.DEFAULT_SCALE
        denoise = denoise or cls.DEFAULT_DENOISE
        
        # Calculate steps - for Flux2 Klein (distilled), we can use fewer steps
        min_steps = 3 if is_flux2 else 10
        steps = steps or max(min_steps, int(ctx.sampling.steps * cls.DEFAULT_STEPS_RATIO))
        
        # Respect user CFG from context
        hires_cfg = ctx.sampling.cfg
        if is_flux or is_flux2:
            hires_cfg = 1.0
        
        try:
            # Import required modules
            from src.Utilities import upscale as upscale_module
            from src.sample import sampling
            from src.hidiffusion import msw_msa_attention
            
            # Calculate new dimensions
            new_width = int(ctx.generation.width * scale)
            new_height = int(ctx.generation.height * scale)
            
            # Get model-specific downscale factor (e.g., 8 for SD, 16 for Flux)
            downscale_factor = 8
            try:
                latent_format = model.get_model_object("latent_format")
                if hasattr(latent_format, "downscale_factor"):
                    downscale_factor = latent_format.downscale_factor
                elif hasattr(latent_format, "spacial_downscale_ratio"):
                    downscale_factor = latent_format.spacial_downscale_ratio
            except Exception:
                pass
            
            # Validate against model capabilities
            new_width, new_height = model.capabilities.validate_resolution(new_width, new_height)
            
            logger.info(f"HiresFix: upscaling from {ctx.generation.width}x{ctx.generation.height} to {new_width}x{new_height}")
            
            # Upscale latents
            latent_upscale = upscale_module.LatentUpscale()
            upscaled = latent_upscale.upscale(
                samples=latents,
                width=new_width,
                height=new_height,
                downscale_factor=downscale_factor,
            )[0]
            
            # Generate new seed for hires pass (PyTorch max: 2**63 - 1)
            hires_seed = random.randint(1, 2**63 - 1)
            
            # Apply HiDiffusion optimizer only for very high resolutions (>2048px)
            # This avoids the grid/weave artifacts reported at standard hires sizes
            if not is_flux and (new_width > 2048 or new_height > 2048):
                try:
                    hidiff_optimizer = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                    optimized_model = hidiff_optimizer.go(model_type="auto", model=model.model)[0]
                    logger.info("HiresFix: Applied HiDiffusion optimization for extreme resolution")
                except Exception:
                    optimized_model = model.model
            else:
                optimized_model = model.model
            
            # Create sampler and run hires pass
            ksampler = sampling.KSampler()
            hires_result = ksampler.sample(
                seed=hires_seed,
                steps=steps,
                cfg=hires_cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=ctx.sampling.scheduler,
                denoise=denoise,
                model=optimized_model,
                positive=positive,
                negative=negative,
                latent_image=upscaled,
                pipeline=True,
                flux=is_flux,
                flux2=is_flux2,
                # CRITICAL: Always disable multi-scale for the hires pass itself
                # Multi-scale downscales during sampling, which defeats the purpose of hires fix
                # and can introduce blurriness or artifacts.
                enable_multiscale=False,
                cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
            )
            
            logger.info("HiresFix: completed successfully")
            return hires_result[0]
            
        except Exception as e:
            logger.exception(f"HiresFix failed: {e}")
            # Return original latents on failure
            return latents
    
    @classmethod
    def apply_to_image(
        cls,
        image: torch.Tensor,
        ctx: "PipelineContext",
        model: "AbstractModel",
        positive: Any,
        negative: Any,
        scale: float = None,
    ) -> torch.Tensor:
        """Apply high-resolution fix starting from a decoded image.
        
        This encodes the image to latents, applies hires fix, then decodes.
        
        Args:
            image: Image tensor in [0, 1] range
            ctx: Pipeline context
            model: The loaded model
            positive: Positive conditioning
            negative: Negative conditioning
            scale: Upscale factor
            
        Returns:
            Enhanced image tensor
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Encode image to latents
            from src.AutoEncoders import VariationalAE
            
            vae_encode = VariationalAE.VAEEncode()
            latents = vae_encode.encode(vae=model.vae, pixels=image)[0]
            
            # Apply hires fix
            enhanced_latents = cls.apply(
                latents=latents,
                ctx=ctx,
                model=model,
                positive=positive,
                negative=negative,
                scale=scale,
            )
            
            # Decode back to image
            return model.decode(enhanced_latents["samples"])
            
        except Exception as e:
            logger.exception(f"HiresFix (image mode) failed: {e}")
            return image
    
    @classmethod
    def is_enabled(cls, ctx: "PipelineContext") -> bool:
        """Check if HiresFix should be applied based on context.
        
        Args:
            ctx: Pipeline context
            
        Returns:
            True if HiresFix should be applied
        """
        return ctx.features.hires_fix
