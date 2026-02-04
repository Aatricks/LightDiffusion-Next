"""Image-to-image processor for LightDiffusion-Next.

This processor handles image-to-image generation and upscaling
using the Ultimate SD Upscale approach.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from src.Core.PipelineContext import PipelineContext
    from src.Core.AbstractModel import AbstractModel


class Img2Img:
    """Image-to-image generation and upscaling processor.
    
    Uses Ultimate SD Upscale for high-quality image transformation
    and super-resolution.
    """
    
    # Default settings
    DEFAULT_UPSCALE_BY = 2
    DEFAULT_STEPS = 8
    DEFAULT_CFG = 6
    DEFAULT_DENOISE = 0.3
    DEFAULT_SCHEDULER = "karras"
    DEFAULT_TILE_WIDTH = 512
    DEFAULT_TILE_HEIGHT = 512
    DEFAULT_MASK_BLUR = 16
    DEFAULT_TILE_PADDING = 32
    DEFAULT_UPSCALER = "RealESRGAN_x4plus.pth"
    
    @classmethod
    def apply(
        cls,
        ctx: "PipelineContext",
        model: "AbstractModel",
        positive: Any,
        negative: Any,
        image_path: str = None,
        image_tensor: torch.Tensor = None,
        upscale_by: float = None,
        denoise: float = None,
    ) -> torch.Tensor:
        """Apply image-to-image transformation.
        
        Args:
            ctx: Pipeline context with configuration
            model: The loaded model instance
            positive: Positive conditioning
            negative: Negative conditioning
            image_path: Path to input image (used if image_tensor not provided)
            image_tensor: Input image tensor [B, H, W, C] or [H, W, C]
            upscale_by: Upscale factor (default: 2)
            denoise: Denoising strength (default: 0.3)
            
        Returns:
            Processed image tensor
        """
        logger = logging.getLogger(__name__)
        
        # Determine input source
        if image_tensor is None:
            source_path = image_path or ctx.features.img2img_image
            if source_path is None:
                raise ValueError("No input image provided for img2img")
            
            # Load image from path
            image_tensor = cls._load_image(source_path)
        
        # Determine upscale factor from context dimensions if not provided
        if upscale_by is None:
            input_w = image_tensor.shape[2] # [B, H, W, C]
            target_w = ctx.generation.width
            if target_w and target_w != input_w:
                upscale_by = target_w / input_w
                logger.info(f"Img2Img: calculated upscale_by={upscale_by:.2f} from target width {target_w}")
            else:
                upscale_by = cls.DEFAULT_UPSCALE_BY
        
        denoise = denoise or cls.DEFAULT_DENOISE
        
        # Determine model flags
        is_flux = getattr(model.capabilities, "is_flux", False)
        is_flux2 = getattr(model.capabilities, "is_flux2", False)
        
        # Adjust CFG for Flux models
        img2img_cfg = cls.DEFAULT_CFG
        if is_flux or is_flux2:
            img2img_cfg = 1.0
        
        try:
            # Import required modules
            from src.UltimateSDUpscale import UltimateSDUpscale, USDU_upscaler
            
            # Load upscaler model
            upscale_loader = USDU_upscaler.UpscaleModelLoader()
            upscale_model = upscale_loader.load_model(cls.DEFAULT_UPSCALER)[0]
            
            # Initialize Ultimate SD Upscale
            upscaler = UltimateSDUpscale.UltimateSDUpscale()
            
            # Get current seed from context
            current_seed = ctx.seed
            
            logger.info(f"Img2Img: processing with {upscale_by}x upscale, denoise={denoise}")
            
            # Run upscaling
            result = upscaler.upscale(
                upscale_by=upscale_by,
                seed=current_seed,
                steps=cls.DEFAULT_STEPS,
                cfg=img2img_cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=cls.DEFAULT_SCHEDULER,
                denoise=denoise,
                mode_type="Linear",
                tile_width=cls.DEFAULT_TILE_WIDTH,
                tile_height=cls.DEFAULT_TILE_HEIGHT,
                mask_blur=cls.DEFAULT_MASK_BLUR,
                tile_padding=cls.DEFAULT_TILE_PADDING,
                seam_fix_mode="Half Tile",
                seam_fix_denoise=0.2,
                seam_fix_width=64,
                seam_fix_mask_blur=16,
                seam_fix_padding=32,
                force_uniform_tiles="enable",
                image=image_tensor,
                model=model.model,
                positive=positive,
                negative=negative,
                vae=model.vae,
                upscale_model=upscale_model,
                pipeline=True,
            )
            
            logger.info("Img2Img: completed successfully")
            return result[0]
            
        except Exception as e:
            logger.exception(f"Img2Img failed: {e}")
            # Return original image on failure
            return image_tensor
    
    @classmethod
    def _load_image(cls, path: str) -> torch.Tensor:
        """Load an image from disk and convert to tensor.
        
        Args:
            path: Path to the image file
            
        Returns:
            Image tensor in [B, H, W, C] format, normalized to [0, 1]
        """
        img = Image.open(path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
        
        # Add batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    @classmethod
    def simple_img2img(
        cls,
        ctx: "PipelineContext",
        model: "AbstractModel",
        positive: Any,
        negative: Any,
        image_tensor: torch.Tensor,
        denoise: float = 0.75,
    ) -> dict:
        """Simple image-to-image without upscaling.
        
        Encodes the input image to latents and runs diffusion with
        the specified denoising strength.
        
        Args:
            ctx: Pipeline context
            model: The loaded model
            positive: Positive conditioning
            negative: Negative conditioning
            image_tensor: Input image tensor
            denoise: Denoising strength (0.0 = no change, 1.0 = full generation)
            
        Returns:
            Dictionary with 'samples' key containing generated latents
        """
        logger = logging.getLogger(__name__)
        
        try:
            from src.AutoEncoders import VariationalAE
            from src.sample import sampling
            from src.hidiffusion import msw_msa_attention
            
            # Determine model flags
            is_flux = getattr(model.capabilities, "is_flux", False)
            is_flux2 = getattr(model.capabilities, "is_flux2", False)
            
            # Encode image to latents
            vae_encode = VariationalAE.VAEEncode()
            latents = vae_encode.encode(vae=model.vae, pixels=image_tensor)[0]
            
            # Apply HiDiffusion optimizer (not for Flux)
            if not is_flux:
                try:
                    hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                    optimized_model = hidiff.go(model_type="auto", model=model.model)[0]
                except Exception:
                    optimized_model = model.model
            else:
                optimized_model = model.model
            
            # Run sampling with denoise < 1.0
            ksampler = sampling.KSampler()
            result = ksampler.sample(
                seed=ctx.seed,
                steps=ctx.sampling.steps,
                cfg=ctx.sampling.cfg if not is_flux else 1.0,
                sampler_name=ctx.sampling.sampler,
                scheduler=ctx.sampling.scheduler,
                denoise=denoise,
                model=optimized_model,
                positive=positive,
                negative=negative,
                latent_image=latents,
                pipeline=True,
                flux=is_flux,
                flux2=is_flux2,
                enable_multiscale=False if is_flux else ctx.sampling.enable_multiscale,
                cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
            )
            
            return result[0]
            
        except Exception as e:
            logger.exception(f"Simple img2img failed: {e}")
            raise
    
    @classmethod
    def is_enabled(cls, ctx: "PipelineContext") -> bool:
        """Check if Img2Img mode is enabled.
        
        Args:
            ctx: Pipeline context
            
        Returns:
            True if img2img mode is enabled
        """
        return ctx.features.img2img
