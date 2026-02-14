"""SDXL model adapter for LightDiffusion-Next.

Provides a clean interface to SDXL models that inherits from
AbstractModel and wraps the existing infrastructure.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from src.Core.AbstractModel import AbstractModel, ModelCapabilities

if TYPE_CHECKING:
    from src.Core.Context import Context


class SDXLModel(AbstractModel):
    """SDXL model implementation.
    
    Wraps the existing SDXL model loading and inference code
    with the clean AbstractModel interface.
    
    Note: SDXL uses dual CLIP (L + G) and requires size conditioning.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the SDXL model adapter.
        
        Args:
            model_path: Path to the model checkpoint (safetensors/pt)
        """
        super().__init__(model_path)
        self._clip_skip = -2
    
    def _create_capabilities(self) -> ModelCapabilities:
        """Create capabilities for SDXL models."""
        return ModelCapabilities(
            min_resolution=512,
            max_resolution=4096,
            preferred_resolution=1024,
            requires_resolution_multiple=64,
            supports_hires_fix=True,
            supports_img2img=True,
            supports_inpainting=True,
            supports_controlnet=True,
            supports_stable_fast=True,
            supports_deepcache=True,
            supports_tome=True,
            supports_lora=True,
            uses_dual_clip=True,
            requires_size_conditioning=True,
        )
    
    def load(self, model_path: str = None) -> "SDXLModel":
        """Load the SDXL model from disk.
        
        Args:
            model_path: Optional override for the model path
            
        Returns:
            Self for method chaining
        """
        logger = logging.getLogger(__name__)
        
        path = model_path or self.model_path
        if path is None:
            # Use default SDXL checkpoint
            path = "./include/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
        
        # Guard: Don't reload if already loaded with same path
        if self._loaded and self.model_path == path:
            logger.info(f"SDXLModel: Already loaded {path}, skipping redundant load")
            return self
            
        self.model_path = path
        
        try:
            from src.FileManaging import Loader
            
            loader = Loader.CheckpointLoaderSimple()
            result = loader.load_checkpoint(ckpt_name=path)
            
            self.model = result[0]
            self.clip = result[1]
            self.vae = result[2]
            self._loaded = True
            
            logger.info(f"SDXLModel: loaded {path}")
            
        except Exception as e:
            logger.exception(f"SDXLModel: failed to load {path}: {e}")
            raise
        
        return self
    
    def get_model_object(self, name: str) -> Any:
        """Get an attribute from the underlying model."""
        if self.model:
            return self.model.get_model_object(name)
        return None

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        clip_skip: int = None,
    ) -> tuple[Any, Any]:
        """Encode text prompts into conditioning tensors.
        
        SDXL uses dual CLIP encoders (L + G) which are handled
        internally by the existing infrastructure.
        
        Args:
            prompt: Positive prompt(s) to encode
            negative_prompt: Negative prompt(s) to encode
            clip_skip: Number of CLIP layers to skip (default: -2)
            
        Returns:
            Tuple of (positive_conditioning, negative_conditioning)
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before encoding prompts")
        
        clip_skip = clip_skip if clip_skip is not None else self._clip_skip
        
        try:
            from src.clip import Clip
            
            # Apply CLIP skip
            clip_layer = Clip.CLIPSetLastLayer()
            processed_clip = clip_layer.set_last_layer(
                stop_at_clip_layer=clip_skip,
                clip=self.clip,
            )[0]
            
            # Encode prompts
            encoder = Clip.CLIPTextEncode()
            
            positive = encoder.encode(
                text=prompt,
                clip=processed_clip,
            )[0]
            
            negative = encoder.encode(
                text=negative_prompt,
                clip=processed_clip,
            )[0]
            
            return positive, negative
            
        except Exception as e:
            logging.getLogger(__name__).exception(f"Prompt encoding failed: {e}")
            raise
    
    def generate(
        self,
        ctx: "Context",
        positive: Any,
        negative: Any,
        latent_image: Optional[Any] = None,
        start_step: Optional[int] = None,
        last_step: Optional[int] = None,
        disable_noise: bool = False,
        callback: Optional[Callable] = None,
    ) -> dict:
        """Generate latents using the sampler.
        
        Args:
            ctx: Pipeline context with generation parameters
            positive: Positive conditioning
            negative: Negative conditioning
            latent_image: Optional existing latent to continue from
            start_step: Optional step to start sampling from
            last_step: Optional step to stop sampling at
            
        Returns:
            Dictionary with 'samples' key containing generated latents
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before generating")
        
        # Validate resolution for SDXL
        width, height = self.capabilities.validate_resolution(
            ctx.generation.width,
            ctx.generation.height
        )
        
        # Log if resolution was adjusted
        if width != ctx.generation.width or height != ctx.generation.height:
            logging.getLogger(__name__).info(
                f"SDXL: adjusted resolution from {ctx.generation.width}x{ctx.generation.height} "
                f"to {width}x{height}"
            )
        
        # Inject size conditioning into positive and negative conditioning
        for cond_list in [positive, negative]:
            for cond_item in cond_list:
                if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                    cond_item[1].update({
                        "width": width,
                        "height": height,
                        "crop_w": 0,
                        "crop_h": 0,
                        "target_width": width,
                        "target_height": height,
                    })
        
        try:
            from src.sample import sampling
            from src.Utilities import Latent
            from src.hidiffusion import msw_msa_attention
            
            # Use provided latent or create empty one
            if latent_image is not None:
                latent = latent_image
            else:
                # Create empty latent with validated dimensions
                latent_gen = Latent.EmptyLatentImage()
                latent = latent_gen.generate(
                    width=width,
                    height=height,
                    batch_size=ctx.generation.batch,
                )[0]
                
                # Add seeds for deterministic noise
                latent["seeds"] = ctx.seeds[:ctx.generation.batch] if ctx.seeds else [ctx.seed]
            
            # Apply HiDiffusion optimization
            try:
                # Clone model before patching to avoid persistent state
                patch_model = self.model.clone()
                hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                optimized_model = hidiff.go(model_type="sdxl", model=patch_model)[0]
            except Exception:
                optimized_model = self.model
            
            # Run sampling
            ksampler = sampling.KSampler()
            result = ksampler.sample(
                seed=ctx.seed,
                steps=ctx.sampling.steps,
                cfg=ctx.sampling.cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=ctx.sampling.scheduler,
                denoise=ctx.sampling.denoise,
                pipeline=True,
                model=optimized_model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                start_step=start_step,
                last_step=last_step,
                disable_noise=disable_noise,
                callback=callback or ctx.callback,
                enable_multiscale=ctx.sampling.enable_multiscale,
                multiscale_factor=ctx.sampling.multiscale_factor,
                multiscale_fullres_start=ctx.sampling.multiscale_fullres_start,
                multiscale_fullres_end=ctx.sampling.multiscale_fullres_end,
                multiscale_intermittent_fullres=ctx.sampling.multiscale_intermittent_fullres,
                cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
                batched_cfg=ctx.sampling.batched_cfg,
                dynamic_cfg_rescaling=ctx.sampling.dynamic_cfg_rescaling,
                dynamic_cfg_method=ctx.sampling.dynamic_cfg_method,
                dynamic_cfg_percentile=ctx.sampling.dynamic_cfg_percentile,
                dynamic_cfg_target_scale=ctx.sampling.dynamic_cfg_target_scale,
                adaptive_noise_enabled=ctx.sampling.adaptive_noise_enabled,
                adaptive_noise_method=ctx.sampling.adaptive_noise_method,
            )
            
            return result[0]
            
        except Exception as e:
            logging.getLogger(__name__).exception(f"Generation failed: {e}")
            raise
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space.
        
        Args:
            latents: Latent tensor or dict with 'samples' key
            
        Returns:
            Decoded image tensor in [0, 1] range
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before decoding")
        
        try:
            from src.AutoEncoders import VariationalAE
            
            decoder = VariationalAE.VAEDecode()
            
            # Handle both raw tensor and dict input
            if isinstance(latents, dict):
                samples = latents
            else:
                samples = {"samples": latents}
            
            result = decoder.decode(
                samples=samples,
                vae=self.vae,
                flux=getattr(self.vae, "flux", False),
            )
            
            return result[0]
            
        except Exception as e:
            logging.getLogger(__name__).exception(f"Decoding failed: {e}")
            raise
