"""SD1.5 model adapter for LightDiffusion-Next.

Provides a clean interface to the SD1.5 model that inherits from
AbstractModel and wraps the existing infrastructure.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from src.Core.AbstractModel import AbstractModel, ModelCapabilities

if TYPE_CHECKING:
    from src.Core.PipelineContext import PipelineContext


class SD15Model(AbstractModel):
    """SD1.5 model implementation.
    
    Wraps the existing SD15 model loading and inference code
    with the clean AbstractModel interface.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the SD15 model adapter.
        
        Args:
            model_path: Path to the model checkpoint (safetensors/pt)
        """
        super().__init__(model_path)
        self._clip_skip = -2
    
    def _create_capabilities(self) -> ModelCapabilities:
        """Create capabilities for SD1.5 models."""
        return ModelCapabilities(
            min_resolution=256,
            max_resolution=2048,
            preferred_resolution=512,
            requires_resolution_multiple=64,
            supports_hires_fix=True,
            supports_img2img=True,
            supports_inpainting=True,
            supports_controlnet=True,
            supports_stable_fast=True,
            supports_deepcache=True,
            supports_tome=True,
            uses_dual_clip=False,
            requires_size_conditioning=False,
        )
    
    def load(self, model_path: str = None) -> "SD15Model":
        """Load the SD1.5 model from disk.
        
        Args:
            model_path: Optional override for the model path
            
        Returns:
            Self for method chaining
        """
        logger = logging.getLogger(__name__)
        
        path = model_path or self.model_path
        if path is None:
            # Use default checkpoint
            path = "./include/checkpoints/DreamShaper_8_pruned.safetensors"
        
        # Guard: Don't reload if already loaded with same path
        if self._loaded and self.model_path == path:
            logger.info(f"SD15Model: Already loaded {path}, skipping redundant load")
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
            
            logger.info(f"SD15Model: loaded {path}")
            
        except Exception as e:
            logger.exception(f"SD15Model: failed to load {path}: {e}")
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
        ctx: "PipelineContext",
        positive: Any,
        negative: Any,
    ) -> dict:
        """Generate latents using the sampler.
        
        Args:
            ctx: Pipeline context with generation parameters
            positive: Positive conditioning
            negative: Negative conditioning
            
        Returns:
            Dictionary with 'samples' key containing generated latents
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before generating")
        
        try:
            from src.sample import sampling
            from src.Utilities import Latent
            from src.hidiffusion import msw_msa_attention
            
            # Create empty latent
            latent_gen = Latent.EmptyLatentImage()
            latent = latent_gen.generate(
                width=ctx.generation.width,
                height=ctx.generation.height,
                batch_size=ctx.generation.batch,
            )[0]
            
            # Add seeds for deterministic noise
            latent["seeds"] = ctx.seeds[:ctx.generation.batch] if ctx.seeds else [ctx.seed]
            
            # Apply HiDiffusion optimization only for high resolutions
            if ctx.generation.width > 1024 or ctx.generation.height > 1024:
                try:
                    hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                    optimized_model = hidiff.go(model_type="sd15", model=self.model)[0]
                except Exception:
                    optimized_model = self.model
            else:
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
            )
            
            return result[0]
            
        except Exception as e:
            logging.getLogger(__name__).exception(f"Decoding failed: {e}")
            raise
