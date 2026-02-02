"""Core Pipeline orchestrator for LightDiffusion-Next.

This module provides the main Pipeline class - a clean, linear orchestrator
that coordinates model loading, generation, and post-processing.

The Pipeline is designed to be:
- Simple: <100 lines of core logic
- Modular: Delegates to Models and Processors
- Extensible: Easy to add new processing steps

Architecture:
    [Context] -> [Load Model] -> [Encode] -> [Generate] -> [Decode] -> [Processors] -> [Result]
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch

from src.Core.Context import Context
from src.Core.Models import create_model
from src.Core.AbstractModel import AbstractModel
from src.Processors import HiresFix, Adetailer, AutoHDRProcessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    images: list[torch.Tensor] = field(default_factory=list)
    latents: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for legacy compatibility."""
        return {
            "images": self.images,
            "latents": self.latents,
            **self.metadata,
        }


class Pipeline:
    """Main generation pipeline orchestrator.
    
    This class coordinates the entire generation flow in a clean,
    linear manner. Each step is isolated and the Context flows through.
    
    Usage:
        ctx = Context(prompt="a cat", width=512, height=512)
        pipeline = Pipeline()
        result = pipeline.run(ctx)
    """
    
    def __init__(
        self,
        model_factory: Callable[[str], AbstractModel] = None,
        default_lora: Optional[tuple[str, float, float]] = ("add_detail.safetensors", 0.7, 0.7),
    ):
        """Initialize the pipeline.
        
        Args:
            model_factory: Function to create models (default: create_model)
            default_lora: Default LoRA to apply (name, model_str, clip_str) or None
        """
        self.model_factory = model_factory or create_model
        self.default_lora = default_lora
        self._model: Optional[AbstractModel] = None
    
    def run(self, ctx: Context) -> Context:
        """Run the full generation pipeline.
        
        Args:
            ctx: Configured Context with all parameters
            
        Returns:
            Context with generated images in current_image
        """
        self._check_interrupt()
        
        # 1. Load model
        model = self._load_model(ctx)
        
        # 2. Apply optimizations
        self._apply_optimizations(ctx, model)
        
        # 3. Encode prompts
        positive, negative = self._encode_prompts(ctx, model)
        ctx.positive_cond = positive
        ctx.negative_cond = negative
        
        # 4. Generate for each seed
        from src.FileManaging import ImageSaver
        saver = ImageSaver.SaveImage()
        
        for i, seed in enumerate(ctx.seeds[:ctx.generation.number]):
            self._check_interrupt()
            ctx.seed = seed
            
            # Generate latents
            latents = model.generate(ctx, positive, negative)
            ctx.current_latents = latents["samples"]
            
            # Apply HiresFix if enabled
            if HiresFix.is_enabled(ctx):
                latents = HiresFix.apply(latents, ctx, model, positive, negative)
                ctx.current_latents = latents["samples"]
            
            self._check_interrupt()
            
            # Decode to image
            image = model.decode(ctx.current_latents)
            ctx.current_image = image
            
            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Apply Adetailer if enabled (handles its own saving)
            if Adetailer.is_enabled(ctx):
                ctx.current_image, _ = Adetailer.apply(ctx.current_image, ctx, model, negative)
            else:
                # Save the image
                prefix = "LD-HF" if ctx.features.hires_fix else "LD"
                saver.save_images_async(
                    filename_prefix=prefix,
                    images=ctx.current_image,
                    prompt=str(ctx.prompt),
                    extra_pnginfo=ctx.build_metadata(),
                )
        
        ctx.save_seed()
        return ctx
    
    def run_img2img(self, ctx: Context) -> Context:
        """Run image-to-image generation pipeline.
        
        Args:
            ctx: Context with img2img_image set
            
        Returns:
            Context with generated images
        """
        from src.Processors import Img2Img
        from src.FileManaging import ImageSaver
        
        self._check_interrupt()
        
        model = self._load_model(ctx)
        self._apply_optimizations(ctx, model)
        
        # Higher LoRA strength for img2img
        if self.default_lora:
            try:
                model.apply_lora(self.default_lora[0], 2.0, 2.0)
            except Exception as e:
                logger.warning(f"LoRA failed: {e}")
        
        positive, negative = self._encode_prompts(ctx, model)
        saver = ImageSaver.SaveImage()
        
        for seed in ctx.seeds[:ctx.generation.number]:
            self._check_interrupt()
            ctx.seed = seed
            
            result = Img2Img.apply(ctx, model, positive, negative)
            ctx.current_image = result
            
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            saver.save_images(
                filename_prefix="LD-I2I",
                images=ctx.current_image,
                prompt=str(ctx.prompt),
                extra_pnginfo=ctx.build_metadata({"img2img": "True"}),
            )
        
        ctx.save_seed()
        return ctx
    
    def run_batched(self, ctx: Context, per_sample_info: list = None) -> dict:
        """Run batched multi-prompt generation.
        
        Args:
            ctx: Context with list of prompts
            per_sample_info: Per-sample overrides
            
        Returns:
            Dictionary mapping request_ids to results
        """
        import uuid
        from src.FileManaging import ImageSaver
        from src.Utilities import Latent
        from src.sample import sampling
        from src.hidiffusion import msw_msa_attention
        
        self._check_interrupt()
        
        prompts = list(ctx.prompt)
        total_batch = len(prompts)
        per_sample_info = per_sample_info or [{} for _ in range(total_batch)]
        
        # Setup negatives
        if isinstance(ctx.negative_prompt, (list, tuple)):
            negatives = list(ctx.negative_prompt)
        else:
            negatives = [ctx.negative_prompt] * total_batch
        
        model = self._load_model(ctx)
        self._apply_optimizations(ctx, model)
        
        # Encode all prompts
        positive, negative = model.encode_prompt(prompts, negatives)
        
        # Add batch routing
        if isinstance(positive, list):
            for i, entry in enumerate(positive):
                if len(entry) > 1 and isinstance(entry[1], dict):
                    entry[1]["batch_index"] = [i]
        
        # Generate all latents
        latent_gen = Latent.EmptyLatentImage()
        latent = latent_gen.generate(ctx.width, ctx.height, total_batch)[0]
        latent["seeds"] = ctx.seeds[:total_batch]
        
        try:
            hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
            opt_model = hidiff.go(model_type="auto", model=model.model)[0]
        except Exception:
            opt_model = model.model
        
        ksampler = sampling.KSampler()
        batch_latents = ksampler.sample(
            seed=None,
            steps=ctx.sampling.steps,
            cfg=ctx.sampling.cfg,
            sampler_name=ctx.sampling.sampler,
            scheduler=ctx.sampling.scheduler,
            denoise=1.0,
            pipeline=True,
            model=opt_model,
            positive=positive,
            negative=negative,
            latent_image=latent,
            enable_multiscale=ctx.sampling.enable_multiscale,
            multiscale_factor=ctx.sampling.multiscale_factor,
            multiscale_fullres_start=ctx.sampling.multiscale_fullres_start,
            multiscale_fullres_end=ctx.sampling.multiscale_fullres_end,
            cfg_free_enabled=ctx.sampling.cfg_free_enabled,
            cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
        )
        
        # Decode all
        images = model.decode(batch_latents[0]["samples"])
        
        if AutoHDRProcessor.is_enabled(ctx):
            images = AutoHDRProcessor.apply(images, ctx)
        
        # Process individually
        saver = ImageSaver.SaveImage()
        results = {}
        
        for i in range(total_batch):
            self._check_interrupt()
            
            info = per_sample_info[i] if i < len(per_sample_info) else {}
            req_id = info.get("request_id", uuid.uuid4().hex[:8])
            prefix = info.get("filename_prefix", f"LD-REQ-{req_id}")
            
            final = images[i]
            
            # Per-sample HiresFix
            if info.get("hires_fix", False):
                try:
                    single_latent = {"samples": batch_latents[0]["samples"][i:i+1]}
                    single_ctx = ctx.clone()
                    single_ctx.seed = ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed
                    
                    hires = HiresFix.apply(
                        single_latent, single_ctx, model,
                        [positive[i]] if isinstance(positive, list) else positive,
                        [negative[i]] if isinstance(negative, list) else negative,
                    )
                    final = model.decode(hires["samples"])[0]
                    if AutoHDRProcessor.is_enabled(ctx):
                        final = AutoHDRProcessor.apply(final, ctx)
                except Exception as e:
                    logger.warning(f"Batch hires_fix failed: {e}")
            
            # Per-sample Adetailer
            if info.get("adetailer", False):
                try:
                    single_ctx = ctx.clone()
                    single_ctx.seed = ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed
                    final, saved = Adetailer.apply(final, single_ctx, model, negative)
                    results.setdefault(req_id, []).extend(
                        s.get("ui", {}).get("images", [s]) for s in saved
                    )
                except Exception as e:
                    logger.warning(f"Batch adetailer failed: {e}")
            
            # Save
            meta = ctx.build_metadata({
                "seed": str(ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed),
                "prompt": prompts[i],
            })
            saved = saver.save_images(prefix, [final], prompts[i], meta)
            results.setdefault(req_id, []).extend(
                saved.get("ui", {}).get("images", [saved])
            )
        
        return {"batched_results": results}
    
    def _load_model(self, ctx: Context) -> AbstractModel:
        """Load the model for this context.
        
        Uses ModelFactory for auto-detection when model_path is empty or
        set to the special __FLUX2_KLEIN__ marker.
        """
        path = ctx.model_path
        
        # Handle special Flux2 Klein marker or empty path
        if path == "__FLUX2_KLEIN__":
            # Explicitly request Flux2 Klein
            model = self.model_factory(model_path=None, model_type="Flux2Klein")
        elif not path:
            # Auto-detect model type (may detect Flux2 components)
            model = self.model_factory(model_path=None)
        else:
            # Specific checkpoint path provided
            model = self.model_factory(model_path=path)
        
        model.load()
        self._model = model
        return model
    
    def _apply_optimizations(self, ctx: Context, model: AbstractModel) -> None:
        """Apply all configured optimizations to the model."""
        # LoRA - only if model supports it
        if self.default_lora and getattr(model.capabilities, 'supports_lora', True):
            try:
                model.apply_lora(*self.default_lora)
            except Exception as e:
                logger.warning(f"LoRA failed: {e}")
        
        # StableFast
        if ctx.generation.stable_fast:
            model.apply_stable_fast(enable_cuda_graph=True)
        
        # DeepCache
        if ctx.sampling.deepcache_enabled:
            model.apply_deepcache(
                ctx.sampling.deepcache_interval,
                ctx.sampling.deepcache_depth,
                ctx.sampling.deepcache_start_step,
                ctx.sampling.deepcache_end_step,
            )
    
    def _encode_prompts(self, ctx: Context, model: AbstractModel) -> tuple[Any, Any]:
        """Encode prompts to conditioning tensors."""
        return model.encode_prompt(ctx.prompt, ctx.negative_prompt)
    
    def _check_interrupt(self) -> None:
        """Check for user interrupt."""
        from src.user import app_instance
        app = getattr(app_instance, "app", None)
        if app and getattr(app, "interrupt_flag", False):
            raise InterruptedError("Generation interrupted")


# Singleton default pipeline
_default_pipeline: Optional[Pipeline] = None


def get_default_pipeline() -> Pipeline:
    """Get the default pipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline
