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
import os
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
        
        # 1. Load base model
        model = self._load_model(ctx)
        
        # 2. Apply optimizations to base model
        self._apply_optimizations(ctx, model)
        
        # 3. Encode prompts for base model
        positive, negative = self._encode_prompts(ctx, model)
        ctx.positive_cond = positive
        ctx.negative_cond = negative
        
        # 4. Handle refiner preparation if enabled
        refiner_model = None
        ref_positive, ref_negative = None, None
        use_refiner = bool(ctx.generation.refiner_model_path and ctx.generation.refiner_switch_step is not None)
        
        if use_refiner:
            print(f"Refiner enabled: {os.path.basename(ctx.generation.refiner_model_path)} (Switch at step {ctx.generation.refiner_switch_step})")
            # We don't load it yet to save VRAM, but we need to know if we should unload base later
        
        # 5. Generate for each seed
        from src.FileManaging import ImageSaver
        saver = ImageSaver.SaveImage()
        
        for i, seed in enumerate(ctx.seeds[:ctx.generation.number]):
            self._check_interrupt()
            ctx.seed = seed
            
            # Stage 1: Base model generation
            if use_refiner:
                steps_for_base = ctx.generation.refiner_switch_step
                print(f"Stage 1: Running Base model ({steps_for_base}/{ctx.sampling.steps} steps)...")
                latents = model.generate(
                    ctx, positive, negative, 
                    last_step=ctx.generation.refiner_switch_step,
                    callback=ctx.callback
                )
            else:
                latents = model.generate(ctx, positive, negative, callback=ctx.callback)
            
            ctx.current_latents = latents["samples"]
            
            # Stage 2: Refiner model generation
            if use_refiner:
                self._check_interrupt()
                
                # Load refiner model (this will unload base model if necessary)
                refiner_model = self._load_refiner_model(ctx)
                self._apply_optimizations(ctx, refiner_model)
                
                # Encode prompts for refiner (it has different CLIP)
                ref_positive, ref_negative = self._encode_prompts(ctx, refiner_model)
                
                # Disable multi-scale for refiner pass (always)
                orig_ms = ctx.sampling.enable_multiscale
                ctx.sampling.enable_multiscale = False
                
                steps_for_refiner = ctx.sampling.steps - ctx.generation.refiner_switch_step
                print(f"Stage 2: Running Refiner model ({steps_for_refiner}/{ctx.sampling.steps} steps)...")
                latents = refiner_model.generate(
                    ctx, ref_positive, ref_negative,
                    latent_image=latents,
                    start_step=ctx.generation.refiner_switch_step,
                    disable_noise=True,
                    callback=ctx.callback
                )
                ctx.current_latents = latents["samples"]
                ctx.sampling.enable_multiscale = orig_ms
                
                # If we have more seeds, we'll need to reload base model in the next iteration
                # _load_model handles this automatically
            
            # 6. Post-processing
            
            # Apply HiresFix if enabled (uses the model currently loaded, which might be refiner!)
            # Note: Usually HiresFix is done with the base model for better consistency,
            # but some people like refining the hires pass too.
            current_model = refiner_model if use_refiner else model
            
            if HiresFix.is_enabled(ctx):
                self._check_interrupt()
                # HiresFix might need base model prompts if it was trained on them
                hf_pos = ref_positive if use_refiner and ref_positive else positive
                hf_neg = ref_negative if use_refiner and ref_negative else negative
                
                latents = HiresFix.apply(latents, ctx, current_model, hf_pos, hf_neg)
                ctx.current_latents = latents["samples"]
            
            self._check_interrupt()
            
            # Decode to image (uses VAE from current model)
            image = current_model.decode(ctx.current_latents)
            ctx.current_image = image
            
            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                self._check_interrupt()
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Apply Adetailer if enabled (handles its own saving)
            if Adetailer.is_enabled(ctx):
                self._check_interrupt()
                ctx.current_image, _ = Adetailer.apply(ctx.current_image, ctx, current_model, hf_neg)
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
        
        Supports two modes:
        1. Upscale mode: When target dimensions are larger than input (uses USDU)
        2. Diffusion mode: True img2img with denoising strength (uses simple_img2img)
        
        Args:
            ctx: Context with img2img_image set
            
        Returns:
            Context with generated images
        """
        from src.Processors import Img2Img
        from src.FileManaging import ImageSaver
        from PIL import Image
        import numpy as np
        import torch
        
        self._check_interrupt()
        
        model = self._load_model(ctx)
        self._apply_optimizations(ctx, model)
        
        positive, negative = self._encode_prompts(ctx, model)
        saver = ImageSaver.SaveImage()
        
        # Load input image to determine mode
        img_path = ctx.features.img2img_image
        if not img_path:
            raise ValueError("No input image provided for img2img")
        
        img = Image.open(img_path)
        input_w, input_h = img.size
        target_w, target_h = ctx.generation.width, ctx.generation.height
        
        # Convert image to tensor [B, H, W, C]
        img_array = np.array(img.convert("RGB"))
        img_tensor = torch.from_numpy(img_array).float().cpu() / 255.0
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Determine mode: upscale if target is larger, otherwise diffusion
        use_upscale = (target_w > input_w * 1.1) or (target_h > input_h * 1.1)
        denoise = ctx.features.img2img_denoise
        
        # Inject SDXL size conditioning if required
        if getattr(model.capabilities, 'requires_size_conditioning', False):
            for cond_list in [positive, negative]:
                for cond_item in cond_list:
                    if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                        cond_item[1].update({
                            "width": target_w,
                            "height": target_h,
                            "crop_w": 0,
                            "crop_h": 0,
                            "target_width": target_w,
                            "target_height": target_h,
                        })
        
        logger.info(f"Img2Img: input={input_w}x{input_h}, target={target_w}x{target_h}, denoise={denoise:.2f}, mode={'upscale' if use_upscale else 'diffusion'}")
        
        for seed in ctx.seeds[:ctx.generation.number]:
            self._check_interrupt()
            ctx.seed = seed
            
            if use_upscale:
                # Use USDU upscaler (existing behavior)
                # Higher LoRA strength for img2img upscaling
                if self.default_lora and getattr(model.capabilities, 'supports_lora', True):
                    try:
                        model.apply_lora(self.default_lora[0], 2.0, 2.0)
                    except Exception as e:
                        logger.warning(f"LoRA failed: {e}")
                
                result = Img2Img.apply(ctx, model, positive, negative, image_tensor=img_tensor, denoise=denoise)
                ctx.current_image = result
            else:
                # True diffusion-based img2img with denoising strength
                # Resize input image to target dimensions if different
                if input_w != target_w or input_h != target_h:
                    resized_img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    img_array = np.array(resized_img.convert("RGB"))
                    img_tensor = torch.from_numpy(img_array).float().cpu() / 255.0
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                
                # Run simple_img2img for true diffusion-based generation
                latents = Img2Img.simple_img2img(
                    ctx, model, positive, negative,
                    image_tensor=img_tensor,
                    denoise=denoise,
                )
                ctx.current_latents = latents["samples"]
                
                # Decode to image
                image = model.decode(ctx.current_latents)
                ctx.current_image = image
            
            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Save the image with metadata including denoise value
            saver.save_images_async(
                filename_prefix="LD-I2I",
                images=ctx.current_image,
                prompt=str(ctx.prompt),
                extra_pnginfo=ctx.build_metadata({
                    "img2img": "True",
                    "img2img_denoise": str(denoise),
                    "img2img_mode": "upscale" if use_upscale else "diffusion",
                }),
            )
        
        ctx.save_seed()
        return ctx
    
    def run_controlnet(self, ctx: Context) -> Context:
        """Run ControlNet-style generation using Canny edges + img2img.
        
        This uses edge detection to preserve structure while allowing
        color and content changes via high-denoise img2img.
        
        Args:
            ctx: Context with controlnet_model, img2img_image set
            
        Returns:
            Context with generated images
        """
        from src.Processors import ControlNet as CNProcessor
        from src.FileManaging import ImageSaver
        from PIL import Image
        import numpy as np
        
        self._check_interrupt()
        
        # Validate inputs
        if not ctx.features.img2img_image:
            raise ValueError("No input image provided for ControlNet")
        
        model = self._load_model(ctx)
        self._apply_optimizations(ctx, model)
        
        # Load and preprocess input image
        img_path = ctx.features.img2img_image
        img = Image.open(img_path)
        img = img.resize((ctx.generation.width, ctx.generation.height), Image.Resampling.LANCZOS)
        
        # Convert to tensor [B, H, W, C]
        img_array = np.array(img.convert("RGB"))
        img_tensor = torch.from_numpy(img_array).float().cpu() / 255.0
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Apply preprocessor (Canny edge detection by default)
        control_image = CNProcessor.ControlNetProcessor.preprocess_image(
            img_tensor,
            preprocessor=ctx.features.controlnet_type,
        )
        
        strength = ctx.features.controlnet_strength
        logger.info(f"ControlNet-style: {ctx.features.controlnet_type} edges, strength={strength}")
        
        # Encode prompts
        positive, negative = self._encode_prompts(ctx, model)
        
        saver = ImageSaver.SaveImage()
        
        is_flux2 = getattr(model.capabilities, "is_flux2", False)
        
        for seed in ctx.seeds[:ctx.generation.number]:
            self._check_interrupt()
            ctx.seed = seed
            
            # Use the Canny+img2img approach, passing original image for blending
            latents, ctx = CNProcessor.apply_controlnet_to_img2img(
                ctx, model, positive, negative,
                control_image=control_image,
                strength=strength,
                original_image=img_tensor,
            )
            ctx.current_latents = latents["samples"]
            
            # Decode to image
            image = model.decode(ctx.current_latents)
            ctx.current_image = image
            
            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Save with metadata
            saver.save_images_async(
                filename_prefix="LD-CN",
                images=ctx.current_image,
                prompt=str(ctx.prompt),
                extra_pnginfo=ctx.build_metadata({
                    "controlnet_style": "True",
                    "controlnet_strength": str(strength),
                    "controlnet_type": ctx.features.controlnet_type,
                }),
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
        
        Optimized to reuse existing loaded model if it matches the request.
        """
        path = ctx.model_path
        
        # 1. Determine target model type for reuse check
        from src.Core.Models.ModelFactory import detect_model_type
        target_type = "Flux2Klein" if path == "__FLUX2_KLEIN__" else detect_model_type(path)
        
        # 2. Check if current model can be reused
        if self._model is not None and self._model.is_loaded:
            current_type = self._model.__class__.__name__.replace("Model", "")
            
            # Match if paths are identical OR if both are Flux2 (auto-detected/marker)
            paths_match = (self._model.model_path == path)
            types_match = (current_type == target_type)
            
            if paths_match or (not path and types_match) or (path == "__FLUX2_KLEIN__" and target_type == "Flux2Klein" and types_match):
                logger.info(f"Reusing currently loaded {current_type} model")
                return self._model
            
            # 3. Different model requested: UNLOAD OLD ONE FIRST to free VRAM
            logger.info(f"Unloading {current_type} model to load {target_type}")
            self._model.unload()
            self._model = None
            
            # Also clear the global model cache used by CheckpointLoader
            try:
                from src.Device.ModelCache import clear_model_cache
                clear_model_cache()
            except Exception:
                pass
            
            # Force cleanup to prevent memory pressure/stuttering during transition
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        # 4. Create and load new model instance
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
    
    def _load_refiner_model(self, ctx: Context) -> AbstractModel:
        """Load the refiner model for this context.
        
        Optimized to reuse existing loaded model if it matches the refiner path.
        """
        path = ctx.generation.refiner_model_path
        if not path:
            raise ValueError("refiner_model_path is required for refiner pass")
            
        # 1. Determine target model type
        from src.Core.Models.ModelFactory import detect_model_type
        target_type = detect_model_type(path)
        
        # 2. Check if current model can be reused
        if self._model is not None and self._model.is_loaded:
            if self._model.model_path == path:
                logger.info(f"Reusing currently loaded model as refiner")
                return self._model
            
            # 3. Different model requested: UNLOAD OLD ONE FIRST to free VRAM
            logger.info(f"Unloading current model to load refiner {target_type}")
            self._model.unload()
            # self._model = None # Don't set to None yet, we'll replace it
            
            # Also clear the global model cache
            try:
                from src.Device.ModelCache import clear_model_cache
                clear_model_cache()
            except Exception:
                pass
            
            # Force cleanup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        # 4. Create and load new model instance
        model = self.model_factory(model_path=path)
        model.load()
        self._model = model
        return model

    def _apply_optimizations(self, ctx: Context, model: AbstractModel) -> None:
        """Apply all configured optimizations to the model."""
        # LoRA - only if model supports it and matches default LoRA type
        # Default LoRA (add_detail) is SD1.5 (context_dim 768)
        is_sd15 = False
        try:
            is_sd15 = model.get_model_object("context_dim") == 768
        except Exception:
            pass

        if self.default_lora and getattr(model.capabilities, 'supports_lora', True):
            # Only apply default detailing LoRA to SD1.5 models
            if not is_sd15 and self.default_lora[0] == "add_detail.safetensors":
                logger.debug(f"Skipping default SD1.5 LoRA for non-SD1.5 model")
            else:
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
