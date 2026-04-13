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

    def _apply_runtime_preferences(self, ctx: Context, model: AbstractModel) -> None:
        """Apply request-scoped runtime preferences that should track reused models."""
        model.set_vae_autotune(ctx.generation.vae_autotune)
    
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
        self._apply_runtime_preferences(ctx, model)
        
        # 2. Apply optimizations to base model
        mo = getattr(model, 'model', None)
        mo_opts = getattr(mo, 'model_options', {}) if mo is not None else {}
        if not mo_opts.get("model_function_wrapper"):
            self._apply_optimizations(ctx, model)
        
        # 3. Encode prompts for base model
        positive, negative = self._encode_prompts(ctx, model)
        ctx.positive_cond = positive
        ctx.negative_cond = negative
        
        # 4. Handle refiner preparation if enabled (SDXL only)
        refiner_model = None
        ref_positive, ref_negative = None, None
        
        is_sdxl = getattr(model.capabilities, "uses_dual_clip", False)
        use_refiner = bool(
            is_sdxl and 
            ctx.generation.refiner_model_path and 
            ctx.generation.refiner_switch_step is not None and
            0 < ctx.generation.refiner_switch_step < ctx.sampling.steps
        )
        
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
            
            # Decode latents to image
            ctx.current_image = model.decode(ctx.current_latents)
            
            # 6. Post-processing
            
# Apply HiresFix if enabled. Prefer running hires pass with the base model
            # and base prompts for consistency; using a refiner for the hires pass can
            # introduce artifacts because its UNet/CLIP can differ from the base model.
            current_model = model
            # Prefer base prompts for hires pass (refiner prompts tend to mismatch)
            hf_pos = positive
            hf_neg = negative

            if HiresFix.is_enabled(ctx):
                self._check_interrupt()
                logger.info(f"HiresFix: using base model for hires pass (use_refiner={use_refiner})")
                # If a refiner was used earlier we may have unloaded the base model to free VRAM.
                # Ensure the base model is reloaded and optimized before running the hires pass so
                # downstream code (sampler / CFGGuider) can access model.model_options etc.
                if use_refiner and (not model.is_loaded or getattr(model, "model", None) is None):
                    logger.info("HiresFix: reloading base model for hires pass (was unloaded by refiner)")
                    model = self._load_model(ctx)
                    # Re-apply optimizations (LoRA / StableFast / FP8 / DeepCache) to the reloaded model
                    self._apply_optimizations(ctx, model)
                    # Re-encode prompts for the reloaded base model to ensure conditioning matches
                    try:
                        hf_pos, hf_neg = self._encode_prompts(ctx, model)
                    except Exception:
                        # Fallback to previously-encoded conditioning if re-encoding fails
                        hf_pos, hf_neg = hf_pos, hf_neg
                    current_model = model
                # HiresFix might still need base model prompts if it was trained on them
                latents = HiresFix.apply(latents, ctx, current_model, hf_pos, hf_neg, callback=ctx.callback)
                ctx.current_latents = latents["samples"]
            if AutoHDRProcessor.is_enabled(ctx):
                self._check_interrupt()
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Apply Adetailer if enabled (handles its own saving)
            if Adetailer.is_enabled(ctx):
                self._check_interrupt()
                if use_refiner:
                    # Reload base model for ADetailer - the refiner's UNet/CLIP
                    # is not suited for text-guided crop enhancement
                    ad_model = self._load_model(ctx)
                    ad_pos, ad_neg = self._encode_prompts(ctx, ad_model)
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, ad_model,
                        positive=ad_pos, negative=ad_neg,
                        callback=ctx.callback
                    )
                else:
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, current_model,
                        positive=hf_pos, negative=hf_neg,
                        callback=ctx.callback
                    )
            else:
                # Save the image synchronously so the server can reliably find it
                prefix = "LD-HF" if ctx.features.hires_fix else "LD"
                filename_prefix = f"{ctx.features.request_filename_prefix}_{prefix}" if ctx.features.request_filename_prefix else prefix
                images = ctx.current_image if isinstance(ctx.current_image, list) else [ctx.current_image]
                saver.save_images(images, filename_prefix=filename_prefix, prompt=str(ctx.prompt), extra_pnginfo=ctx.build_metadata(), store_bytes_prefix=ctx.features.request_filename_prefix)
        
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
                
                result = Img2Img.apply(ctx, model, positive, negative, image_tensor=img_tensor, denoise=denoise, callback=ctx.callback)
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
                
                # Check if refiner is enabled BEFORE running base model (SDXL only)
                is_sdxl = getattr(model.capabilities, "uses_dual_clip", False)
                use_refiner = bool(
                    is_sdxl and 
                    ctx.generation.refiner_model_path and 
                    ctx.generation.refiner_switch_step is not None and
                    0 < ctx.generation.refiner_switch_step < ctx.sampling.steps
                )
                refiner_model = None
                ref_negative = None
                base_last_step = ctx.generation.refiner_switch_step if use_refiner else None
                
                if use_refiner:
                    print(f"Stage 1: Running Base model ({ctx.generation.refiner_switch_step}/{ctx.sampling.steps} steps)...")
                
                # Run simple_img2img for true diffusion-based generation
                latents = Img2Img.simple_img2img(
                    ctx, model, positive, negative,
                    image_tensor=img_tensor,
                    denoise=denoise,
                    last_step=base_last_step,
                    callback=ctx.callback,
                )
                ctx.current_latents = latents["samples"]
                
                # Apply refiner if enabled
                if use_refiner:
                    self._check_interrupt()
                    
                    # Load refiner model
                    refiner_model = self._load_refiner_model(ctx)
                    self._apply_optimizations(ctx, refiner_model)
                    
                    # Encode prompts for refiner (it has different CLIP)
                    ref_positive, ref_negative = self._encode_prompts(ctx, refiner_model)
                    
                    # Disable multi-scale for refiner pass
                    orig_ms = ctx.sampling.enable_multiscale
                    ctx.sampling.enable_multiscale = False
                    
                    steps_for_refiner = ctx.sampling.steps - ctx.generation.refiner_switch_step
                    print(f"Img2Img Refiner: Running {steps_for_refiner}/{ctx.sampling.steps} steps...")
                    refiner_latents = refiner_model.generate(
                        ctx, ref_positive, ref_negative,
                        latent_image=latents,
                        start_step=ctx.generation.refiner_switch_step,
                        disable_noise=True,
                        callback=ctx.callback
                    )
                    ctx.current_latents = refiner_latents["samples"]
                    ctx.sampling.enable_multiscale = orig_ms
                    
                    # Decode using refiner's VAE
                    image = refiner_model.decode(ctx.current_latents)
                else:
                    # Decode to image using base model
                    image = model.decode(ctx.current_latents)
                
                ctx.current_image = image
            
            # Apply Adetailer if enabled
            from src.Processors import Adetailer
            if Adetailer.is_enabled(ctx):
                self._check_interrupt()
                if not use_upscale and use_refiner:
                    # Reload base model for ADetailer - the refiner's UNet/CLIP
                    # is not suited for text-guided crop enhancement
                    ad_model = self._load_model(ctx)
                    ad_pos, ad_neg = self._encode_prompts(ctx, ad_model)
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, ad_model,
                        positive=ad_pos, negative=ad_neg,
                        callback=ctx.callback
                    )
                else:
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, model,
                        positive=positive, negative=negative,
                        callback=ctx.callback
                    )

            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Save the image with metadata including denoise value
            filename_prefix = "LD-I2I"
            if ctx.features.request_filename_prefix:
                filename_prefix = f"{ctx.features.request_filename_prefix}_{filename_prefix}"
            images = ctx.current_image if isinstance(ctx.current_image, list) else [ctx.current_image]
            saver.save_images(images, filename_prefix=filename_prefix, prompt=str(ctx.prompt), extra_pnginfo=ctx.build_metadata({
                "img2img": "True",
                "img2img_denoise": str(denoise),
                "img2img_mode": "upscale" if use_upscale else "diffusion",
            }), store_bytes_prefix=ctx.features.request_filename_prefix)
        
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
        
        # Check if refiner is enabled (SDXL only)
        is_sdxl = getattr(model.capabilities, "uses_dual_clip", False)
        use_refiner = bool(
            is_sdxl and 
            ctx.generation.refiner_model_path and 
            ctx.generation.refiner_switch_step is not None and
            0 < ctx.generation.refiner_switch_step < ctx.sampling.steps
        )
        refiner_model = None
        ref_negative = None
        
        if use_refiner:
            print(f"Refiner enabled for ControlNet: {os.path.basename(ctx.generation.refiner_model_path)} (Switch at step {ctx.generation.refiner_switch_step})")
        
        for seed in ctx.seeds[:ctx.generation.number]:
            self._check_interrupt()
            ctx.seed = seed
            
            # Use the Canny+img2img approach, passing original image for blending
            # When refiner is enabled, stop base model at refiner switch step
            base_last_step = ctx.generation.refiner_switch_step if use_refiner else None
            if use_refiner:
                print(f"Stage 1: Running Base model ({ctx.generation.refiner_switch_step}/{ctx.sampling.steps} steps)...")
            
            latents, ctx = CNProcessor.apply_controlnet_to_img2img(
                ctx, model, positive, negative,
                control_image=control_image,
                strength=strength,
                original_image=img_tensor,
                last_step=base_last_step,
                callback=ctx.callback,
            )
            ctx.current_latents = latents["samples"]
            
            # Apply refiner if enabled
            if use_refiner:
                self._check_interrupt()
                
                # Load refiner model
                refiner_model = self._load_refiner_model(ctx)
                self._apply_optimizations(ctx, refiner_model)
                
                # Encode prompts for refiner (it has different CLIP)
                ref_positive, ref_negative = self._encode_prompts(ctx, refiner_model)
                
                # Disable multi-scale for refiner pass
                orig_ms = ctx.sampling.enable_multiscale
                ctx.sampling.enable_multiscale = False
                
                steps_for_refiner = ctx.sampling.steps - ctx.generation.refiner_switch_step
                print(f"ControlNet Refiner: Running {steps_for_refiner}/{ctx.sampling.steps} steps...")
                refiner_latents = refiner_model.generate(
                    ctx, ref_positive, ref_negative,
                    latent_image=latents,
                    start_step=ctx.generation.refiner_switch_step,
                    disable_noise=True,
                    callback=ctx.callback
                )
                ctx.current_latents = refiner_latents["samples"]
                ctx.sampling.enable_multiscale = orig_ms
                
                # Decode using refiner's VAE
                image = refiner_model.decode(ctx.current_latents)
            else:
                # Decode to image using base model
                image = model.decode(ctx.current_latents)
            
            ctx.current_image = image
            
            # Apply Adetailer if enabled
            from src.Processors import Adetailer
            if Adetailer.is_enabled(ctx):
                self._check_interrupt()
                if use_refiner:
                    # Reload base model for ADetailer - the refiner's UNet/CLIP
                    # is not suited for text-guided crop enhancement
                    ad_model = self._load_model(ctx)
                    ad_pos, ad_neg = self._encode_prompts(ctx, ad_model)
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, ad_model,
                        positive=ad_pos, negative=ad_neg,
                        callback=ctx.callback
                    )
                else:
                    ctx.current_image, _ = Adetailer.apply(
                        ctx.current_image, ctx, model,
                        positive=positive, negative=negative,
                        callback=ctx.callback
                    )

            # Apply AutoHDR if enabled
            if AutoHDRProcessor.is_enabled(ctx):
                ctx.current_image = AutoHDRProcessor.apply(ctx.current_image, ctx)
            
            # Save with metadata
            filename_prefix = "LD-CN"
            if ctx.features.request_filename_prefix:
                filename_prefix = f"{ctx.features.request_filename_prefix}_{filename_prefix}"
            images = ctx.current_image if isinstance(ctx.current_image, list) else [ctx.current_image]
            saver.save_images(images, filename_prefix=filename_prefix, prompt=str(ctx.prompt), extra_pnginfo=ctx.build_metadata({
                "controlnet_style": "True",
                "controlnet_strength": str(strength),
                "controlnet_type": ctx.features.controlnet_type,
            }), store_bytes_prefix=ctx.features.request_filename_prefix)
        
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
        from src.Processors import Img2Img
        
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
        
        # Add batch routing so positive and negative conditioning stay aligned.
        for cond_list in (positive, negative):
            if isinstance(cond_list, list):
                for i, entry in enumerate(cond_list):
                    if len(entry) > 1 and isinstance(entry[1], dict):
                        entry[1]["batch_index"] = [i]
        
        # Determine latent channels (SD1.5/SDXL=4, SD3/Flux1=16, Flux2=32)
        latent_channels = 4
        try:
            lf = model.get_model_object("latent_format")
            if lf and hasattr(lf, "latent_channels"):
                latent_channels = lf.latent_channels
        except Exception:
            pass

        # Architecture flags for sampler
        is_flux = getattr(model.capabilities, "is_flux", False) or (latent_channels == 16)
        is_flux2 = getattr(model.capabilities, "is_flux2", False) or (latent_channels == 32)

        # Generate all latents with correct channel count
        latent_gen = Latent.EmptyLatentImage()
        latent = latent_gen.generate(ctx.width, ctx.height, total_batch, channels=latent_channels)[0]
        latent["seeds"] = ctx.seeds[:total_batch]
        
        # Apply HiDiffusion (multiscale) if enabled
        # CRITICAL: HiDiffusion MSW-MSA is for UNet (SD1.5/SDXL) only. 
        # DiT models like Flux will suffer from tiling artifacts if patched.
        is_flux_or_flux2 = is_flux or is_flux2
        
        if ctx.sampling.enable_multiscale and not is_flux_or_flux2:
            try:
                # Clone model before patching to avoid persistent state across batches
                base_inner = getattr(model, 'model', model)
                patch_model = base_inner.clone() if hasattr(base_inner, 'clone') else base_inner
                hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                opt_model = hidiff.go(model_type="auto", model=patch_model)[0]
                if not hasattr(opt_model, "get_model_object") and hasattr(model, "get_model_object"):
                    opt_model.get_model_object = model.get_model_object
                if not hasattr(opt_model, "load_device") and hasattr(model, "load_device"):
                    opt_model.load_device = model.load_device
            except Exception as e:
                logger.warning(f"Failed to apply HiDiffusion: {e}")
                opt_model = model
        else:
            if ctx.sampling.enable_multiscale and is_flux_or_flux2:
                logger.info("HiDiffusion disabled: not compatible with Flux architecture")
            opt_model = model
        
        # Determine if refiner is enabled (SDXL only)
        is_sdxl = getattr(model.capabilities, "uses_dual_clip", False)
        use_refiner = bool(
            is_sdxl and 
            ctx.generation.refiner_model_path and 
            ctx.generation.refiner_switch_step is not None and
            0 < ctx.generation.refiner_switch_step < ctx.sampling.steps
        )
        
        ksampler = sampling.KSampler()
        
        # Distilled Flux2 Klein safety defaults
        # These models are extremely sensitive to CFG > 1.2 and work best with specific samplers
        if is_flux2:
            if ctx.sampling.cfg > 1.2:
                logger.info(f"Flux2 Klein detected: capping CFG from {ctx.sampling.cfg} to 1.0 for distilled quality")
                ctx.sampling.cfg = 1.0
            if ctx.sampling.sampler not in ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "uni_pc"]:
                logger.info(f"Flux2 Klein detected: switching sampler to 'euler' for compatibility")
                ctx.sampling.sampler = "euler"

        batched_img2img_tensor = None
        batched_img2img_denoise = ctx.features.img2img_denoise
        if ctx.features.img2img and ctx.features.img2img_image:
            from PIL import Image
            import numpy as np

            input_image = Image.open(ctx.features.img2img_image).convert("RGB")
            target_size = (ctx.generation.width, ctx.generation.height)
            if input_image.size != target_size:
                input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)

            input_array = np.array(input_image)
            batched_img2img_tensor = torch.from_numpy(input_array).float().cpu() / 255.0
            batched_img2img_tensor = batched_img2img_tensor.unsqueeze(0).repeat(total_batch, 1, 1, 1)

            if getattr(model.capabilities, "requires_size_conditioning", False):
                for cond_list in (positive, negative):
                    for cond_item in cond_list:
                        if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                            cond_item[1].update({
                                "width": ctx.generation.width,
                                "height": ctx.generation.height,
                                "crop_w": 0,
                                "crop_h": 0,
                                "target_width": ctx.generation.width,
                                "target_height": ctx.generation.height,
                            })
        
        if use_refiner:
            print(f"Batched Refiner enabled: {os.path.basename(ctx.generation.refiner_model_path)} (Switch at step {ctx.generation.refiner_switch_step})")
            
            # Stage 1: Base model generation
            print(f"Stage 1: Running Base model ({ctx.generation.refiner_switch_step}/{ctx.sampling.steps} steps)...")
            if batched_img2img_tensor is not None:
                batch_latents = (
                    Img2Img.simple_img2img(
                        ctx,
                        model,
                        positive,
                        negative,
                        image_tensor=batched_img2img_tensor,
                        denoise=batched_img2img_denoise,
                        last_step=ctx.generation.refiner_switch_step,
                        callback=ctx.callback,
                    ),
                )
            else:
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
                    last_step=ctx.generation.refiner_switch_step,
                    enable_multiscale=ctx.sampling.enable_multiscale,
                    multiscale_factor=ctx.sampling.multiscale_factor,
                    multiscale_fullres_start=ctx.sampling.multiscale_fullres_start,
                    multiscale_fullres_end=ctx.sampling.multiscale_fullres_end,
                    cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                    cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
                    flux=is_flux,
                    flux2=is_flux2,
                    callback=ctx.callback,
                )
            
            self._check_interrupt()
            
            # Stage 2: Refiner model generation
            # Explicitly clear Stage 1 objects to free VRAM for refiner
            import gc
            if 'opt_model' in locals(): del opt_model
            if 'positive' in locals(): del positive
            if 'negative' in locals(): del negative
            
            # CRITICAL: The local variable 'model' still holds the Base model.
            # We must unload it and delete the reference so refcount hits 0.
            if 'model' in locals() and model is not None:
                model.unload()
                del model
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            refiner_model = self._load_refiner_model(ctx)
            # Skip optimizations if already applied (check model_function_wrapper)
            mo = getattr(refiner_model, 'model', None)
            mo_opts = getattr(mo, 'model_options', {}) if mo is not None else {}
            if not mo_opts.get("model_function_wrapper"):
                self._apply_optimizations(ctx, refiner_model)
            
            # Encode prompts for refiner
            ref_positive, ref_negative = refiner_model.encode_prompt(prompts, negatives)
            
            # Re-apply batch routing to refiner conditioning if needed
            if isinstance(ref_positive, list):
                for i, entry in enumerate(ref_positive):
                    if len(entry) > 1 and isinstance(entry[1], dict):
                        entry[1]["batch_index"] = [i]
            
            # Apply resolution conditioning for SDXL refiner if required
            if getattr(refiner_model.capabilities, 'requires_size_conditioning', False):
                for cond_list in [ref_positive, ref_negative]:
                    for cond_item in cond_list:
                        if len(cond_item) > 1 and isinstance(cond_item[1], dict):
                            cond_item[1].update({
                                "width": ctx.generation.width,
                                "height": ctx.generation.height,
                                "crop_w": 0,
                                "crop_h": 0,
                                "target_width": ctx.generation.width,
                                "target_height": ctx.generation.height,
                            })

            # HiDiffusion optimization for refiner: NEVER use multi-scale for refiner pass
            opt_refy = getattr(refiner_model, 'model', refiner_model)

            # Disable multi-scale for refiner pass
            orig_ms = ctx.sampling.enable_multiscale
            ctx.sampling.enable_multiscale = False
            
            steps_for_refiner = ctx.sampling.steps - ctx.generation.refiner_switch_step
            print(f"Stage 2: Running Refiner model ({steps_for_refiner}/{ctx.sampling.steps} steps)...")
            
            batch_latents = ksampler.sample(
                seed=None,
                steps=ctx.sampling.steps,
                cfg=ctx.sampling.cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=ctx.sampling.scheduler,
                denoise=1.0,
                pipeline=True,
                model=opt_refy,
                positive=ref_positive,
                negative=ref_negative,
                latent_image=batch_latents[0],
                start_step=ctx.generation.refiner_switch_step,
                disable_noise=True,
                callback=ctx.callback,
                cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
            )
            ctx.sampling.enable_multiscale = orig_ms
            # Use refiner for decoding
            model = refiner_model 
        else:
            # Normal single-stage generation
            if batched_img2img_tensor is not None:
                batch_latents = (
                    Img2Img.simple_img2img(
                        ctx,
                        model,
                        positive,
                        negative,
                        image_tensor=batched_img2img_tensor,
                        denoise=batched_img2img_denoise,
                        callback=ctx.callback,
                    ),
                )
            else:
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
                    flux=is_flux,
                    flux2=is_flux2,
                    callback=ctx.callback,
                )
        
        # Hires/Adetailer prompts - use refiner prompts if refiner was used
        if use_refiner:
            hf_pos = ref_positive
            hf_neg = ref_negative
        else:
            hf_pos = positive
            hf_neg = negative

        # Decode all
        images = model.decode(batch_latents[0]["samples"])
        
        if AutoHDRProcessor.is_enabled(ctx):
            images = AutoHDRProcessor.apply(images, ctx)
        
        # If refiner was used, reload base model for ADetailer.
        # The refiner's UNet/CLIP is optimized for short refinement passes,
        # not for the text-guided crop enhancement that ADetailer performs.
        ad_model = model
        ad_pos = hf_pos
        ad_neg = hf_neg
        if use_refiner:
            needs_adetailer = any(
                (per_sample_info[j] if j < len(per_sample_info) else {}).get("adetailer", False)
                for j in range(total_batch)
            )
            if needs_adetailer:
                ad_model = self._load_model(ctx)
                self._apply_optimizations(ctx, ad_model)
                ad_pos, ad_neg = ad_model.encode_prompt(prompts, negatives)
                if isinstance(ad_pos, list):
                    for idx, entry in enumerate(ad_pos):
                        if len(entry) > 1 and isinstance(entry[1], dict):
                            entry[1]["batch_index"] = [idx]
        
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

                    # Default to the currently-loaded model (may be refiner)
                    hires_model = model
                    hires_pos = [hf_pos[i]] if isinstance(hf_pos, list) else hf_pos
                    hires_neg = [hf_neg[i]] if isinstance(hf_neg, list) else hf_neg

                    # If a refiner was used, prefer reloading the base model for the hires pass.
                    # Attempt to reload + optimize the base model and re-encode the single-sample
                    # prompts; fall back to existing behavior on any failure.
                    if use_refiner:
                        try:
                            base_model = self._load_model(ctx)
                            self._apply_optimizations(ctx, base_model)

                            # Re-encode only the single sample for the reloaded base model
                            single_pos, single_neg = base_model.encode_prompt([prompts[i]], [negatives[i]])
                            if isinstance(single_pos, list):
                                single_pos = single_pos[0]
                                single_neg = single_neg[0]

                            hires_model = base_model
                            hires_pos = [single_pos] if isinstance(hf_pos, list) else single_pos
                            hires_neg = [single_neg] if isinstance(hf_neg, list) else single_neg
                        except Exception:
                            # If reload/encode fails, continue with the previously-loaded model
                            hires_model = model
                            hires_pos = [hf_pos[i]] if isinstance(hf_pos, list) else hf_pos
                            hires_neg = [hf_neg[i]] if isinstance(hf_neg, list) else hf_neg

                    hires = HiresFix.apply(
                        single_latent, single_ctx, hires_model,
                        hires_pos,
                        hires_neg,
                        callback=ctx.callback,
                    )

                    final = hires_model.decode(hires["samples"])[0]
                    if AutoHDRProcessor.is_enabled(ctx):
                        final = AutoHDRProcessor.apply(final, ctx)
                except Exception as e:
                    logger.warning(f"Batch hires_fix failed: {e}")
            
            # Per-sample Adetailer
            if info.get("adetailer", False):
                try:
                    single_ctx = ctx.clone()
                    single_ctx.seed = ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed
                    final, saved = Adetailer.apply(
                        final, single_ctx, ad_model,
                        positive=[ad_pos[i]] if isinstance(ad_pos, list) else ad_pos,
                        negative=[ad_neg[i]] if isinstance(ad_neg, list) else ad_neg,
                        callback=ctx.callback
                    )
                    for s in saved:
                        results.setdefault(req_id, []).extend(
                            s.get("ui", {}).get("images", [s])
                        )
                except Exception as e:
                    logger.warning(f"Batch adetailer failed: {e}")
            
            # Save
            meta = ctx.build_metadata({
                "seed": str(ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed),
                "prompt": prompts[i],
            })
            saved = saver.save_images([final], prefix, prompts[i], meta, store_bytes_prefix=prefix)
            results.setdefault(req_id, []).extend(
                saved.get("ui", {}).get("images", [saved])
            )
        
        return {"batched_results": results}
    
    def _clear_model_patches(self, model: AbstractModel) -> None:
        """Clear all patches from the model to ensure a clean state."""
        if model and hasattr(model, "model") and model.model:
            # Clear transformer patches (HiDiffusion, etc.)
            if hasattr(model.model, "model_options"):
                to = model.model.model_options.get("transformer_options", {})
                if "patches" in to:
                    logger.debug(f"Clearing {len(to['patches'])} patches from model")
                    to["patches"] = {}
            
            # Clear Token Merging
            if hasattr(model.model, "remove_tome"):
                model.model.remove_tome()

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
                self._clear_model_patches(self._model)
                return self._model
            
            # 3. Different model requested: UNLOAD OLD ONE FIRST to free VRAM
            logger.info(f"Unloading {current_type} model to load {target_type}")
            self._model.unload()
            self._model = None
            
            # Clear prompt cache since the CLIP model is changing
            try:
                from src.Utilities.prompt_cache import clear_prompt_cache
                clear_prompt_cache()
            except Exception:
                pass
            
            # Force cleanup to prevent memory pressure/stuttering during transition
            import gc
            gc.collect()
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
                self._clear_model_patches(self._model)
                return self._model
            
            # 3. Different model requested: UNLOAD OLD ONE FIRST to free VRAM
            logger.info(f"Unloading current model to load refiner {target_type}")
            self._model.unload()
            # self._model = None # Don't set to None yet, we'll replace it
            
            # Force cleanup
            import gc
            gc.collect()
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
        self._apply_runtime_preferences(ctx, model)

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
        
        # StableFast and torch.compile are mutually exclusive
        if ctx.generation.stable_fast:
            model.apply_stable_fast(enable_cuda_graph=True)
        elif ctx.generation.torch_compile:
            model.apply_torch_compile()
        
        # FP8 quantization (hardware-gated, applies independently)
        if ctx.generation.fp8_inference or ctx.generation.weight_quantization == "fp8":
            model.apply_fp8()
        elif ctx.generation.weight_quantization == "nvfp4":
            model.apply_nvfp4()
        
        # Token Merging (ToMe)
        if ctx.sampling.tome_enabled and getattr(model.capabilities, 'supports_tome', True):
            try:
                if hasattr(model.model, 'apply_tome'):
                    model.model.apply_tome(
                        ratio=ctx.sampling.tome_ratio,
                        max_downsample=ctx.sampling.tome_max_downsample,
                    )
            except Exception as e:
                logger.warning(f"ToMe application failed: {e}")
        
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


def reset_default_pipeline() -> None:
    """Release the singleton pipeline and any loaded model it still owns."""
    global _default_pipeline
    if _default_pipeline is not None:
        try:
            if _default_pipeline._model is not None and _default_pipeline._model.is_loaded:
                _default_pipeline._model.unload()
        except Exception:
            pass
        _default_pipeline = None
