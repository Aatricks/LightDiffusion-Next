"""LightDiffusion-Next Modular Pipeline.

This module provides a clean, modular pipeline for image generation that
replaces the monolithic if/else structure with pluggable components.

Architecture:
    - PipelineContext: Holds all configuration in organized dataclasses
    - AbstractModel: Interface for SD15/SDXL models
    - Processors: Post-processing components (HiresFix, Adetailer, Img2Img)

The main flow is now:
    1. Create context from parameters
    2. Load model
    3. Encode prompts
    4. Generate latents
    5. Apply optional processors (HiresFix, Adetailer)
    6. Decode and save

Example:
    from src.user.pipeline import pipeline
    
    pipeline(
        prompt="a beautiful landscape",
        w=512,
        h=512,
        hires_fix=True,
        adetailer=True,
    )
"""

import argparse
import logging
import os
import random
import time
from typing import Any, Optional

import torch

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the downloader check once at module load
from src.FileManaging import Downloader
Downloader.CheckAndDownload()

# Load last seed
try:
    with open(os.path.join("./include/", "last_seed.txt"), "r") as f:
        _last_seed = int(f.read().strip())
except Exception:
    _last_seed = random.randint(1, 2**64)


def _check_interruption():
    """Check if generation has been interrupted by the user."""
    from src.user import app_instance
    
    app = getattr(app_instance, "app", None)
    if app is not None and getattr(app, "interrupt_flag", False):
        raise InterruptedError("Generation interrupted")


def _resolve_checkpoint_path(model_path: str = None, realistic_model: bool = False) -> str:
    """Determine the checkpoint path based on priority and flags."""
    if model_path:
        return model_path
    if realistic_model:
        return "./include/checkpoints/DreamShaper_8_pruned.safetensors"
    return "./include/checkpoints/DreamShaper_8_pruned.safetensors"


def pipeline(
    prompt: str | list,
    w: int,
    h: int,
    number: int = 1,
    batch: int = 1,
    scheduler: str = "ays",
    sampler: str = "dpmpp_sde_cfgpp",
    steps: int = 20,
    hires_fix: bool = False,
    adetailer: bool = False,
    enhance_prompt: bool = False,
    img2img: bool = False,
    stable_fast: bool = False,
    reuse_seed: bool = False,
    autohdr: bool = True,
    realistic_model: bool = False,
    model_path: str | None = None,
    negative_prompt: str = "",
    # Multi-scale diffusion parameters
    multiscale_preset: str = "quality",
    enable_multiscale: bool = True,
    multiscale_factor: float = 0.5,
    multiscale_fullres_start: int = 3,
    multiscale_fullres_end: int = 8,
    multiscale_intermittent_fullres: bool = False,
    # DeepCache parameters
    deepcache_enabled: bool = False,
    deepcache_interval: int = 3,
    deepcache_depth: int = 2,
    deepcache_start_step: int = 0,
    deepcache_end_step: int = 1000,
    # CFG-free sampling parameters
    cfg_free_enabled: bool = False,
    cfg_free_start_percent: float = 70.0,
    # Token Merging parameters
    tome_enabled: bool = False,
    tome_ratio: float = 0.5,
    tome_max_downsample: int = 1,
    # Advanced CFG optimization parameters
    batched_cfg: bool = True,
    dynamic_cfg_rescaling: bool = False,
    dynamic_cfg_method: str = "variance",
    dynamic_cfg_percentile: float = 95.0,
    dynamic_cfg_target_scale: float = 7.0,
    adaptive_noise_enabled: bool = False,
    adaptive_noise_method: str = "complexity",
    # Img2img parameters
    img2img_image: str | None = None,
    # Batched mode parameters
    per_sample_info: list | None = None,
) -> dict:
    """Run the LightDiffusion pipeline.

    This is the main entry point for image generation. It uses a modular
    architecture where features are cleanly separated into components.

    Args:
        prompt: Text prompt(s) for generation
        w: Width of generated image
        h: Height of generated image
        number: Number of images to generate
        batch: Batch size for generation
        scheduler: Scheduler to use (normal, karras, simple, beta, ays, ays_sd15, ays_sdxl)
        sampler: Sampler to use (euler, euler_ancestral, dpmpp_2m_cfgpp, dpmpp_sde_cfgpp, etc)
        steps: Number of sampling steps
        hires_fix: Enable high-resolution fix
        adetailer: Enable automatic face/body enhancement
        enhance_prompt: Enable Ollama prompt enhancement
        img2img: Enable image-to-image mode
        stable_fast: Enable StableFast optimization
        reuse_seed: Reuse the last seed
        autohdr: Enable AutoHDR post-processing
        realistic_model: Use the realistic model
        model_path: Explicit path to model checkpoint
        negative_prompt: Negative prompt
        multiscale_preset: Preset for multi-scale diffusion
        enable_multiscale: Enable multi-scale diffusion
        deepcache_enabled: Enable DeepCache acceleration
        cfg_free_enabled: Enable CFG-free sampling
        tome_enabled: Enable Token Merging
        img2img_image: Path to source image for img2img
        per_sample_info: Per-sample data for batched mode

    Returns:
        Dictionary with generation results and metadata
    """
    global _last_seed
    
    # Clear any previous interrupt
    from src.user import app_instance
    app_ref = getattr(app_instance, "app", None)
    if app_ref is not None:
        app_ref.clear_interrupt()
    
    _check_interruption()
    
    # Create pipeline context from kwargs
    from src.Core import PipelineContext, create_model
    
    ctx = PipelineContext.from_kwargs(
        prompt=prompt,
        w=w,
        h=h,
        number=number,
        batch=batch,
        scheduler=scheduler,
        sampler=sampler,
        steps=steps,
        hires_fix=hires_fix,
        adetailer=adetailer,
        enhance_prompt=enhance_prompt,
        img2img=img2img,
        stable_fast=stable_fast,
        reuse_seed=reuse_seed,
        autohdr=autohdr,
        realistic_model=realistic_model,
        model_path=model_path,
        negative_prompt=negative_prompt,
        multiscale_preset=multiscale_preset,
        enable_multiscale=enable_multiscale,
        multiscale_factor=multiscale_factor,
        multiscale_fullres_start=multiscale_fullres_start,
        multiscale_fullres_end=multiscale_fullres_end,
        multiscale_intermittent_fullres=multiscale_intermittent_fullres,
        deepcache_enabled=deepcache_enabled,
        deepcache_interval=deepcache_interval,
        deepcache_depth=deepcache_depth,
        deepcache_start_step=deepcache_start_step,
        deepcache_end_step=deepcache_end_step,
        cfg_free_enabled=cfg_free_enabled,
        cfg_free_start_percent=cfg_free_start_percent,
        tome_enabled=tome_enabled,
        tome_ratio=tome_ratio,
        tome_max_downsample=tome_max_downsample,
        batched_cfg=batched_cfg,
        dynamic_cfg_rescaling=dynamic_cfg_rescaling,
        dynamic_cfg_method=dynamic_cfg_method,
        dynamic_cfg_percentile=dynamic_cfg_percentile,
        dynamic_cfg_target_scale=dynamic_cfg_target_scale,
        adaptive_noise_enabled=adaptive_noise_enabled,
        adaptive_noise_method=adaptive_noise_method,
        img2img_image=img2img_image,
    )
    
    # Handle prompt enhancement
    original_prompt = prompt
    enhancement_applied = False
    
    if enhance_prompt:
        ctx, enhancement_applied = _apply_prompt_enhancement(ctx)
    
    # Update seeds from context
    if reuse_seed:
        ctx.seeds = [_last_seed] * ctx.total_images
        ctx.seed = _last_seed
    
    # Save seed for potential reuse
    ctx.save_last_seed()
    _last_seed = ctx.seeds[-1] if ctx.seeds else ctx.seed
    
    # Resolve model path
    resolved_path = _resolve_checkpoint_path(model_path, realistic_model)
    ctx.generation.model_path = resolved_path
    
    with torch.inference_mode():
        # ===== IMAGE-TO-IMAGE MODE =====
        if ctx.features.img2img:
            return _run_img2img_pipeline(ctx, original_prompt, enhancement_applied)
        
        # ===== BATCHED MODE =====
        if ctx.is_batched:
            return _run_batched_pipeline(ctx, per_sample_info, original_prompt, enhancement_applied)
        
        # ===== STANDARD TEXT-TO-IMAGE MODE =====
        return _run_standard_pipeline(ctx, original_prompt, enhancement_applied)


def _apply_prompt_enhancement(ctx: "PipelineContext") -> tuple["PipelineContext", bool]:
    """Apply prompt enhancement using Ollama if available.
    
    Args:
        ctx: Pipeline context
        
    Returns:
        Tuple of (updated_context, enhancement_applied)
    """
    from src.Utilities import Enhancer
    
    try:
        if isinstance(ctx.prompt, (list, tuple)):
            enhanced_prompts = []
            for p in ctx.prompt:
                try:
                    enhanced = Enhancer.enhance_prompt(p)
                    enhanced_prompts.append(enhanced if enhanced else p)
                except Exception:
                    enhanced_prompts.append(p)
            ctx.prompt = enhanced_prompts
        else:
            enhanced = Enhancer.enhance_prompt(ctx.prompt)
            if enhanced:
                ctx.prompt = enhanced
        return ctx, True
    except Exception:
        return ctx, False


def _run_standard_pipeline(
    ctx: "PipelineContext",
    original_prompt: Any,
    enhancement_applied: bool,
) -> dict:
    """Run standard text-to-image generation.
    
    Args:
        ctx: Pipeline context with all configuration
        original_prompt: Original prompt before enhancement
        enhancement_applied: Whether prompt was enhanced
        
    Returns:
        Dictionary with generation results
    """
    from src.Core import create_model
    from src.Processors import HiresFix, Adetailer
    from src.FileManaging import ImageSaver
    from src.AutoHDR import ahdr
    
    # Create and load model
    model = create_model(ctx.generation.model_path)
    model.load()
    
    # Apply LoRA
    try:
        model.apply_lora("add_detail.safetensors", 0.7, 0.7)
    except Exception as e:
        logger.warning(f"LoRA loading failed: {e}")
    
    # Apply optimizations based on context settings
    if ctx.generation.stable_fast:
        model.apply_stable_fast(enable_cuda_graph=False)
    
    if ctx.sampling.deepcache_enabled:
        model.apply_deepcache(
            cache_interval=ctx.sampling.deepcache_interval,
            cache_depth=ctx.sampling.deepcache_depth,
            start_step=ctx.sampling.deepcache_start_step,
            end_step=ctx.sampling.deepcache_end_step,
        )
    
    # Encode prompts
    positive, negative = model.encode_prompt(
        prompt=ctx.prompt,
        negative_prompt=ctx.negative_prompt,
    )
    
    # Generate for each seed
    saveimage = ImageSaver.SaveImage()
    hdr = ahdr.HDREffects()
    
    for i, current_seed in enumerate(ctx.seeds[:ctx.generation.number]):
        _check_interruption()
        
        # Update context with current seed
        ctx.seed = current_seed
        
        # Generate latents
        latents = model.generate(ctx, positive, negative)
        
        # ===== HIRES FIX =====
        if HiresFix.is_enabled(ctx):
            latents = HiresFix.apply(
                latents=latents,
                ctx=ctx,
                model=model,
                positive=positive,
                negative=negative,
            )
        
        _check_interruption()
        
        # Decode latents to image
        image = model.decode(latents["samples"])
        
        # Apply AutoHDR
        if ctx.generation.autohdr:
            try:
                hdr_result = hdr.apply_hdr2(image)
                image = hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result
            except Exception:
                pass
        
        # ===== ADETAILER =====
        if Adetailer.is_enabled(ctx):
            image, _ = Adetailer.apply(
                image=image,
                ctx=ctx,
                model=model,
                negative=negative,
            )
        else:
            # Save the generated image
            metadata = ctx.build_metadata()
            prefix = "LD-HF" if ctx.features.hires_fix else "LD"
            
            saveimage.save_images_async(
                filename_prefix=prefix,
                images=image,
                prompt=ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt),
                extra_pnginfo=metadata,
            )
    
    return {
        "original_prompt": original_prompt,
        "used_prompt": ctx.prompt,
        "enhancement_applied": enhancement_applied,
    }


def _run_batched_pipeline(
    ctx: "PipelineContext",
    per_sample_info: list | None,
    original_prompt: Any,
    enhancement_applied: bool,
) -> dict:
    """Run batched multi-prompt generation.
    
    Args:
        ctx: Pipeline context
        per_sample_info: Per-sample configuration
        original_prompt: Original prompts before enhancement
        enhancement_applied: Whether prompts were enhanced
        
    Returns:
        Dictionary with batched results mapping
    """
    import uuid
    from src.Core import create_model
    from src.Processors import HiresFix, Adetailer
    from src.FileManaging import ImageSaver
    from src.AutoHDR import ahdr
    
    prompts = list(ctx.prompt)
    total_batch = len(prompts)
    
    # Build negative prompts list
    if isinstance(ctx.negative_prompt, (list, tuple)):
        negatives = list(ctx.negative_prompt)
    else:
        negatives = [ctx.negative_prompt] * total_batch
    
    # Ensure per_sample_info exists
    if per_sample_info is None:
        per_sample_info = [{} for _ in range(total_batch)]
    
    # Create and load model
    model = create_model(ctx.generation.model_path)
    model.load()
    
    # Apply LoRA
    try:
        model.apply_lora("add_detail.safetensors", 0.7, 0.7)
    except Exception as e:
        logger.warning(f"LoRA loading failed: {e}")
    
    # Apply optimizations
    if ctx.generation.stable_fast:
        model.apply_stable_fast(enable_cuda_graph=True)
    
    if ctx.sampling.deepcache_enabled:
        model.apply_deepcache(
            cache_interval=ctx.sampling.deepcache_interval,
            cache_depth=ctx.sampling.deepcache_depth,
        )
    
    # Encode all prompts
    positive, negative = model.encode_prompt(
        prompt=prompts,
        negative_prompt=negatives,
    )
    
    # Add batch index routing
    if isinstance(positive, list):
        for i, entry in enumerate(positive):
            if len(entry) > 1 and isinstance(entry[1], dict):
                entry[1]["batch_index"] = [i]
    
    if isinstance(negative, list):
        for i, entry in enumerate(negative):
            if len(entry) > 1 and isinstance(entry[1], dict):
                entry[1]["batch_index"] = [i]
    
    # Update context for batched generation
    batch_ctx = ctx.clone()
    batch_ctx.generation.batch = total_batch
    
    # Generate all latents in one pass
    from src.sample import sampling
    from src.Utilities import Latent
    from src.hidiffusion import msw_msa_attention
    
    latent_gen = Latent.EmptyLatentImage()
    latent = latent_gen.generate(
        width=ctx.generation.width,
        height=ctx.generation.height,
        batch_size=total_batch,
    )[0]
    latent["seeds"] = ctx.seeds[:total_batch]
    
    # Apply HiDiffusion
    try:
        hidiff = msw_msa_attention.ApplyMSWMSAAttentionSimple()
        optimized_model = hidiff.go(model_type="auto", model=model.model)[0]
    except Exception:
        optimized_model = model.model
    
    ksampler = sampling.KSampler()
    batch_latents = ksampler.sample(
        seed=None,
        steps=ctx.sampling.steps,
        cfg=ctx.sampling.cfg,
        sampler_name=ctx.sampling.sampler,
        scheduler=ctx.sampling.scheduler,
        denoise=1.0,
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
    )
    
    # Decode all images
    images = model.decode(batch_latents[0]["samples"])
    
    # Apply AutoHDR
    hdr = ahdr.HDREffects()
    if ctx.generation.autohdr:
        try:
            hdr_result = hdr.apply_hdr2(images)
            images = hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result
        except Exception:
            pass
    
    # Process each image individually
    saveimage = ImageSaver.SaveImage()
    results_map = {}
    
    for i in range(total_batch):
        _check_interruption()
        
        info = per_sample_info[i] if i < len(per_sample_info) else {}
        req_id = info.get("request_id", uuid.uuid4().hex[:8])
        filename_prefix = info.get("filename_prefix", f"LD-REQ-{req_id}")
        
        final_image = images[i]
        
        # Per-sample HiresFix
        if info.get("hires_fix", False):
            try:
                single_latent = {"samples": batch_latents[0]["samples"][i:i+1]}
                single_ctx = ctx.clone()
                single_ctx.seed = ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed
                
                hires_latents = HiresFix.apply(
                    latents=single_latent,
                    ctx=single_ctx,
                    model=model,
                    positive=[positive[i]] if isinstance(positive, list) else positive,
                    negative=[negative[i]] if isinstance(negative, list) else negative,
                )
                decoded = model.decode(hires_latents["samples"])
                if ctx.generation.autohdr:
                    hdr_result = hdr.apply_hdr2(decoded)
                    final_image = (hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result)[0]
                else:
                    final_image = decoded[0]
            except Exception as e:
                logger.exception(f"Per-sample hires_fix failed for index {i}: {e}")
        
        # Per-sample Adetailer
        if info.get("adetailer", False):
            try:
                single_ctx = ctx.clone()
                single_ctx.seed = ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed
                
                final_image, saved = Adetailer.apply(
                    image=final_image,
                    ctx=single_ctx,
                    model=model,
                    negative=[negative[i]] if isinstance(negative, list) else negative,
                )
                results_map.setdefault(req_id, [])
                for s in saved:
                    try:
                        results_map[req_id].extend(s.get("ui", {}).get("images", []))
                    except Exception:
                        results_map[req_id].append(s)
            except Exception as e:
                logger.exception(f"Per-sample adetailer failed for index {i}: {e}")
        
        # Save final image
        sample_meta = ctx.build_metadata({
            "seed": str(ctx.seeds[i] if i < len(ctx.seeds) else ctx.seed),
            "prompt": prompts[i],
            "negative_prompt": negatives[i],
        })
        
        saved = saveimage.save_images(
            filename_prefix=filename_prefix,
            images=[final_image],
            prompt=prompts[i],
            extra_pnginfo=sample_meta,
        )
        
        results_map.setdefault(req_id, [])
        try:
            results_map[req_id].extend(saved.get("ui", {}).get("images", []))
        except Exception:
            results_map[req_id].append(saved)
    
    return {"batched_results": results_map}


def _run_img2img_pipeline(
    ctx: "PipelineContext",
    original_prompt: Any,
    enhancement_applied: bool,
) -> dict:
    """Run image-to-image generation.
    
    Args:
        ctx: Pipeline context
        original_prompt: Original prompt
        enhancement_applied: Whether prompt was enhanced
        
    Returns:
        Dictionary with generation results
    """
    from src.Core import create_model
    from src.Processors import Img2Img
    from src.FileManaging import ImageSaver
    from src.AutoHDR import ahdr
    
    # Create and load model
    model = create_model(ctx.generation.model_path)
    model.load()
    
    # Apply LoRA with higher strength for img2img
    try:
        model.apply_lora("add_detail.safetensors", 2.0, 2.0)
    except Exception as e:
        logger.warning(f"LoRA loading failed: {e}")
    
    # Apply optimizations
    if ctx.generation.stable_fast:
        model.apply_stable_fast(enable_cuda_graph=True)
    
    if ctx.sampling.deepcache_enabled:
        model.apply_deepcache(
            cache_interval=ctx.sampling.deepcache_interval,
            cache_depth=ctx.sampling.deepcache_depth,
        )
    
    # Encode prompts
    positive, negative = model.encode_prompt(
        prompt=ctx.prompt,
        negative_prompt=ctx.negative_prompt,
    )
    
    saveimage = ImageSaver.SaveImage()
    hdr = ahdr.HDREffects()
    
    for current_seed in ctx.seeds[:ctx.generation.number]:
        _check_interruption()
        ctx.seed = current_seed
        
        # Apply img2img processor
        result = Img2Img.apply(
            ctx=ctx,
            model=model,
            positive=positive,
            negative=negative,
        )
        
        _check_interruption()
        
        # Apply AutoHDR
        if ctx.generation.autohdr:
            try:
                hdr_result = hdr.apply_hdr2(result)
                result = hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result
            except Exception:
                pass
        
        # Save result
        metadata = ctx.build_metadata({"img2img": "True", "upscale_model": "RealESRGAN_x4plus.pth"})
        
        saveimage.save_images(
            filename_prefix="LD-I2I",
            images=result,
            prompt=ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt),
            extra_pnginfo=metadata,
        )
    
    return {
        "original_prompt": original_prompt,
        "used_prompt": ctx.prompt,
        "enhancement_applied": enhancement_applied,
    }


# ===== CLI INTERFACE =====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LightDiffusion pipeline.")
    parser.add_argument("prompt", type=str, help="The prompt for the pipeline.")
    parser.add_argument("width", type=int, help="The width of the generated image.")
    parser.add_argument("height", type=int, help="The height of the generated image.")
    parser.add_argument("number", type=int, help="The number of images to generate.")
    parser.add_argument("batch", type=int, help="The batch size.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ays",
        choices=["normal", "karras", "simple", "beta", "ays", "ays_sd15", "ays_sdxl", "ays_flux"],
        help="The scheduler to use for sampling.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="dpmpp_sde_cfgpp",
        choices=["euler", "euler_ancestral", "euler_cfgpp", "euler_ancestral_cfgpp", "dpmpp_2m_cfgpp", "dpmpp_sde_cfgpp"],
        help="The sampler to use for sampling.",
    )
    parser.add_argument("--steps", type=int, default=20, help="The number of sampling steps.")
    parser.add_argument("--hires-fix", action="store_true", help="Enable high-resolution fix.")
    parser.add_argument("--adetailer", action="store_true", help="Enable automatic face and body enhancing.")
    parser.add_argument("--enhance-prompt", action="store_true", help="Enable Ollama prompt enhancement.")
    parser.add_argument("--img2img", action="store_true", help="Enable image-to-image mode.")
    parser.add_argument("--stable-fast", action="store_true", help="Enable StableFast mode.")
    parser.add_argument("--reuse-seed", action="store_true", help="Enable seed reuse.")
    parser.add_argument("--autohdr", action="store_true", help="Enable the AutoHDR mode.")
    parser.add_argument("--realistic-model", action="store_true", help="Use the realistic model.")
    parser.add_argument("--model-path", type=str, default="", help="Optional path to a model file.")
    parser.add_argument("--multiscale-preset", type=str, choices=["quality", "performance", "balanced", "disabled"], help="Multiscale preset.")
    parser.add_argument("--enable-multiscale", action="store_true", default=True, help="Enable multi-scale diffusion.")
    parser.add_argument("--multiscale-factor", type=float, default=0.5, help="Scale factor for intermediate steps.")
    parser.add_argument("--multiscale-fullres-start", type=int, default=3, help="First steps at full resolution.")
    parser.add_argument("--multiscale-fullres-end", type=int, default=8, help="Last steps at full resolution.")
    parser.add_argument("--multiscale-intermittent-fullres", action="store_true", help="Enable intermittent full-res.")
    parser.add_argument("--deepcache", action="store_true", help="Enable DeepCache acceleration.")
    parser.add_argument("--deepcache-interval", type=int, default=3, help="Steps between cache updates.")
    parser.add_argument("--deepcache-depth", type=int, default=2, help="U-Net depth for caching.")
    parser.add_argument("--deepcache-start-step", type=int, default=0, help="Start applying DeepCache.")
    parser.add_argument("--deepcache-end-step", type=int, default=1000, help="Stop applying DeepCache.")
    
    args = parser.parse_args()
    
    pipeline(
        args.prompt,
        args.width,
        args.height,
        args.number,
        args.batch,
        args.scheduler,
        args.sampler,
        args.steps,
        args.hires_fix,
        args.adetailer,
        args.enhance_prompt,
        args.img2img,
        args.stable_fast,
        args.reuse_seed,
        args.autohdr,
        args.realistic_model,
        args.model_path or None,
        multiscale_preset=args.multiscale_preset,
        enable_multiscale=args.enable_multiscale,
        multiscale_factor=args.multiscale_factor,
        multiscale_fullres_start=args.multiscale_fullres_start,
        multiscale_fullres_end=args.multiscale_fullres_end,
        multiscale_intermittent_fullres=args.multiscale_intermittent_fullres,
        deepcache_enabled=args.deepcache,
        deepcache_interval=args.deepcache_interval,
        deepcache_depth=args.deepcache_depth,
        deepcache_start_step=args.deepcache_start_step,
        deepcache_end_step=args.deepcache_end_step,
    )
