"""LightDiffusion-Next Pipeline Entry Point.

This module provides the main `pipeline()` function that all UIs call.
It's a thin wrapper around the Core Pipeline class for backward compatibility.

Usage:
    from src.user.pipeline import pipeline
    
    result = pipeline(
        prompt="a beautiful landscape",
        w=512, h=512,
        hires_fix=True,
        adetailer=True,
    )
"""

import logging
import os
import random
from typing import Callable

import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_checkpoint_path(realistic_model: bool = False) -> str:
    """Resolve the checkpoint path based on model settings."""
    return "./include/checkpoints/DreamShaper_8_pruned.safetensors"


# Initialize downloader check once at module load
from src.FileManaging import Downloader
Downloader.CheckAndDownload()

# Load last seed
try:
    with open(os.path.join("./include/", "last_seed.txt"), "r") as f:
        _last_seed = int(f.read().strip())
except Exception:
    _last_seed = random.randint(1, 2**63 - 1)


def pipeline(
    prompt: str | list,
    w: int,
    h: int,
    number: int = 1,
    batch: int = 1,
    scheduler: str = "ays",
    sampler: str = "dpmpp_sde",
    steps: int = 20,
    cfg_scale: float = 7.0,
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
    # Multi-scale diffusion
    multiscale_preset: str = "disabled",
    enable_multiscale: bool = False,
    multiscale_factor: float = 0.5,
    multiscale_fullres_start: int = 3,
    multiscale_fullres_end: int = 8,
    multiscale_intermittent_fullres: bool = False,
    # DeepCache
    deepcache_enabled: bool = False,
    deepcache_interval: int = 3,
    deepcache_depth: int = 2,
    deepcache_start_step: int = 0,
    deepcache_end_step: int = 1000,
    # CFG-free
    cfg_free_enabled: bool = False,
    cfg_free_start_percent: float = 70.0,
    # Token Merging
    tome_enabled: bool = False,
    tome_ratio: float = 0.5,
    tome_max_downsample: int = 1,
    # Advanced CFG
    batched_cfg: bool = True,
    dynamic_cfg_rescaling: bool = False,
    dynamic_cfg_method: str = "variance",
    dynamic_cfg_percentile: float = 95.0,
    dynamic_cfg_target_scale: float = 7.0,
    adaptive_noise_enabled: bool = False,
    adaptive_noise_method: str = "complexity",
    # Img2img
    img2img_image: str | None = None,
    request_filename_prefix: str | None = None,
    img2img_denoise: float = 0.75,  # Denoising strength: 0=no change, 1=full gen
    # Refiner
    refiner_model_path: str | None = None,
    refiner_switch_step: int | None = None,
    # ControlNet
    controlnet_model: str | None = None,
    controlnet_strength: float = 1.0,
    controlnet_type: str = "canny",
    # Batched mode
    per_sample_info: list | None = None,
    # External callback
    callback: Callable | None = None,
) -> dict:
    """Run the LightDiffusion pipeline.

    This is the main entry point for image generation. All parameters
    are collected into a Context and passed to the Pipeline.

    Args:
        prompt: Text prompt(s) for generation
        w: Width of generated image
        h: Height of generated image
        number: Number of images to generate
        batch: Batch size
        scheduler: Scheduler name
        sampler: Sampler name
        steps: Sampling steps
        hires_fix: Enable high-resolution fix
        adetailer: Enable face/body enhancement
        enhance_prompt: Enable Ollama prompt enhancement
        img2img: Enable image-to-image mode
        stable_fast: Enable StableFast optimization
        reuse_seed: Reuse last seed
        autohdr: Enable AutoHDR
        realistic_model: Use realistic model
        model_path: Path to model checkpoint
        negative_prompt: Negative prompt
        multiscale_preset: Multi-scale preset
        enable_multiscale: Enable multi-scale diffusion
        deepcache_enabled: Enable DeepCache
        cfg_free_enabled: Enable CFG-free sampling
        tome_enabled: Enable Token Merging
        img2img_image: Source image for img2img
        per_sample_info: Per-sample data for batched mode

    Returns:
        Dictionary with generation results
    """
    global _last_seed
    
    # Clear interrupt flag
    from src.user import app_instance
    app_ref = getattr(app_instance, "app", None)
    if app_ref is not None:
        app_ref.clear_interrupt()
    
    # Build context from kwargs
    from src.Core.Context import Context
    from src.Core.Pipeline import Pipeline, get_default_pipeline
    
    ctx = Context.from_kwargs(
        prompt=prompt,
        w=w, h=h,
        number=number,
        batch=batch,
        scheduler=scheduler,
        sampler=sampler,
        steps=steps,
        cfg_scale=cfg_scale,
        hires_fix=hires_fix,
        adetailer=adetailer,
        enhance_prompt=enhance_prompt,
        img2img=img2img,
        stable_fast=stable_fast,
        reuse_seed=reuse_seed,
        autohdr=autohdr,
        model_path=model_path or (
            "./include/checkpoints/DreamShaper_8_pruned.safetensors" if realistic_model
            else "./include/checkpoints/DreamShaper_8_pruned.safetensors"
        ),
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
        request_filename_prefix=request_filename_prefix,
        img2img_denoise=img2img_denoise,
        refiner_model_path=refiner_model_path,
        refiner_switch_step=refiner_switch_step,
        controlnet_model=controlnet_model,
        controlnet_strength=controlnet_strength,
        controlnet_type=controlnet_type,
    )
    
    # Handle prompt enhancement
    original_prompt = prompt
    enhancement_applied = False
    
    if enhance_prompt:
        ctx, enhancement_applied = _enhance_prompt(ctx)
    
    # Handle seed reuse
    if reuse_seed:
        ctx.seeds = [_last_seed] * ctx.total_images
        ctx.seed = _last_seed
    
    # Save seed for future reuse
    _last_seed = ctx.seeds[-1] if ctx.seeds else ctx.seed
    
    # Setup default callback for UI preview
    # Setup default callback for UI preview
    def default_callback(args: dict):
        from src.user import app_instance
        from src.AutoEncoders import taesd
        app_ref = getattr(app_instance, "app", None)
        
        # Streamlit/Gradio UI preview
        if app_ref is not None:
            step = args.get("i", 0)
            x0 = args.get("denoised")
            total_steps = args.get("total_steps", ctx.sampling.steps)
            
            # Update progress tracker
            if total_steps > 0:
                app_ref.progress.set((step + 1) / total_steps)
            
            # Update preview (x0 is the denoised latent estimate)
            if x0 is not None:
                is_flux = x0.shape[1] in (16, 32)
                # taesd_preview handles PIL conversion and calls app_ref.update_image
                taesd.taesd_preview(x0, flux=is_flux, step=step, total_steps=total_steps)
            else:
                # Just update step info if no image is available
                app_ref.update_image(app_ref.preview_images, step=step, total_steps=total_steps)
        
        # Chain external callback if provided
        if callback is not None:
            try:
                callback(args)
            except Exception:
                pass
            
    ctx.callback = default_callback
    
    # Run pipeline
    pipeline_instance = get_default_pipeline()
    
    with torch.inference_mode():
        if ctx.features.controlnet_model:
            # ControlNet mode (uses input image for control, generates new content)
            pipeline_instance.run_controlnet(ctx)
        elif ctx.features.img2img:
            pipeline_instance.run_img2img(ctx)
        elif ctx.is_batched:
            return pipeline_instance.run_batched(ctx, per_sample_info)
        else:
            pipeline_instance.run(ctx)
    
    return {
        "original_prompt": original_prompt,
        "used_prompt": ctx.prompt,
        "enhancement_applied": enhancement_applied,
    }


def _enhance_prompt(ctx: "Context") -> tuple["Context", bool]:
    """Apply Ollama prompt enhancement if available."""
    from src.Utilities import Enhancer
    
    try:
        if isinstance(ctx.prompt, (list, tuple)):
            enhanced = []
            for p in ctx.prompt:
                try:
                    e = Enhancer.enhance_prompt(p)
                    enhanced.append(e if e else p)
                except Exception:
                    enhanced.append(p)
            ctx.prompt = enhanced
        else:
            e = Enhancer.enhance_prompt(ctx.prompt)
            if e:
                ctx.prompt = e
        return ctx, True
    except Exception:
        return ctx, False


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightDiffusion Pipeline CLI")
    parser.add_argument("prompt", type=str, help="Generation prompt")
    parser.add_argument("width", type=int, help="Image width")
    parser.add_argument("height", type=int, help="Image height")
    parser.add_argument("number", type=int, default=1, help="Number of images")
    parser.add_argument("batch", type=int, default=1, help="Batch size")
    parser.add_argument("--scheduler", type=str, default="karras")
    parser.add_argument("--sampler", type=str, default="dpmpp_2m_cfgpp")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hires-fix", action="store_true")
    parser.add_argument("--adetailer", action="store_true")
    parser.add_argument("--stable-fast", action="store_true")
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument("--model-path", type=str, default="")
    
    args = parser.parse_args()
    
    pipeline(
        args.prompt,
        args.width,
        args.height,
        args.number,
        args.batch,
        scheduler=args.scheduler,
        sampler=args.sampler,
        steps=args.steps,
        hires_fix=args.hires_fix,
        adetailer=args.adetailer,
        stable_fast=args.stable_fast,
        deepcache_enabled=args.deepcache,
        model_path=args.model_path or None,
    )
