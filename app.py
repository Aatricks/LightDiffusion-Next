from __future__ import annotations

import glob
import os
import time
import uuid
from typing import Any, Optional

import gradio as gr
import spaces
from PIL import Image

from src.Core.Models.ModelFactory import list_available_models
from src.Device.ModelCache import get_model_cache
from src.user import app_instance
from src.user.pipeline import pipeline


SCHEDULER_CHOICES = [
    "ays",
    "ays_sd15",
    "ays_sdxl",
    "karras",
    "normal",
    "simple",
    "beta",
]

SAMPLER_CHOICES = [
    "dpmpp_sde_cfgpp",
    "dpmpp_2m_cfgpp",
    "euler",
    "euler_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "euler_cfgpp",
    "euler_ancestral_cfgpp",
]


def _list_model_mapping() -> list[tuple[str, str]]:
    return list_available_models(return_mapping=True)


def _model_choices() -> list[str]:
    return [name for name, _ in _list_model_mapping()]


def _resolve_model_path(display_name: Optional[str]) -> Optional[str]:
    if not display_name:
        return None

    for name, path in _list_model_mapping():
        if name == display_name:
            return path
    return None


def _load_recent_images(
    prefix: Optional[str] = None,
    started_at: Optional[float] = None,
    limit: int = 12,
) -> list[Image.Image]:
    files: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files.extend(glob.glob(os.path.join(".", "output", "**", ext), recursive=True))

    filtered: list[str] = []
    for path in files:
        basename = os.path.basename(path)
        if prefix and prefix not in basename:
            continue
        if started_at is not None:
            try:
                if os.path.getmtime(path) < (started_at - 1.0):
                    continue
            except OSError:
                continue
        filtered.append(path)

    filtered.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    images: list[Image.Image] = []
    for path in filtered[:limit]:
        try:
            with Image.open(path) as img:
                images.append(img.copy())
        except Exception:
            continue
    return images


def _refresh_history() -> tuple[list[Image.Image], str]:
    images = _load_recent_images(limit=48)
    if not images:
        return [], "No generated images found yet."
    return images, f"Loaded {len(images)} recent images from `output/`."


def _interrupt_generation() -> str:
    app_instance.app.request_interrupt()
    return "Interrupt requested. The current generation will stop at the next safe check."


@spaces.GPU(duration=240)
def _run_generation(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_images: int,
    batch_size: int,
    scheduler: str,
    sampler: str,
    steps: int,
    guidance_scale: float,
    model_name: Optional[str],
    hires_fix: bool,
    adetailer: bool,
    enhance_prompt: bool,
    img2img_enabled: bool,
    img2img_image: Optional[str],
    img2img_denoise: float,
    stable_fast: bool,
    reuse_seed: bool,
    enable_multiscale: bool,
    multiscale_intermittent: bool,
    multiscale_factor: float,
    multiscale_fullres_start: int,
    multiscale_fullres_end: int,
    keep_models_loaded: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[list[Image.Image], str, dict[str, Any], list[Image.Image]]:
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    if img2img_enabled and not img2img_image:
        raise gr.Error("Upload an input image or disable Img2Img.")

    request_prefix = f"LD-GRADIO-{uuid.uuid4().hex[:8]}"
    started_at = time.time()

    app = app_instance.app
    app.clear_interrupt()
    app.cleanup_all_previews()
    app.previewer_var.set(True)

    try:
        try:
            get_model_cache().set_keep_models_loaded(bool(keep_models_loaded))
        except Exception:
            pass

        model_path = _resolve_model_path(model_name)

        def _progress_callback(args: dict[str, Any]) -> None:
            step = int(args.get("i", 0))
            total = int(args.get("total_steps", steps))
            if total > 0:
                progress(
                    min((step + 1) / total, 1.0),
                    desc=f"Sampling step {step + 1}/{total}",
                )

        progress(0, desc="Preparing generation")

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            w=int(width),
            h=int(height),
            number=int(num_images),
            batch=int(batch_size),
            scheduler=scheduler,
            sampler=sampler,
            steps=int(steps),
            cfg_scale=float(guidance_scale),
            hires_fix=bool(hires_fix),
            adetailer=bool(adetailer),
            enhance_prompt=bool(enhance_prompt),
            img2img=bool(img2img_enabled),
            img2img_image=img2img_image if img2img_enabled else None,
            img2img_denoise=float(img2img_denoise),
            stable_fast=bool(stable_fast),
            reuse_seed=bool(reuse_seed),
            autohdr=True,
            realistic_model=False,
            model_path=model_path,
            enable_multiscale=bool(enable_multiscale),
            multiscale_intermittent_fullres=bool(multiscale_intermittent),
            multiscale_factor=float(multiscale_factor),
            multiscale_fullres_start=int(multiscale_fullres_start),
            multiscale_fullres_end=int(multiscale_fullres_end),
            request_filename_prefix=request_prefix,
            callback=_progress_callback,
        )

        progress(1, desc="Generation complete")

        final_images = _load_recent_images(
            prefix=request_prefix,
            started_at=started_at,
            limit=max(1, int(num_images)),
        )
        if not final_images and adetailer:
            final_images = _load_recent_images(
                started_at=started_at,
                limit=max(1, int(num_images)),
            )

        preview_images = list(app.preview_images[:4]) if app.preview_images else []

        if not final_images:
            raise gr.Error("Generation completed but no output images were found in `output/`.")

        used_prompt = result.get("used_prompt", prompt) if isinstance(result, dict) else prompt
        metadata = {
            "request_prefix": request_prefix,
            "model_name": model_name or "auto/default",
            "used_prompt": used_prompt,
            "enhancement_applied": bool(result.get("enhancement_applied")) if isinstance(result, dict) else False,
            "img2img_enabled": bool(img2img_enabled),
            "adetailer": bool(adetailer),
            "hires_fix": bool(hires_fix),
        }
        status = f"Generated {len(final_images)} image(s) using `{sampler}` + `{scheduler}`."
        return final_images, status, metadata, preview_images
    finally:
        app.clear_interrupt()


def _build_demo() -> gr.Blocks:
    default_models = _model_choices()
    default_model = default_models[0] if default_models else None

    with gr.Blocks(title="LightDiffusion-Next ZeroGPU") as demo:
        gr.Markdown(
            """
            # LightDiffusion-Next
            ZeroGPU-compatible Gradio UI. The generation function is wrapped with `@spaces.GPU`
            so Hugging Face can allocate a GPU only while inference is running.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Describe the image you want to generate")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4)",
                )

                with gr.Row():
                    width = gr.Slider(256, 1536, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1536, value=512, step=64, label="Height")

                with gr.Row():
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Images")
                    batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")

                with gr.Row():
                    scheduler = gr.Dropdown(SCHEDULER_CHOICES, value="ays", label="Scheduler")
                    sampler = gr.Dropdown(SAMPLER_CHOICES, value="dpmpp_sde_cfgpp", label="Sampler")

                with gr.Row():
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.0, step=0.1, label="CFG")

                model_name = gr.Dropdown(
                    choices=default_models,
                    value=default_model,
                    allow_custom_value=False,
                    label="Model",
                )

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        hires_fix = gr.Checkbox(label="HiresFix", value=False)
                        adetailer = gr.Checkbox(label="ADetailer", value=False)
                        enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                        stable_fast = gr.Checkbox(label="Stable-Fast", value=False)
                    with gr.Row():
                        reuse_seed = gr.Checkbox(label="Reuse Last Seed", value=False)
                        enable_multiscale = gr.Checkbox(label="Multiscale", value=False)
                        multiscale_intermittent = gr.Checkbox(label="Intermittent Fullres", value=True)
                        keep_models_loaded = gr.Checkbox(label="Keep Models Loaded", value=False)
                    with gr.Row():
                        multiscale_factor = gr.Slider(0.25, 1.0, value=0.5, step=0.05, label="Multiscale Factor")
                        multiscale_fullres_start = gr.Slider(1, 20, value=10, step=1, label="Fullres Start")
                        multiscale_fullres_end = gr.Slider(1, 20, value=8, step=1, label="Fullres End")

                with gr.Accordion("Img2Img", open=False):
                    img2img_enabled = gr.Checkbox(label="Enable Img2Img", value=False)
                    img2img_image = gr.Image(label="Input Image", type="filepath")
                    img2img_denoise = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Denoise Strength")

                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    interrupt_button = gr.Button("Interrupt", variant="stop")
                    refresh_models_button = gr.Button("Refresh Models")

            with gr.Column(scale=3):
                status = gr.Markdown("Ready.")
                gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
                metadata = gr.JSON(label="Generation Metadata")
                preview_gallery = gr.Gallery(label="Last Preview Frames", columns=4, height="auto")

        with gr.Tab("History"):
            history_status = gr.Markdown("No generated images loaded yet.")
            history_gallery = gr.Gallery(label="Recent Output Images", columns=4, height="auto")
            refresh_history = gr.Button("Refresh History")

        refresh_models_button.click(
            fn=lambda: gr.update(
                choices=_model_choices(),
                value=(_model_choices()[0] if _model_choices() else None),
            ),
            outputs=model_name,
            queue=False,
        )

        interrupt_button.click(_interrupt_generation, outputs=status, queue=False)
        refresh_history.click(_refresh_history, outputs=[history_gallery, history_status], queue=False)
        demo.load(_refresh_history, outputs=[history_gallery, history_status], queue=False)

        generate_button.click(
            _run_generation,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                num_images,
                batch_size,
                scheduler,
                sampler,
                steps,
                guidance_scale,
                model_name,
                hires_fix,
                adetailer,
                enhance_prompt,
                img2img_enabled,
                img2img_image,
                img2img_denoise,
                stable_fast,
                reuse_seed,
                enable_multiscale,
                multiscale_intermittent,
                multiscale_factor,
                multiscale_fullres_start,
                multiscale_fullres_end,
                keep_models_loaded,
            ],
            outputs=[gallery, status, metadata, preview_gallery],
        )

    return demo


demo = _build_demo()
demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
