import glob
import gradio as gr
import sys
import os
from PIL import Image
import numpy as np
import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.user.pipeline import pipeline
import torch


def load_generated_images():
    """Load generated images with given prefix from disk"""
    image_files = glob.glob("./_internal/output/**/*.png")

    # If there are no image files, return
    if not image_files:
        return []

    # Sort files by modification time in descending order
    image_files.sort(key=os.path.getmtime, reverse=True)

    # Get most recent timestamp
    latest_time = os.path.getmtime(image_files[0])

    # Get all images from same batch (within 1 second of most recent)
    batch_images = []
    for file in image_files:
        if abs(os.path.getmtime(file) - latest_time) < 1.0:
            try:
                img = Image.open(file)
                batch_images.append(img)
            except:
                continue

    if not batch_images:
        return []
    return batch_images


@spaces.GPU
def generate_images(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    batch_size: int = 1,
    hires_fix: bool = False,
    adetailer: bool = False,
    enhance_prompt: bool = False,
    img2img_enabled: bool = False,
    img2img_image: str = None,
    stable_fast: bool = False,
    reuse_seed: bool = False,
    flux_enabled: bool = False,
    prio_speed: bool = False,
    realistic_model: bool = False,
    multiscale_enabled: bool = True,
    multiscale_intermittent: bool = False,
    multiscale_factor: float = 0.5,
    multiscale_fullres_start: int = 3,
    multiscale_fullres_end: int = 8,
    keep_models_loaded: bool = True,
    progress=gr.Progress(),
):
    """Generate images using the LightDiffusion pipeline"""
    try:
        # Set model persistence preference
        from modules.Device.ModelCache import set_keep_models_loaded

        set_keep_models_loaded(keep_models_loaded)

        if img2img_enabled and img2img_image is not None:
            # Convert numpy array to PIL Image
            if isinstance(img2img_image, np.ndarray):
                img_pil = Image.fromarray(img2img_image)
                img_pil.save("temp_img2img.png")
                prompt = "temp_img2img.png"

        # Run pipeline and capture saved images
        with torch.inference_mode():
            pipeline(
                prompt=prompt,
                w=width,
                h=height,
                number=num_images,
                batch=batch_size,
                hires_fix=hires_fix,
                adetailer=adetailer,
                enhance_prompt=enhance_prompt,
                img2img=img2img_enabled,
                stable_fast=stable_fast,
                reuse_seed=reuse_seed,
                flux_enabled=flux_enabled,
                prio_speed=prio_speed,
                autohdr=True,
                realistic_model=realistic_model,
                enable_multiscale=multiscale_enabled,
                multiscale_intermittent_fullres=multiscale_intermittent,
                multiscale_factor=multiscale_factor,
                multiscale_fullres_start=multiscale_fullres_start,
                multiscale_fullres_end=multiscale_fullres_end,
            )

        # Clean up temporary file if it exists
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")

        return load_generated_images()

    except Exception:
        import traceback

        print(traceback.format_exc())
        # Clean up temporary file if it exists
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")
        return [Image.new("RGB", (512, 512), color="black")]


def get_vram_info():
    """Get VRAM usage information"""
    try:
        from modules.Device.ModelCache import get_memory_info

        info = get_memory_info()
        return f"""
**VRAM Usage:**
- Total: {info["total_vram"]:.1f} GB
- Used: {info["used_vram"]:.1f} GB
- Free: {info["free_vram"]:.1f} GB
- Keep Models Loaded: {info["keep_loaded"]}
- Has Cached Checkpoint: {info["has_cached_checkpoint"]}
"""
    except Exception as e:
        return f"Error getting VRAM info: {e}"


def clear_model_cache_ui():
    """Clear model cache from UI"""
    try:
        from modules.Device.ModelCache import clear_model_cache

        clear_model_cache()
        return "✅ Model cache cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing cache: {e}"


def apply_multiscale_preset(preset_name):
    """Apply multiscale preset values to the UI components"""
    if preset_name == "None":
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    try:
        from modules.sample.multiscale_presets import get_preset_parameters

        params = get_preset_parameters(preset_name)

        return (
            gr.update(value=params["enable_multiscale"]),
            gr.update(value=params["multiscale_factor"]),
            gr.update(value=params["multiscale_fullres_start"]),
            gr.update(value=params["multiscale_fullres_end"]),
            gr.update(value=params["multiscale_intermittent_fullres"]),
        )
    except Exception as e:
        print(f"Error applying preset {preset_name}: {e}")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


# Create Gradio interface
with gr.Blocks(title="LightDiffusion Web UI") as demo:
    gr.Markdown("# LightDiffusion Web UI")
    gr.Markdown("Generate AI images using LightDiffusion")
    gr.Markdown(
        "This is the demo for LightDiffusion, the fastest diffusion backend for generating images. https://github.com/LightDiffusion/LightDiffusion-Next"
    )

    with gr.Row():
        with gr.Column():
            # Input components
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")

            with gr.Row():
                width = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64, label="Width"
                )
                height = gr.Slider(
                    minimum=64, maximum=2048, value=512, step=64, label="Height"
                )

            with gr.Row():
                num_images = gr.Slider(
                    minimum=1, maximum=10, value=1, step=1, label="Number of Images"
                )
                batch_size = gr.Slider(
                    minimum=1, maximum=4, value=1, step=1, label="Batch Size"
                )

            with gr.Row():
                hires_fix = gr.Checkbox(label="HiRes Fix")
                adetailer = gr.Checkbox(label="Auto Face/Body Enhancement")
                enhance_prompt = gr.Checkbox(label="Enhance Prompt")
                stable_fast = gr.Checkbox(label="Stable Fast Mode")

            with gr.Row():
                reuse_seed = gr.Checkbox(label="Reuse Seed")
                flux_enabled = gr.Checkbox(label="Flux Mode")
                prio_speed = gr.Checkbox(label="Prioritize Speed")
                realistic_model = gr.Checkbox(label="Realistic Model")

            with gr.Row():
                multiscale_enabled = gr.Checkbox(
                    label="Multi-Scale Diffusion", value=True
                )
                img2img_enabled = gr.Checkbox(label="Image to Image Mode")
                keep_models_loaded = gr.Checkbox(
                    label="Keep Models in VRAM",
                    value=True,
                    info="Keep models loaded for instant reuse (faster but uses more VRAM)",
                )

            img2img_image = gr.Image(label="Input Image for img2img", visible=False)

            # Multi-scale preset selection
            with gr.Row():
                multiscale_preset = gr.Dropdown(
                    label="Multi-Scale Preset",
                    choices=["None", "quality", "performance", "balanced", "disabled"],
                    value="None",
                    info="Select a preset to automatically configure multi-scale settings",
                )
                multiscale_intermittent = gr.Checkbox(
                    label="Intermittent Full-Res",
                    value=False,
                    info="Enable intermittent full-resolution rendering in low-res region",
                )

            with gr.Row():
                multiscale_factor = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Multi-Scale Factor",
                )
                multiscale_fullres_start = gr.Slider(
                    minimum=0, maximum=10, value=3, step=1, label="Full-Res Start Steps"
                )
                multiscale_fullres_end = gr.Slider(
                    minimum=0, maximum=20, value=8, step=1, label="Full-Res End Steps"
                )

            # Make input image visible only when img2img is enabled
            img2img_enabled.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[img2img_enabled],
                outputs=[img2img_image],
            )

            # Handle preset changes
            multiscale_preset.change(
                fn=apply_multiscale_preset,
                inputs=[multiscale_preset],
                outputs=[
                    multiscale_enabled,
                    multiscale_factor,
                    multiscale_fullres_start,
                    multiscale_fullres_end,
                    multiscale_intermittent,
                ],
            )

            generate_btn = gr.Button("Generate")

            # Model Cache Management
            with gr.Accordion("Model Cache Management", open=False):
                with gr.Row():
                    vram_info_btn = gr.Button("🔍 Check VRAM Usage")
                    clear_cache_btn = gr.Button("🗑️ Clear Model Cache")
                vram_info_display = gr.Markdown("")
                cache_status_display = gr.Markdown("")

        # Output gallery
        gallery = gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=[2],
            rows=[2],
            object_fit="contain",
            height="auto",
        )

    # Connect generate button to pipeline
    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt,
            width,
            height,
            num_images,
            batch_size,
            hires_fix,
            adetailer,
            enhance_prompt,
            img2img_enabled,
            img2img_image,
            stable_fast,
            reuse_seed,
            flux_enabled,
            prio_speed,
            realistic_model,
            multiscale_enabled,
            multiscale_intermittent,
            multiscale_factor,
            multiscale_fullres_start,
            multiscale_fullres_end,
            keep_models_loaded,
        ],
        outputs=gallery,
    )

    # Connect VRAM info and cache management buttons
    vram_info_btn.click(
        fn=get_vram_info,
        outputs=vram_info_display,
    )

    clear_cache_btn.click(
        fn=clear_model_cache_ui,
        outputs=cache_status_display,
    )


def is_huggingface_space():
    return "SPACE_ID" in os.environ


# For local testing
if __name__ == "__main__":
    if is_huggingface_space():
        demo.launch(
            debug=False,
            server_name="0.0.0.0",
            server_port=7860,  # Standard HF Spaces port
        )
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            auth=None,
            share=True,  # Only enable sharing locally
            debug=True,
        )
