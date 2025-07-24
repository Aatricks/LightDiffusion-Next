import glob
import gradio as gr
import sys
import os
from PIL import Image
import numpy as np
import spaces
import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.user.pipeline import pipeline
from src.user import app_instance


# Settings persistence
SETTINGS_FILE = "webui_settings.json"

# Global variables for generation state
generation_in_progress = False


def get_preview_images():
    """Get current preview images for display during generation"""
    if app_instance.app.previewer_var.get():
        return app_instance.app.get_latest_previews()
    return []


def update_main_gallery_with_preview():
    """Update the main gallery with preview images during generation"""
    global generation_in_progress
    if generation_in_progress and app_instance.app.previewer_var.get():
        previews = app_instance.app.get_latest_previews()
        if previews:
            return previews
    return gr.update()


def check_preview_updates():
    """Check for new preview images and return them"""
    global generation_in_progress

    if generation_in_progress and app_instance.app.previewer_var.get():
        previews = app_instance.app.get_latest_previews()
        if previews:
            status = f"🎨 Generating... (Preview: {len(previews)} images)"
            return previews, status
        else:
            status = "🎨 Generating..."
            return [], status
    else:
        status = "✅ Ready to generate"
        return [], status


def get_default_settings():
    """Get default settings for the webui"""
    return {
        "prompt": "",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "num_images": 1,
        "batch_size": 1,
        "hires_fix": False,
        "adetailer": False,
        "enhance_prompt": False,
        "img2img_enabled": False,
        "stable_fast": False,
        "reuse_seed": False,
        "flux_enabled": False,
        "prio_speed": False,
        "realistic_model": False,
        "multiscale_enabled": True,
        "multiscale_intermittent": False,
        "multiscale_factor": 0.5,
        "multiscale_fullres_start": 3,
        "multiscale_fullres_end": 8,
        "keep_models_loaded": True,
        "multiscale_preset": "quality",
        "enable_preview": True,  # New setting for real-time preview
    }


def load_settings():
    """Load settings from disk"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved_settings = json.load(f)
                # Merge with defaults to handle new settings
                default_settings = get_default_settings()
                default_settings.update(saved_settings)
                return default_settings
    except Exception as e:
        print(f"Error loading settings: {e}")
    return get_default_settings()


def save_settings(settings):
    """Save settings to disk"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")


def update_generation_status():
    """Update generation status"""
    global generation_in_progress
    if generation_in_progress:
        return "🎨 Generating..."
    else:
        return "✅ Ready to generate"


def update_settings(**kwargs):
    """Update and save specific settings"""
    settings = load_settings()
    settings.update(kwargs)
    save_settings(settings)
    return settings


def load_generated_images():
    """Load generated images with given prefix from disk"""
    image_files = glob.glob("./output/**/*.png")

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
            except Exception:
                continue

    if not batch_images:
        return []
    return batch_images


def load_all_generated_images():
    """Load all generated images from output folders for history view"""
    image_files = glob.glob("./output/**/*.png", recursive=True)

    if not image_files:
        return [], "No images found in output folders."

    # Sort files by modification time in descending order (newest first)
    image_files.sort(key=os.path.getmtime, reverse=True)

    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            # Store the image with metadata
            img.info = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "folder": os.path.basename(os.path.dirname(file_path)),
                "modified": datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{img.size[0]}x{img.size[1]}",
            }
            images.append(img)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            continue

    info_text = f"Found {len(images)} images in history."
    return images, info_text


def get_image_info(evt: gr.SelectData):
    """Get detailed information about a selected image"""
    if evt.index is None:
        return "No image selected."

    try:
        images, _ = load_all_generated_images()
        if evt.index < len(images):
            img = images[evt.index]
            info = img.info
            return f"""**Image Information:**
- **Filename:** {info.get("filename", "Unknown")}
- **Folder:** {info.get("folder", "Unknown")}
- **Size:** {info.get("size", "Unknown")}
- **Modified:** {info.get("modified", "Unknown")}
- **Path:** {info.get("path", "Unknown")}"""
        else:
            return "Image index out of range."
    except Exception as e:
        return f"Error getting image info: {e}"


def delete_selected_image(evt: gr.SelectData):
    """Delete a selected image from the history"""
    if evt.index is None:
        updated_images, info = load_all_generated_images()
        return "No image selected.", updated_images, info

    try:
        images, _ = load_all_generated_images()
        if evt.index < len(images):
            img = images[evt.index]
            file_path = img.info.get("path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                # Reload images after deletion
                updated_images, info = load_all_generated_images()
                return (
                    f"✅ Successfully deleted {os.path.basename(file_path)}",
                    updated_images,
                    info,
                )
            else:
                updated_images, info = load_all_generated_images()
                return "❌ File not found or invalid path.", updated_images, info
        else:
            updated_images, info = load_all_generated_images()
            return "❌ Image index out of range.", updated_images, info
    except Exception as e:
        updated_images, info = load_all_generated_images()
        return f"❌ Error deleting image: {e}", updated_images, info


# Global variable to store the last selected image index
selected_image_index = None


def store_selected_image(evt: gr.SelectData):
    """Store the selected image index and return image info"""
    global selected_image_index
    selected_image_index = evt.index
    return get_image_info(evt)


def delete_stored_image():
    """Delete the stored selected image"""
    global selected_image_index
    if selected_image_index is None:
        updated_images, info = load_all_generated_images()
        return "No image selected.", updated_images, info

    try:
        images, _ = load_all_generated_images()
        if selected_image_index < len(images):
            img = images[selected_image_index]
            file_path = img.info.get("path")
            if file_path and os.path.exists(file_path):
                filename = os.path.basename(file_path)
                os.remove(file_path)
                selected_image_index = None  # Reset selection
                # Reload images after deletion
                updated_images, info = load_all_generated_images()
                return f"✅ Successfully deleted {filename}", updated_images, info
            else:
                updated_images, info = load_all_generated_images()
                return "❌ File not found or invalid path.", updated_images, info
        else:
            updated_images, info = load_all_generated_images()
            return "❌ Image index out of range.", updated_images, info
    except Exception as e:
        updated_images, info = load_all_generated_images()
        return f"❌ Error deleting image: {e}", updated_images, info


def clear_all_images():
    """Clear all generated images"""
    try:
        image_files = glob.glob("./output/**/*.png", recursive=True)
        deleted_count = 0

        for file_path in image_files:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        return (
            f"✅ Successfully deleted {deleted_count} images.",
            [],
            "No images found in output folders.",
        )
    except Exception as e:
        updated_images, info = load_all_generated_images()
        return f"❌ Error clearing images: {e}", updated_images, info


def refresh_image_history():
    """Refresh the image history gallery"""
    return load_all_generated_images()


@spaces.GPU
def generate_images_with_preview(
    prompt: str = "",
    negative_prompt: str = "",
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
    enable_preview: bool = True,
    progress=gr.Progress(),
):
    """Generate images with real-time preview updates in the main gallery"""
    global generation_in_progress

    try:
        generation_in_progress = True

        # Clear previous preview images
        app_instance.app.cleanup_all_previews()

        # Set preview enabled state in app_instance
        app_instance.app.previewer_var.set(enable_preview)

        # Auto-save settings
        update_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images=num_images,
            batch_size=batch_size,
            hires_fix=hires_fix,
            adetailer=adetailer,
            enhance_prompt=enhance_prompt,
            img2img_enabled=img2img_enabled,
            stable_fast=stable_fast,
            reuse_seed=reuse_seed,
            flux_enabled=flux_enabled,
            prio_speed=prio_speed,
            realistic_model=realistic_model,
            multiscale_enabled=multiscale_enabled,
            multiscale_intermittent=multiscale_intermittent,
            multiscale_factor=multiscale_factor,
            multiscale_fullres_start=multiscale_fullres_start,
            multiscale_fullres_end=multiscale_fullres_end,
            keep_models_loaded=keep_models_loaded,
            enable_preview=enable_preview,
        )

        # Set model persistence preference
        from src.Device.ModelCache import set_keep_models_loaded

        set_keep_models_loaded(keep_models_loaded)

        if img2img_enabled and img2img_image is not None:
            # Convert numpy array to PIL Image
            if isinstance(img2img_image, np.ndarray):
                img_pil = Image.fromarray(img2img_image)
                img_pil.save("temp_img2img.png")
                prompt = "temp_img2img.png"

        # Start generation in background and monitor for previews
        import threading
        import time

        final_images = []

        def run_generation():
            nonlocal final_images
            final_images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
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

        # Start generation in background thread
        gen_thread = threading.Thread(target=run_generation)
        gen_thread.start()

        # Monitor for preview updates while generation runs
        last_preview_time = 0
        while gen_thread.is_alive():
            current_previews = app_instance.app.get_latest_previews()
            if (
                current_previews
                and app_instance.app.last_preview_time > last_preview_time
            ):
                last_preview_time = app_instance.app.last_preview_time
                yield current_previews  # Yield preview images to update gallery
            time.sleep(0.5)  # Check every 0.5 seconds

        # Wait for generation to complete
        gen_thread.join()

        # Clean up temporary file if it exists
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")

        # Clear preview images and return final results
        app_instance.app.cleanup_all_previews()

        generation_in_progress = False

        # Return final images
        yield load_generated_images()

    except Exception:
        import traceback

        print(traceback.format_exc())
        generation_in_progress = False

        # Clear preview images on error
        app_instance.app.cleanup_all_previews()

        # Clean up temporary file if it exists
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")
        yield [Image.new("RGB", (512, 512), color="black")]


def get_vram_info():
    """Get VRAM usage information"""
    try:
        from src.Device.ModelCache import get_memory_info

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
        from src.Device.ModelCache import clear_model_cache

        clear_model_cache()
        return "✅ Model cache cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing cache: {e}"


def apply_multiscale_preset(preset_name):
    """Apply multiscale preset values to the UI components"""
    if preset_name == "None":
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    try:
        from src.sample.multiscale_presets import get_preset_parameters

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
    # Load saved settings
    saved_settings = load_settings()

    gr.Markdown("# LightDiffusion Web UI")
    gr.Markdown("Generate AI images using LightDiffusion")
    gr.Markdown(
        "This is the demo for LightDiffusion, the fastest diffusion backend for generating images. https://github.com/LightDiffusion/LightDiffusion-Next"
    )

    with gr.Tabs():
        # Generation Tab
        with gr.TabItem("🎨 Generate Images"):
            with gr.Row():
                with gr.Column():
                    # Input components
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value=saved_settings["prompt"],
                    )

                    # Negative prompt in accordion (hidden by default)
                    with gr.Accordion("Negative Prompt", open=False):
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Enter what you don't want to see in the image...",
                            value=saved_settings["negative_prompt"],
                            info="Describe what you want to avoid in the generated image",
                        )

                    with gr.Row():
                        width = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=saved_settings["width"],
                            step=64,
                            label="Width",
                        )
                        height = gr.Slider(
                            minimum=64,
                            maximum=2048,
                            value=saved_settings["height"],
                            step=64,
                            label="Height",
                        )

                    with gr.Row():
                        num_images = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=saved_settings["num_images"],
                            step=1,
                            label="Number of Images",
                        )
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=4,
                            value=saved_settings["batch_size"],
                            step=1,
                            label="Batch Size",
                        )

                    with gr.Row():
                        hires_fix = gr.Checkbox(
                            label="HiRes Fix", value=saved_settings["hires_fix"]
                        )
                        adetailer = gr.Checkbox(
                            label="Auto Face/Body Enhancement",
                            value=saved_settings["adetailer"],
                        )
                        enhance_prompt = gr.Checkbox(
                            label="Enhance Prompt",
                            value=saved_settings["enhance_prompt"],
                        )
                        stable_fast = gr.Checkbox(
                            label="Stable Fast Mode",
                            value=saved_settings["stable_fast"],
                        )

                    with gr.Row():
                        reuse_seed = gr.Checkbox(
                            label="Reuse Seed", value=saved_settings["reuse_seed"]
                        )
                        flux_enabled = gr.Checkbox(
                            label="Flux Mode", value=saved_settings["flux_enabled"]
                        )
                        prio_speed = gr.Checkbox(
                            label="Prioritize Speed", value=saved_settings["prio_speed"]
                        )
                        realistic_model = gr.Checkbox(
                            label="Realistic Model",
                            value=saved_settings["realistic_model"],
                        )

                    with gr.Row():
                        img2img_enabled = gr.Checkbox(
                            label="Image to Image Mode",
                            value=saved_settings["img2img_enabled"],
                        )
                        keep_models_loaded = gr.Checkbox(
                            label="Keep Models in VRAM",
                            value=saved_settings["keep_models_loaded"],
                            info="Keep models loaded for instant reuse (faster but uses more VRAM)",
                        )
                        enable_preview = gr.Checkbox(
                            label="Real-time Preview",
                            value=saved_settings["enable_preview"],
                            info="Show TAESD preview images during generation",
                        )

                    img2img_image = gr.Image(
                        label="Input Image for img2img", visible=False
                    )

                    # Multi-scale diffusion settings
                    with gr.Accordion("Multi-Scale Diffusion Settings", open=False):
                        multiscale_enabled = gr.Checkbox(
                            label="Enable Multi-Scale Diffusion",
                            value=saved_settings["multiscale_enabled"],
                        )

                        # Multi-scale preset selection
                        with gr.Row():
                            multiscale_preset = gr.Dropdown(
                                label="Multi-Scale Preset",
                                choices=[
                                    "None",
                                    "quality",
                                    "performance",
                                    "balanced",
                                    "disabled",
                                ],
                                value=saved_settings["multiscale_preset"],
                                info="Select a preset to automatically configure multi-scale settings",
                            )
                            multiscale_intermittent = gr.Checkbox(
                                label="Intermittent Full-Res",
                                value=saved_settings["multiscale_intermittent"],
                                info="Enable intermittent full-resolution rendering in low-res region",
                            )

                        with gr.Row():
                            multiscale_factor = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=saved_settings["multiscale_factor"],
                                step=0.1,
                                label="Multi-Scale Factor",
                            )
                            multiscale_fullres_start = gr.Slider(
                                minimum=0,
                                maximum=10,
                                value=saved_settings["multiscale_fullres_start"],
                                step=1,
                                label="Full-Res Start Steps",
                            )
                            multiscale_fullres_end = gr.Slider(
                                minimum=0,
                                maximum=20,
                                value=saved_settings["multiscale_fullres_end"],
                                step=1,
                                label="Full-Res End Steps",
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

                # Output gallery with preview support
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height="auto",
                        )

                        # Status display for generation progress
                        generation_status = gr.Markdown(
                            "Ready to generate", elem_id="status"
                        )

        # Image History Tab
        with gr.TabItem("📸 Image History"):
            with gr.Row():
                with gr.Column(scale=3):
                    history_gallery = gr.Gallery(
                        label="All Generated Images",
                        show_label=True,
                        elem_id="history_gallery",
                        columns=[3],
                        rows=[3],
                        object_fit="contain",
                        height="600px",
                        allow_preview=True,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Image Controls")
                    refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")

                    gr.Markdown("### Image Information")
                    image_info_display = gr.Markdown("Select an image to view details.")

                    gr.Markdown("### Actions")
                    delete_btn = gr.Button(
                        "🗑️ Delete Selected Image", variant="secondary"
                    )
                    clear_all_btn = gr.Button("⚠️ Clear All Images", variant="stop")

                    action_status = gr.Markdown("")

            history_info = gr.Markdown("")

    # Connect preview refresh button
    def update_preview_interface():
        """Update preview interface based on generation status and enable_preview setting"""
        global generation_in_progress
        previews, status = check_preview_updates()

        if generation_in_progress and app_instance.app.previewer_var.get():
            return (
                status,  # generation_status
                previews,  # preview_gallery
                gr.update(visible=True),  # preview_gallery visible
                gr.update(visible=True),  # preview_refresh_btn visible
            )
        else:
            return (
                status,  # generation_status
                [],  # preview_gallery
                gr.update(visible=False),  # preview_gallery visible
                gr.update(visible=False),  # preview_refresh_btn visible
            )

    # Create a timer for automatic preview updates
    timer = gr.Timer(value=1.0)  # 1 second interval

    def auto_refresh_preview():
        """Auto-refresh preview only during generation"""
        if generation_in_progress:
            status, _, _, _ = update_preview_interface()
            return status
        else:
            # Return current state without changes when not generating
            return "✅ Ready to generate"

    timer.tick(fn=auto_refresh_preview, outputs=[generation_status])

    # Connect generate button to pipeline
    generate_btn.click(
        fn=generate_images_with_preview,
        inputs=[
            prompt,
            negative_prompt,
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
            enable_preview,
        ],
        outputs=gallery,
    )

    # Auto-save settings when UI values change
    def save_ui_settings(
        prompt_val,
        negative_prompt_val,
        width_val,
        height_val,
        num_images_val,
        batch_size_val,
        hires_fix_val,
        adetailer_val,
        enhance_prompt_val,
        img2img_enabled_val,
        stable_fast_val,
        reuse_seed_val,
        flux_enabled_val,
        prio_speed_val,
        realistic_model_val,
        multiscale_enabled_val,
        multiscale_intermittent_val,
        multiscale_factor_val,
        multiscale_fullres_start_val,
        multiscale_fullres_end_val,
        keep_models_loaded_val,
        multiscale_preset_val,
        enable_preview_val,
    ):
        update_settings(
            prompt=prompt_val,
            negative_prompt=negative_prompt_val,
            width=width_val,
            height=height_val,
            num_images=num_images_val,
            batch_size=batch_size_val,
            hires_fix=hires_fix_val,
            adetailer=adetailer_val,
            enhance_prompt=enhance_prompt_val,
            img2img_enabled=img2img_enabled_val,
            stable_fast=stable_fast_val,
            reuse_seed=reuse_seed_val,
            flux_enabled=flux_enabled_val,
            prio_speed=prio_speed_val,
            realistic_model=realistic_model_val,
            multiscale_enabled=multiscale_enabled_val,
            multiscale_intermittent=multiscale_intermittent_val,
            multiscale_factor=multiscale_factor_val,
            multiscale_fullres_start=multiscale_fullres_start_val,
            multiscale_fullres_end=multiscale_fullres_end_val,
            keep_models_loaded=keep_models_loaded_val,
            multiscale_preset=multiscale_preset_val,
            enable_preview=enable_preview_val,
        )

    # Connect all UI components to auto-save
    ui_components = [
        prompt,
        negative_prompt,
        width,
        height,
        num_images,
        batch_size,
        hires_fix,
        adetailer,
        enhance_prompt,
        img2img_enabled,
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
        multiscale_preset,
        enable_preview,
    ]

    for component in ui_components:
        component.change(fn=save_ui_settings, inputs=ui_components, outputs=None)

    # Connect VRAM info and cache management buttons
    vram_info_btn.click(
        fn=get_vram_info,
        outputs=vram_info_display,
    )

    clear_cache_btn.click(
        fn=clear_model_cache_ui,
        outputs=cache_status_display,
    )

    # Connect image history tab components
    # Load images when the interface loads
    demo.load(fn=load_all_generated_images, outputs=[history_gallery, history_info])

    # Refresh button
    refresh_btn.click(fn=refresh_image_history, outputs=[history_gallery, history_info])

    # Image selection for info display and storing selection
    history_gallery.select(fn=store_selected_image, outputs=image_info_display)

    # Delete selected image
    delete_btn.click(
        fn=delete_stored_image, outputs=[action_status, history_gallery, history_info]
    )

    # Clear all images
    clear_all_btn.click(
        fn=clear_all_images, outputs=[action_status, history_gallery, history_info]
    )


def is_huggingface_space():
    return "SPACE_ID" in os.environ


def is_docker_environment():
    return "GRADIO_SERVER_PORT" in os.environ and "GRADIO_SERVER_NAME" in os.environ


# For local testing
if __name__ == "__main__":
    # Ensure preview directory exists
    os.makedirs("./output/preview", exist_ok=True)

    if is_huggingface_space():
        demo.launch(
            debug=False,
            server_name="0.0.0.0",
            server_port=7860,  # Standard HF Spaces port
        )
    elif is_docker_environment():
        # Docker environment - use environment variables
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
        demo.launch(
            debug=False,
            server_name=server_name,
            server_port=server_port,
        )
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=8000,
            auth=None,
            share=True,  # Only enable sharing locally
            debug=True,
        )
