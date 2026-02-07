"""Page rendering helpers for LightDiffusion-Next.

This module contains page-level render functions that can be imported
by the main app entrypoint. Breaking pages out makes the main file
shorter and easier to navigate.
"""
import hashlib
import os

import streamlit as st
from PIL import Image

from src.Device.ModelCache import clear_model_cache, get_memory_info
from ui import settings as ui_settings
from ui.generation import generate_images, prepare_generation, stop_generation
from ui.helpers import compute_display_size, render_responsive_image
from ui.history import (
    clear_history,
    delete_history_entry,
    load_history,
    scan_output_folders,
    search_history,
    get_available_model_types,
    get_history_date_range,
)


def _apply_flux2_optimal_settings(settings: dict) -> None:
    """Apply optimal settings for Flux2 Klein 4B model."""
    # Sampling settings optimized for Flux2 Klein
    settings["cfg_scale"] = 1.0
    settings["sampler"] = "euler"
    settings["scheduler"] = "simple"
    settings["steps"] = 4
    
    # Flux doesn't use negative prompts (like ComfyUI's ConditioningZeroOut)
    # Save the current negative prompt before clearing so it can be restored
    if settings.get("negative_prompt"):
        st.session_state["saved_negative_prompt"] = settings["negative_prompt"]
    settings["negative_prompt"] = ""
    
    # Disable features incompatible with DiT architecture
    settings["multiscale_preset"] = "disabled"
    settings["multiscale_custom"] = False
    settings["deepcache_enabled"] = False
    settings["tome_enabled"] = False
    
    # Disable enhancements that don't work well with Flux2
    settings["hiresfix"] = False
    settings["adetailer"] = False
    settings["stable_fast"] = False
    
    # Flux doesn't support SDXL refiner
    settings["refiner_model_path"] = ""
    
    # Set default resolution if not already at Flux2 size
    if settings.get("width", 512) < 1024 or settings.get("height", 512) < 1024:
        settings["width"] = 1024
        settings["height"] = 1024
        # Sync to widget keys if they exist in session state
        if "width_input" in st.session_state:
            st.session_state.width_input = 1024
        if "height_input" in st.session_state:
            st.session_state.height_input = 1024
        if "preset_selectbox" in st.session_state:
            st.session_state.preset_selectbox = "1024x1024 (Flux2 1:1)"


def _revert_flux2_optimal_settings(settings: dict) -> None:
    """Revert Flux2-specific settings to standard defaults when switching away."""
    settings["cfg_scale"] = 7.0
    settings["sampler"] = "dpmpp_sde_cfgpp"
    settings["scheduler"] = "ays"
    settings["steps"] = 20
    
    # Re-enable standard features
    settings["multiscale_preset"] = "balanced"
    
    # Reset resolution to SD1.5 standard if it was forced to 1024
    if settings.get("width") == 1024 and settings.get("height") == 1024:
        settings["width"] = 512
        settings["height"] = 512
        if "width_input" in st.session_state:
            st.session_state.width_input = 512
        if "height_input" in st.session_state:
            st.session_state.height_input = 512
        if "preset_selectbox" in st.session_state:
            st.session_state.preset_selectbox = "512x512 (SD1.5)"


def _apply_sd15_optimal_settings(settings: dict) -> None:
    """Apply optimal settings for SD1.5 models."""
    # Sampling settings optimized for SD1.5
    settings["cfg_scale"] = 7.0
    settings["sampler"] = "dpmpp_sde_cfgpp"
    settings["scheduler"] = "ays"
    settings["steps"] = 20
    
    # Restore negative prompt if it was saved (when switching from Flux2)
    if st.session_state.get("saved_negative_prompt") and not settings.get("negative_prompt"):
        settings["negative_prompt"] = st.session_state["saved_negative_prompt"]
    
    # Multi-scale and acceleration available for SD1.5
    settings["multiscale_preset"] = "balanced"
    
    # SD1.5 doesn't support SDXL refiner
    settings["refiner_model_path"] = ""
    
    # Set default resolution for SD1.5
    if settings.get("width", 512) >= 1024 or settings.get("height", 512) >= 1024:
        settings["width"] = 512
        settings["height"] = 512
        if "width_input" in st.session_state:
            st.session_state.width_input = 512
        if "height_input" in st.session_state:
            st.session_state.height_input = 512
        if "preset_selectbox" in st.session_state:
            st.session_state.preset_selectbox = "512x512 (SD1.5)"


def _apply_sdxl_optimal_settings(settings: dict) -> None:
    """Apply optimal settings for SDXL models."""
    # Sampling settings optimized for SDXL
    settings["cfg_scale"] = 7.0
    settings["sampler"] = "euler"
    settings["scheduler"] = "ays"
    settings["steps"] = 25
    
    # Restore negative prompt if it was saved (when switching from Flux2)
    if st.session_state.get("saved_negative_prompt") and not settings.get("negative_prompt"):
        settings["negative_prompt"] = st.session_state["saved_negative_prompt"]
    
    # Multi-scale and acceleration available for SDXL
    settings["multiscale_preset"] = "balanced"
    
    # Auto-configure SDXL refiner with switch at step 20
    settings["refiner_switch_step"] = 20
    
    # Try to find an SDXL refiner model in the checkpoints
    try:
        from src.Core.Models.ModelFactory import list_available_models
        available_map = list_available_models(return_mapping=True)
        for display_name, full_path in available_map:
            name_lower = display_name.lower()
            if "refiner" in name_lower or ("sdxl" in name_lower and "refiner" in name_lower):
                settings["refiner_model_path"] = full_path
                break
    except Exception:
        pass
    
    # Set default resolution for SDXL (native 1024x1024)
    if settings.get("width", 512) < 1024 or settings.get("height", 512) < 1024:
        settings["width"] = 1024
        settings["height"] = 1024
        if "width_input" in st.session_state:
            st.session_state.width_input = 1024
        if "height_input" in st.session_state:
            st.session_state.height_input = 1024
        if "preset_selectbox" in st.session_state:
            st.session_state.preset_selectbox = "1024x1024 (SDXL 1:1)"


def _get_detected_model_type(model_path: str) -> str:
    """Detect model type from model path.
    
    Returns:
        'Flux2Klein', 'SDXL', or 'SD15'
    """
    if model_path == "__FLUX2_KLEIN__":
        return "Flux2Klein"
    
    if not model_path:
        return "SD15"  # Default to SD1.5
    
    try:
        from src.Core.Models.ModelFactory import detect_model_type
        return detect_model_type(model_path)
    except Exception:
        return "SD15"


def render_generate_page():
    """Render the main generation page."""
    settings = st.session_state.settings
    controls_disabled = st.session_state.is_generating

    # Define resolution presets mapping
    PRESETS = {
        "512x512 (SD1.5)": (512, 512),
        "768x768 (SD1.5)": (768, 768),
        "512x768 (SD1.5 Portrait)": (512, 768),
        "768x512 (SD1.5 Landscape)": (768, 512),
        "1024x1024 (SDXL 1:1)": (1024, 1024),
        "1152x896 (SDXL 4:3)": (1152, 896),
        "896x1152 (SDXL 3:4)": (896, 1152),
        "1216x832 (SDXL 3:2)": (1216, 832),
        "832x1216 (SDXL 2:3)": (832, 1216),
        "1344x768 (SDXL 16:9)": (1344, 768),
        "768x1344 (SDXL 9:16)": (768, 1344),
        "1024x1024 (Flux2 1:1)": (1024, 1024),
        "1280x768 (Flux2 16:9)": (1280, 768),
        "768x1280 (Flux2 9:16)": (768, 1280),
        "1024x768 (Flux2 4:3)": (1024, 768),
        "768x1024 (Flux2 3:4)": (768, 1024),
    }

    def on_preset_change():
        p = st.session_state.preset_selectbox
        if p in PRESETS:
            w, h = PRESETS[p]
            st.session_state.settings["width"] = w
            st.session_state.settings["height"] = h
            # Sync widget states to avoid one-frame lag
            st.session_state.width_input = w
            st.session_state.height_input = h

    def on_dim_change():
        # Reset preset to Custom on manual dimension change
        st.session_state.preset_selectbox = "Custom"
        # Immediate sync to settings
        st.session_state.settings["width"] = st.session_state.width_input
        st.session_state.settings["height"] = st.session_state.height_input

    # Ensure widget session state is initialized
    if "width_input" not in st.session_state:
        st.session_state.width_input = settings.get("width", 512)
    if "height_input" not in st.session_state:
        st.session_state.height_input = settings.get("height", 512)
    if "preset_selectbox" not in st.session_state:
        st.session_state.preset_selectbox = "Custom"

    with st.sidebar:
        st.markdown(
            '<a href="https://github.com/Aatricks/LightDiffusion-Next" target="_blank" style="text-decoration:none;color:inherit;"><h2 style="margin:0 0 8px 0;">LightDiffusion</h2></a>',
            unsafe_allow_html=True,
        )
        st.header("⚙️ Settings")
        if controls_disabled:
            st.warning("⚠️ Generation in progress — settings are locked until the current job finishes or you press Stop.")

        with st.expander("🎯 Model Selection", expanded=True):
            # Allow the user to pick a model type or file
            try:
                from src.Core.Models.ModelFactory import list_available_models, _find_flux2_components

                available_map = list_available_models(return_mapping=True)
                # Check if Flux2 Klein components exist
                flux2_diff, flux2_te, flux2_vae = _find_flux2_components()
                flux2_available = flux2_diff is not None
            except Exception:
                available_map = []
                flux2_available = False

            # available_map is list of (display_name, full_path)
            display_names = [d for d, _ in available_map]
            mapping = {d: p for d, p in available_map}

            # Build model options with Flux2 Klein if available
            model_options = ["Auto (use default)"]
            if flux2_available:
                model_options.append("Flux2 Klein (auto-detected)")
                mapping["Flux2 Klein (auto-detected)"] = "__FLUX2_KLEIN__"
            model_options.extend(display_names)

            # For current selection, show the basename so it matches the dropdown
            current_full = settings.get("model_path", "")
            if current_full == "__FLUX2_KLEIN__":
                current = "Flux2 Klein (auto-detected)"
            elif current_full:
                current = os.path.basename(current_full)
            else:
                current = "Auto (use default)"
            try:
                idx = model_options.index(current)
            except Exception:
                idx = 0

            sel = st.selectbox("Model", options=model_options, index=idx, disabled=controls_disabled)
            
            # Get previous model type for change detection
            prev_model_path = settings.get("model_path", "")
            prev_model_type = _get_detected_model_type(prev_model_path)
            
            if sel == "Auto (use default)":
                new_model_path = ""
                new_model_type = "SD15"
            elif sel == "Flux2 Klein (auto-detected)":
                new_model_path = "__FLUX2_KLEIN__"
                new_model_type = "Flux2Klein"
            else:
                new_model_path = mapping.get(sel, sel)
                new_model_type = _get_detected_model_type(new_model_path)
            
            # Apply appropriate settings when model type changes
            if new_model_path != prev_model_path:
                settings["model_path"] = new_model_path
                
                if new_model_type != prev_model_type:
                    if new_model_type == "Flux2Klein":
                        _apply_flux2_optimal_settings(settings)
                        st.info("Flux2 Klein selected: Auto-applied optimal settings (CFG 1.0, 4 steps, 1024px)")
                    elif new_model_type == "SDXL":
                        _apply_sdxl_optimal_settings(settings)
                        st.info("SDXL model selected: Auto-applied optimal settings (1024px, CFG 7.0)")
                    elif new_model_type == "SD15":
                        _apply_sd15_optimal_settings(settings)
                        st.info("SD1.5 model selected: Auto-applied optimal settings (512px, CFG 7.0)")
            
            # Store current model type in session state for other parts of the UI
            st.session_state["current_model_type"] = new_model_type

            settings["img2img_mode"] = st.checkbox("Img2Img Mode", value=settings["img2img_mode"], disabled=controls_disabled)

            if settings["img2img_mode"]:
                if controls_disabled:
                    st.info("Image upload is disabled while generation is running.")
                    if settings.get("input_image_path") and os.path.exists(settings.get("input_image_path")):
                        try:
                            st.image(settings.get("input_image_path"), caption="Current Input Image", width='stretch')
                        except Exception:
                            pass
                else:
                    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
                    if uploaded_file:
                        img_path = f"./output/uploaded_{uploaded_file.name}"
                        with open(img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        settings["input_image_path"] = img_path
                        st.image(uploaded_file, caption="Input Image", width='stretch')
                
                # Denoise slider for img2img (always shown when img2img_mode is on)
                settings["img2img_denoise"] = st.slider(
                    "Denoising Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=settings.get("img2img_denoise", 0.75),
                    step=0.05,
                    disabled=controls_disabled,
                    help="0 = keep original image, 1 = full regeneration (like txt2img)"
                )

        with st.expander("🎛️ ControlNet", expanded=False):
            settings["controlnet_enabled"] = st.checkbox(
                "Enable ControlNet",
                value=settings.get("controlnet_enabled", False),
                disabled=controls_disabled,
                help="Use edge detection to preserve image structure while changing content. Requires an input image."
            )

            if settings["controlnet_enabled"]:
                # ControlNet requires an input image – share the same upload
                # widget from img2img or allow a separate upload
                if not settings.get("img2img_mode"):
                    if controls_disabled:
                        st.info("Image upload is disabled while generation is running.")
                        if settings.get("input_image_path") and os.path.exists(settings.get("input_image_path")):
                            try:
                                st.image(settings.get("input_image_path"), caption="ControlNet Input", width='stretch')
                            except Exception:
                                pass
                    else:
                        cn_uploaded = st.file_uploader("Upload Control Image", type=["png", "jpg", "jpeg"], key="cn_upload")
                        if cn_uploaded:
                            cn_path = f"./output/uploaded_{cn_uploaded.name}"
                            with open(cn_path, "wb") as f:
                                f.write(cn_uploaded.getbuffer())
                            settings["input_image_path"] = cn_path
                            st.image(cn_uploaded, caption="Control Input", width='stretch')
                else:
                    st.info("Using the same input image from Img2Img mode.")

                controlnet_types = {
                    "canny": "Canny Edge Detection",
                    "none": "None (use raw image)",
                }
                current_cn_type = settings.get("controlnet_type", "canny")
                settings["controlnet_type"] = st.selectbox(
                    "Preprocessor",
                    options=list(controlnet_types.keys()),
                    format_func=lambda x: controlnet_types[x],
                    index=list(controlnet_types.keys()).index(current_cn_type) if current_cn_type in controlnet_types else 0,
                    disabled=controls_disabled,
                    help="Edge detection method used to extract structure from the input image."
                )

                settings["controlnet_strength"] = st.slider(
                    "Control Strength",
                    min_value=0.0,
                    max_value=2.0,
                    value=settings.get("controlnet_strength", 1.0),
                    step=0.05,
                    disabled=controls_disabled,
                    help="How strongly the structure is preserved. Higher = more faithful to input, lower = more creative freedom."
                )

        with st.expander("SDXL Refiner", expanded=False):
            # Get current model type to determine if refiner should be enabled
            current_model_type = st.session_state.get("current_model_type", "SD15")
            is_sdxl_model = current_model_type == "SDXL"
            
            if not is_sdxl_model:
                # Show info message when refiner is unavailable
                if current_model_type == "Flux2Klein":
                    st.info("Refiner is not available for Flux2 models. The refiner is an SDXL-specific feature.")
                else:
                    st.info("Refiner is not available for SD1.5 models. Select an SDXL model to use the refiner.")
                
                # Ensure refiner is disabled in settings
                settings["refiner_model_path"] = ""
                
                # Show disabled controls for visual consistency
                st.selectbox("Refiner Model", options=["None"], index=0, disabled=True)
                st.slider(
                    "Refiner Switch Step",
                    min_value=0,
                    max_value=settings.get("steps", 150),
                    value=settings.get("refiner_switch_step", 20),
                    disabled=True
                )
            else:
                # SDXL model - refiner is available
                ref_model_options = ["None"] + display_names
                
                current_ref = settings.get("refiner_model_path", "")
                if current_ref:
                    current_ref_name = os.path.basename(current_ref)
                else:
                    current_ref_name = "None"
                
                try:
                    ref_idx = ref_model_options.index(current_ref_name)
                except Exception:
                    ref_idx = 0
                
                ref_sel = st.selectbox("Refiner Model", options=ref_model_options, index=ref_idx, disabled=controls_disabled)
                if ref_sel == "None":
                    settings["refiner_model_path"] = ""
                else:
                    settings["refiner_model_path"] = mapping.get(ref_sel, ref_sel)
                
                settings["refiner_switch_step"] = st.slider(
                    "Refiner Switch Step",
                    min_value=0,
                    max_value=settings.get("steps", 150),
                    value=settings.get("refiner_switch_step", 20),
                    disabled=controls_disabled or not settings["refiner_model_path"]
                )

        with st.expander("Prompt & Text", expanded=True):
            prompt = st.text_area("Prompt", value=settings["prompt"], height=100, key="prompt_input", disabled=controls_disabled)
            settings["prompt"] = prompt
            
            # Disable negative prompt for Flux2 models (they don't use it)
            is_flux2 = st.session_state.get("current_model_type") == "Flux2Klein"
            if is_flux2:
                st.info("Flux models do not use negative prompts.")
                st.text_area("Negative Prompt", value="", height=80, key="negative_prompt_input", disabled=True)
            else:
                negative_prompt = st.text_area("Negative Prompt", value=settings["negative_prompt"], height=80, key="negative_prompt_input", disabled=controls_disabled)
                settings["negative_prompt"] = negative_prompt

        with st.expander("📐 Dimensions & Batch", expanded=True):
            preset_options = ["Custom"] + list(PRESETS.keys())
            # Insert separators for readability
            preset_options.insert(5, "--- SDXL (1.0 MP) ---")
            preset_options.insert(13, "--- Flux2 Klein ---")
            
            st.selectbox(
                "Presets", 
                options=preset_options, 
                key="preset_selectbox", 
                on_change=on_preset_change, 
                disabled=controls_disabled
            )

            col1, col2 = st.columns(2)
            with col1:
                settings["width"] = st.number_input(
                    "Width", 
                    min_value=64, 
                    max_value=2048, 
                    value=settings.get("width", 1024),
                    key="width_input", 
                    on_change=on_dim_change, 
                    step=64, 
                    disabled=controls_disabled
                )
            with col2:
                settings["height"] = st.number_input(
                    "Height", 
                    min_value=64, 
                    max_value=2048, 
                    value=settings.get("height", 1024),
                    key="height_input", 
                    on_change=on_dim_change, 
                    step=64, 
                    disabled=controls_disabled
                )

            settings["num_images"] = st.number_input("Number of Images", min_value=1, max_value=1000, value=settings["num_images"], key="num_images_input", disabled=controls_disabled)
            settings["batch_size"] = st.number_input("Batch Size (images per batch)", min_value=1, max_value=10, value=settings.get("batch_size", 1), key="batch_size_input", disabled=controls_disabled, help="Number of images processed together per internal batch.")


        with st.expander("⚡ Sampling & Scheduling"):
            st.markdown("**Scheduler & Sampler Settings**")

            scheduler_options = {
                "normal": "Normal - Standard linear schedule",
                "karras": "Karras - Improved noise schedule",
                "simple": "Simple - Simplified schedule",
                "beta": "Beta - Alternative schedule",
                "ays": "AYS - Align Your Steps (SD1.5 auto)",
                "ays_sd15": "AYS SD1.5 - Optimized for SD1.5",
                "ays_sdxl": "AYS SDXL - Optimized for SDXL"
            }
            current_scheduler = settings.get("scheduler", "ays")
            settings["scheduler"] = st.selectbox(
                "Scheduler",
                options=list(scheduler_options.keys()),
                format_func=lambda x: scheduler_options[x],
                index=list(scheduler_options.keys()).index(current_scheduler) if current_scheduler in scheduler_options else 0,
                disabled=controls_disabled,
                help="AYS schedulers provide 30-50% speedup by using optimal noise schedules"
            )

            sampler_options = {
                "euler": "Euler - Fast and stable",
                "euler_ancestral": "Euler Ancestral - More variation",
                "euler_cfgpp": "Euler CFG++ - Fast with CFG optimization",
                "euler_ancestral_cfgpp": "Euler Ancestral CFG++ - Variation with CFG++",
                "dpmpp_2m_cfgpp": "DPM++ 2M CFG++ - Balanced with CFG optimization",
                "dpmpp_sde_cfgpp": "DPM++ SDE CFG++ - High quality with CFG++"
            }
            current_sampler = settings.get("sampler", "dpmpp_sde_cfgpp")
            settings["sampler"] = st.selectbox(
                "Sampler",
                options=list(sampler_options.keys()),
                format_func=lambda x: sampler_options[x],
                index=list(sampler_options.keys()).index(current_sampler) if current_sampler in sampler_options else 0,
                disabled=controls_disabled,
                help="CFG++ samplers use dynamic guidance rescaling for improved quality"
            )

            recommended_steps = 20
            if settings.get("scheduler", "normal").startswith("ays"):
                recommended_steps = 10
                st.info("💡 AYS scheduler recommended: 10 steps (equivalent to 20 normal steps)")

            settings["steps"] = st.slider(
                "Sampling Steps",
                min_value=1,
                max_value=150,
                value=settings.get("steps", recommended_steps),
                step=1,
                disabled=controls_disabled,
                help="Number of denoising steps. AYS: 10 steps, Normal: 20 steps typical"
            )

            # CFG Scale slider
            is_flux2 = settings.get("model_path") == "__FLUX2_KLEIN__"
            cfg_help = "Classifier-Free Guidance scale. Flux2: use 1.0, SD1.5/SDXL: use 7.0-8.0"
            if is_flux2:
                cfg_help = "⚡ Flux2 Klein works best with CFG=1.0 (no guidance needed)"
            
            settings["cfg_scale"] = st.slider(
                "CFG Scale",
                min_value=1.0,
                max_value=20.0,
                value=settings.get("cfg_scale", 1.0 if is_flux2 else 7.0),
                step=0.5,
                disabled=controls_disabled,
                help=cfg_help
            )

            st.markdown("**Optimization Caching**")
            settings["prompt_cache_enabled"] = st.checkbox(
                "Enable Prompt Cache",
                value=settings.get("prompt_cache_enabled", True),
                disabled=controls_disabled,
                help="Cache CLIP text embeddings for 5-15% speedup on repeated prompts"
            )

            if settings["prompt_cache_enabled"]:
                try:
                    from src.Utilities import prompt_cache
                    stats = prompt_cache.get_cache_stats()
                    if stats and stats.get('total_requests', 0) > 0:
                        st.text(f"Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']:.1%})")
                        if st.button("Clear Prompt Cache", disabled=controls_disabled):
                            prompt_cache.clear_prompt_cache()
                            st.success("Prompt cache cleared!")
                except Exception:
                    pass

        with st.expander("✨ Enhancements"):
            settings["hiresfix"] = st.checkbox("HiRes Fix", value=settings["hiresfix"], disabled=controls_disabled)
            settings["adetailer"] = st.checkbox("ADetailer", value=settings["adetailer"], disabled=controls_disabled)
            settings["enhance_prompt"] = st.checkbox("Enhance Prompt", value=settings["enhance_prompt"], disabled=controls_disabled)
            settings["stable_fast"] = st.checkbox("Stable Fast", value=settings["stable_fast"], disabled=controls_disabled)

        with st.expander("🔧 Advanced"):
            settings["reuse_seed"] = st.checkbox("Reuse Seed", value=settings["reuse_seed"], disabled=controls_disabled)
            settings["enable_preview"] = st.checkbox("Live Preview", value=settings["enable_preview"], disabled=controls_disabled)

        with st.expander("🔬 Multi-scale"):
            settings["enable_multiscale"] = st.checkbox(
                "Enable Multi-scale", 
                value=settings.get("enable_multiscale", False), 
                help="Start generation at lower resolution and upscale during sampling for speedup",
                disabled=controls_disabled
            )
            
            preset_options = {
                "quality": "Quality - Best image quality with intermittent full-res",
                "balanced": "Balanced - Good quality and performance",
                "performance": "Performance - Maximum speed with aggressive downscaling",
                "disabled": "Disabled - Full resolution throughout",
                "custom": "Custom - Configure all settings manually"
            }
            if settings.get("multiscale_custom", False):
                current_preset = "custom"
            else:
                current_preset = settings.get("multiscale_preset", "balanced")

            selected_preset = st.selectbox("Preset", options=list(preset_options.keys()), format_func=lambda x: preset_options[x], index=list(preset_options.keys()).index(current_preset), disabled=controls_disabled)
            if selected_preset == "custom":
                settings["multiscale_custom"] = True
                settings["multiscale_factor"] = st.slider("Scale Factor", min_value=0.1, max_value=1.0, value=settings.get("multiscale_factor", 0.5), step=0.05, help="Scale factor for intermediate steps", disabled=controls_disabled)
                settings["multiscale_fullres_start"] = st.number_input("Full-res Start Steps", min_value=0, max_value=20, value=settings.get("multiscale_fullres_start", 3), help="Number of first steps at full resolution", disabled=controls_disabled)
                settings["multiscale_fullres_end"] = st.number_input("Full-res End Steps", min_value=0, max_value=20, value=settings.get("multiscale_fullres_end", 8), help="Number of last steps at full resolution", disabled=controls_disabled)
                settings["multiscale_intermittent_fullres"] = st.checkbox("Intermittent Full-res", value=settings.get("multiscale_intermittent_fullres", False), help="Enable intermittent full-res rendering in low-res region", disabled=controls_disabled)
            else:
                settings["multiscale_custom"] = False
                settings["multiscale_preset"] = selected_preset

        with st.expander("⚡ DeepCache Acceleration"):
            st.markdown("**DeepCache** speeds up generation by reusing U-Net features (2-3x faster with minimal quality loss)")
            settings["deepcache_enabled"] = st.checkbox("Enable DeepCache", value=settings.get("deepcache_enabled", False), help="Enable DeepCache acceleration for faster generation", disabled=controls_disabled)

            if settings["deepcache_enabled"]:
                settings["deepcache_interval"] = st.slider("Cache Interval", min_value=1, max_value=10, value=settings.get("deepcache_interval", 3), help="Steps between cache updates (higher = faster but lower quality)", disabled=controls_disabled)
                settings["deepcache_depth"] = st.slider("Cache Depth", min_value=0, max_value=12, value=settings.get("deepcache_depth", 2), help="U-Net depth for caching (higher = more aggressive)", disabled=controls_disabled)

                col1, col2 = st.columns(2)
                with col1:
                    settings["deepcache_start_step"] = st.number_input("Start Step", min_value=0, max_value=1000, value=settings.get("deepcache_start_step", 0), help="Start applying DeepCache at this step", disabled=controls_disabled)
                with col2:
                    settings["deepcache_end_step"] = st.number_input("End Step", min_value=0, max_value=1000, value=settings.get("deepcache_end_step", 1000), help="Stop applying DeepCache at this step", disabled=controls_disabled)

        with st.expander("🎯 CFG-Free Sampling"):
            st.markdown("**CFG-Free Sampling** gradually reduces CFG to 0 in later steps for faster generation with minimal quality impact")
            settings["cfg_free_enabled"] = st.checkbox(
                "Enable CFG-Free Sampling",
                value=settings.get("cfg_free_enabled", False),
                help="Gradually reduce CFG guidance to 0 after a certain percentage of steps",
                disabled=controls_disabled
            )

            if settings["cfg_free_enabled"]:
                settings["cfg_free_start_percent"] = st.slider(
                    "Start Reducing CFG at (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=settings.get("cfg_free_start_percent", 70.0),
                    step=5.0,
                    help="Percentage of steps after which CFG will gradually reduce to 0 (recommended: 60-80%)",
                    disabled=controls_disabled
                )
                st.info(f"💡 CFG will remain at full strength until {settings['cfg_free_start_percent']:.0f}% of steps, then gradually reduce to 0")

        with st.expander("🔀 Token Merging (ToMe)"):
            st.markdown("**Token Merging** reduces computation by merging similar tokens (20-60% speedup)")
            settings["tome_enabled"] = st.checkbox(
                "Enable Token Merging",
                value=settings.get("tome_enabled", False),
                help="Merge similar tokens to reduce computation with minimal quality loss",
                disabled=controls_disabled
            )

            if settings["tome_enabled"]:
                tome_presets = {
                    "conservative": "Conservative - 30% merge (minimal quality impact)",
                    "balanced": "Balanced - 50% merge (recommended)",
                    "aggressive": "Aggressive - 70% merge (max speed)",
                    "custom": "Custom - Manual configuration"
                }

                if settings.get("tome_custom", False):
                    current_preset = "custom"
                else:
                    current_preset = settings.get("tome_preset", "balanced")

                selected_preset = st.selectbox(
                    "ToMe Preset",
                    options=list(tome_presets.keys()),
                    format_func=lambda x: tome_presets[x],
                    index=list(tome_presets.keys()).index(current_preset) if current_preset in tome_presets else 1,
                    disabled=controls_disabled
                )

                if selected_preset == "conservative":
                    settings["tome_custom"] = False
                    settings["tome_preset"] = "conservative"
                    settings["tome_ratio"] = 0.3
                    settings["tome_max_downsample"] = 2
                elif selected_preset == "balanced":
                    settings["tome_custom"] = False
                    settings["tome_preset"] = "balanced"
                    settings["tome_ratio"] = 0.5
                    settings["tome_max_downsample"] = 1
                elif selected_preset == "aggressive":
                    settings["tome_custom"] = False
                    settings["tome_preset"] = "aggressive"
                    settings["tome_ratio"] = 0.7
                    settings["tome_max_downsample"] = 1
                else:  # custom
                    settings["tome_custom"] = True
                    settings["tome_ratio"] = st.slider(
                        "Merge Ratio",
                        min_value=0.0,
                        max_value=0.9,
                        value=settings.get("tome_ratio", 0.5),
                        step=0.05,
                        help="Percentage of tokens to merge (higher = faster but may impact quality)",
                        disabled=controls_disabled
                    )
                    settings["tome_max_downsample"] = st.slider(
                        "Max Downsample Level",
                        min_value=1,
                        max_value=8,
                        value=settings.get("tome_max_downsample", 1),
                        help="Apply only to layers with downsampling <= this value",
                        disabled=controls_disabled
                    )

                st.info(f"💡 ToMe will merge ~{settings['tome_ratio']*100:.0f}% of similar tokens for speedup")

        with st.expander("⚡ Advanced CFG Optimizations"):
            st.caption("Note: Batched CFG (8% speedup) is always enabled by default")
            
            settings["dynamic_cfg_rescaling"] = st.checkbox(
                "Enable Dynamic CFG Rescaling",
                value=settings.get("dynamic_cfg_rescaling", False),
                help="Dynamically adjusts CFG scale based on guidance statistics to prevent over-saturation. Experimental feature.",
                disabled=controls_disabled
            )
            
            if settings["dynamic_cfg_rescaling"]:
                col1, col2 = st.columns(2)
                with col1:
                    settings["dynamic_cfg_method"] = st.selectbox(
                        "Rescaling Method",
                        options=["variance", "range"],
                        index=0 if settings.get("dynamic_cfg_method", "variance") == "variance" else 1,
                        help="Variance: uses spatial variance of guidance. Range: uses percentile-based range.",
                        disabled=controls_disabled
                    )
                with col2:
                    settings["dynamic_cfg_percentile"] = st.slider(
                        "Percentile (Range method)",
                        min_value=80.0,
                        max_value=99.0,
                        value=settings.get("dynamic_cfg_percentile", 95.0),
                        step=1.0,
                        help="Percentile threshold for range-based rescaling.",
                        disabled=controls_disabled
                    )
                settings["dynamic_cfg_target_scale"] = st.slider(
                    "Target CFG Scale",
                    min_value=1.0,
                    max_value=15.0,
                    value=settings.get("dynamic_cfg_target_scale", 7.0),
                    step=0.5,
                    help="Target CFG scale when rescaling is applied.",
                    disabled=controls_disabled
                )
                st.info("💡 Dynamic CFG can improve quality by preventing over-saturation")
            
            settings["adaptive_noise_enabled"] = st.checkbox(
                "Enable Adaptive Noise Scheduling",
                value=settings.get("adaptive_noise_enabled", False),
                help="Dynamically adjusts noise schedule based on image complexity. Experimental feature.",
                disabled=controls_disabled
            )
            
            if settings["adaptive_noise_enabled"]:
                settings["adaptive_noise_method"] = st.selectbox(
                    "Noise Scheduling Method",
                    options=["complexity", "attention"],
                    index=0 if settings.get("adaptive_noise_method", "complexity") == "complexity" else 1,
                    help="Complexity: uses spatial variance. Attention: uses gradient magnitude.",
                    disabled=controls_disabled
                )
                st.info("💡 Adaptive noise can optimize step allocation based on image complexity")

        with st.expander("💾 VRAM & Cache"):
            settings["keep_models_loaded"] = st.checkbox("Keep Models in VRAM", value=settings["keep_models_loaded"], disabled=controls_disabled)

            if st.button("Clear Model Cache", disabled=controls_disabled):
                clear_model_cache()
                st.success("Cache cleared!")
            try:
                mem_info = get_memory_info()
                used_vram = mem_info.get('used_vram', 0)
                total_vram = mem_info.get('total_vram', 0)
                st.text(f"VRAM: {used_vram:.1f}GB / {total_vram:.1f}GB")
            except Exception as e:
                st.text(f"VRAM: Unable to detect ({str(e)})")

        st.divider()
        settings["verbose_mode"] = st.checkbox("Verbose Logging", value=settings["verbose_mode"], disabled=controls_disabled)
        st.session_state.verbose_mode = settings["verbose_mode"]
        settings["ui_scale"] = st.slider("UI Display Scale", min_value=0.5, max_value=3.0, value=settings.get("ui_scale", 1.0), step=0.25, help="Scale factor applied to preview and output display size (independent of image resolution).", disabled=controls_disabled)
        try:
            scale_val = float(settings["ui_scale"])
            base_display = st.session_state.get("display_size", (512, 512))
            st.session_state.ui_display_size = (min(int(base_display[0] * scale_val), 1400), min(int(base_display[1] * scale_val), 1000))
        except Exception:
            pass

    st.session_state.settings = settings
    ui_settings.save_settings(settings)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("🎨 Generate", width='stretch', disabled=st.session_state.is_generating, type="primary", on_click=prepare_generation)
    with col2:
        stop_clicked = st.button("⏹️ Stop", width='stretch', disabled=not st.session_state.is_generating)

    if stop_clicked:
        stop_generation()

    st.divider()

    status_placeholder = st.empty()
    gallery_placeholder = st.empty()
    status_bar = st.empty()

    if st.session_state.generated_image_paths and not st.session_state.is_generating:
        paths = st.session_state.generated_image_paths

        # Determine base display size from the first generated image so the
        # UI matches the actual generated file's aspect ratio instead of the
        # user-entered width/height values (which may be different for
        # Img2Img/ADetailer flows).
        try:
            with Image.open(paths[0]) as first_img:
                first_w, first_h = first_img.size
        except Exception:
            # Fallback to the previously stored display size if the file is
            # not available or cannot be opened.
            stored_ds = st.session_state.get("display_size", (512, 512))
            first_w, first_h = stored_ds

        display_size = compute_display_size(first_w, first_h)
        UI_SCALE = float(settings.get("ui_scale", 1.0))
        UI_MAX_WIDTH = 1400
        UI_MAX_HEIGHT = 1000
        ui_full_w = min(int(display_size[0] * UI_SCALE), UI_MAX_WIDTH)
        ui_full_h = min(int(display_size[1] * UI_SCALE), UI_MAX_HEIGHT)

        # Persist the computed display sizes in session so other UI pieces
        # (and generation code) can reuse them.
        st.session_state.display_size = display_size
        st.session_state.ui_display_size = (ui_full_w, ui_full_h)

        cols_count = min(3, len(paths)) or 1
        cols = st.columns(cols_count)
        for idx, path in enumerate(paths):
            try:
                with Image.open(path) as img:
                    orig_w, orig_h = img.size
                    if len(paths) == 1:
                        tile_w, tile_h = ui_full_w, ui_full_h
                    else:
                        tile_w = max(64, int(ui_full_w / cols_count))
                        tile_h = max(64, int(tile_w * (orig_h / (orig_w or 1))))

                    with cols[idx % cols_count]:
                        render_responsive_image(img, (tile_w, tile_h))
            except Exception as e:
                with cols[idx % cols_count]:
                    st.warning(f"Could not load image: {e}")
    else:
        gallery_placeholder.info("👈 Configure settings and click Generate to create images")

    if st.session_state.get("start_generation", False):
        st.session_state.start_generation = False
        prev_job = st.session_state.get("generation_job") or {}
        prev_thread = prev_job.get("thread") if isinstance(prev_job, dict) else None
        if prev_thread is not None and hasattr(prev_thread, "is_alive") and prev_thread.is_alive():
            status_placeholder.warning("⚠️ Previous generation is still stopping. Please wait a moment and try Generate again.")
            st.session_state.start_generation = False
        else:
            st.session_state.generated_images = []
            st.session_state.generated_image_paths = []
            generate_images(settings, status_placeholder, gallery_placeholder, status_bar)
            st.rerun()


def render_history_page():
    """Render the history page with past generations (delegated)."""
    st.header("📜 Generation History")

    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("🔄 Scan Folders"):
            with st.spinner("Scanning output folders..."):
                history = scan_output_folders()
            st.success(f"Found {len(history)} images!")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All"):
            clear_history()
            st.rerun()

    # Search and Filter UI with auto-search (no Enter needed)
    with st.expander("🔍 Search & Filter", expanded=False):
        filter_cols = st.columns([2, 1, 1, 1])
        with filter_cols[0]:
            # Use on_change to trigger immediate search without Enter key
            def on_search_change():
                """Callback when search text changes - triggers rerun for live search."""
                pass  # The session state update happens automatically
            
            search_keyword = st.text_input(
                "Search prompt",
                key="history_search_keyword",
                placeholder="Type to search...",
                on_change=on_search_change,
                label_visibility="collapsed"
            )
            st.caption("🔍 Search prompts (live)")
        with filter_cols[1]:
            model_types = [""] + get_available_model_types()
            selected_model = st.selectbox(
                "Model type",
                options=model_types,
                format_func=lambda x: "All models" if x == "" else x,
                key="history_filter_model"
            )
        with filter_cols[2]:
            date_from = st.date_input(
                "From date",
                value=None,
                key="history_date_from"
            )
        with filter_cols[3]:
            date_to = st.date_input(
                "To date",
                value=None,
                key="history_date_to"
            )

    # Apply filters
    if search_keyword or selected_model or date_from or date_to:
        history = search_history(
            keyword=search_keyword if search_keyword else None,
            model_type=selected_model if selected_model else None,
            date_from=str(date_from) if date_from else None,
            date_to=str(date_to) if date_to else None,
        )
    else:
        history = load_history()

    with col3:
        st.text(f"Showing: {len(history)}")

    if not history:
        st.info("No generation history yet. Create some images or click 'Scan Folders' to find existing ones!")
        return

    st.divider()

    # Display history in a grid (3 columns)
    cols_per_row = 3
    for idx in range(0, len(history), cols_per_row):
        cols = st.columns(cols_per_row)

        for col_idx, col in enumerate(cols):
            entry_idx = idx + col_idx
            if entry_idx >= len(history):
                break

            entry = history[entry_idx]
            img_path = entry["image_path"]

            with col:
                # Check if image still exists
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        st.image(img, width='stretch')

                        # Compact info
                        with st.expander("ℹ️ Details", expanded=False):
                            st.text(f"🕒 {entry.get('timestamp')}")
                            st.text(f"📐 {entry.get('width')}x{entry.get('height')}")
                            batch = entry.get("batch_size")
                            if batch is not None:
                                st.text(f"🔁 Batch: {batch}")

                            # Key metadata
                            # Prefer top-level values (already sanitized) but
                            # fall back to the raw PNG metadata when the
                            # top-level entry is missing or intentionally
                            # suppressed. This makes the Details panel show
                            # a friendly value while the All metadata view
                            # preserves the full JSON blob.
                            png_meta = entry.get("png_metadata") or {}
                            seed = entry.get("seed") or png_meta.get("seed")
                            sampler = entry.get("sampler") or png_meta.get("sampler")
                            steps = entry.get("steps") or png_meta.get("steps")
                            cfg = entry.get("cfg") or png_meta.get("cfg")
                            if seed:
                                st.text(f"🔢 Seed: {seed}")
                            if sampler:
                                st.text(f"🎛️ Sampler: {sampler}")
                            if steps or cfg:
                                st.text(f"⚙️ Steps/CFG: {steps or '?'} / {cfg or '?'}")
                            # Timing metrics (if available)
                            gen_dur = entry.get("generation_duration")
                            avg_it = entry.get("avg_iters_per_s")
                            if gen_dur is not None:
                                try:
                                    st.text(f"⏱️ Duration: {float(gen_dur):.2f}s")
                                except Exception:
                                    st.text(f"⏱️ Duration: {gen_dur}")
                            if avg_it is not None:
                                try:
                                    st.text(f"⚡ Avg iters/s: {float(avg_it):.2f}")
                                except Exception:
                                    st.text(f"⚡ Avg iters/s: {avg_it}")

                            model_type = entry.get('model_type') or (entry.get('png_metadata', {}).get('model_type'))
                            if model_type:
                                st.text(f"Model: {model_type}")

                            st.text_area(
                                "Prompt",
                                value=entry.get("prompt", ""),
                                height=60,
                                disabled=True,
                                key=f"prompt_{entry_idx}"
                            )

                            # Action buttons
                            col_dl, col_del = st.columns(2)
                            with col_dl:
                                with open(img_path, "rb") as f:
                                        hist_key = hashlib.md5(img_path.encode('utf-8')).hexdigest()[:8]
                                        st.download_button(
                                            label="💾",
                                            data=f,
                                            file_name=os.path.basename(img_path),
                                            mime="image/png",
                                            key=f"download_history_{entry_idx}_{hist_key}",
                                            width='stretch'
                                        )
                            with col_del:
                                if st.button("🗑️", key=f"delete_{entry_idx}", width='stretch'):
                                    if delete_history_entry(entry_idx):
                                        st.rerun()
                            # All metadata expander (minimalistic)
                            with st.expander("🧾 All metadata", expanded=False):
                                meta_display = {k: v for k, v in entry.items() if k != 'png_metadata'}
                                png_meta = entry.get('png_metadata') or {}
                                merged = {"entry": meta_display, "png_metadata": png_meta}
                                try:
                                    st.json(merged)
                                except Exception:
                                    st.text(str(merged))
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
                else:
                    st.warning("⚠️ Image not found")
                    st.text(f"🕒 {entry['timestamp']}")
                    st.caption(os.path.basename(img_path))