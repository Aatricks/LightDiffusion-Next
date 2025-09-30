import streamlit as st
import os
import sys
import json
import glob
import datetime
import threading
import time
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.user.pipeline import pipeline
from src.user import app_instance

# Page configuration
st.set_page_config(
    page_title="LightDiffusion",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalistic styling
st.markdown("""
<style>
    /* Minimal header */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #8892a6;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* Simplified buttons */
    .stButton>button {
        width: 100%;
        background: #667eea;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border-radius: 6px;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: #5568d3;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e1e4e8;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Minimal expanders */
    .stExpander {
        border: none;
        border-top: 1px solid #e1e4e8;
    }
    
    /* Clean images */
    div[data-testid="stImage"] {
        border-radius: 4px;
    }
    
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Reduce padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Settings file
SETTINGS_FILE = "webui_settings.json"


def get_default_settings():
    """Get default settings for the webui"""
    return {
        "prompt": "",
        "negative_prompt": "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic)",
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
        "enable_preview": True,
    }


def load_settings():
    """Load settings from disk"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved_settings = json.load(f)
                default_settings = get_default_settings()
                default_settings.update(saved_settings)
                return default_settings
    except Exception as e:
        st.error(f"Error loading settings: {e}")
    return get_default_settings()


def save_settings(settings):
    """Save settings to disk"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving settings: {e}")


def load_generated_images():
    """Load the most recent batch of generated images"""
    image_files = glob.glob("./output/**/*.png", recursive=True)
    
    if not image_files:
        return []
    
    image_files.sort(key=os.path.getmtime, reverse=True)
    latest_time = os.path.getmtime(image_files[0])
    
    batch_images = []
    for file in image_files:
        if abs(os.path.getmtime(file) - latest_time) < 1.0:
            try:
                img = Image.open(file)
                batch_images.append(img)
            except Exception:
                continue
    
    return batch_images


def load_all_generated_images():
    """Load all generated images for history view"""
    image_files = glob.glob("./output/**/*.png", recursive=True)
    
    if not image_files:
        return []
    
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append({
                "image": img,
                "path": file_path,
                "filename": os.path.basename(file_path),
                "folder": os.path.basename(os.path.dirname(file_path)),
                "modified": datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{img.size[0]}x{img.size[1]}",
            })
        except Exception:
            continue
    
    return images


def get_vram_info():
    """Get VRAM usage information"""
    try:
        from src.Device.ModelCache import get_memory_info
        return get_memory_info()
    except Exception as e:
        return {"error": str(e)}


def clear_model_cache_ui():
    """Clear model cache from UI"""
    try:
        from src.Device.ModelCache import clear_model_cache
        clear_model_cache()
        return True
    except Exception as e:
        st.error(f"Error clearing cache: {e}")
        return False


def apply_multiscale_preset(preset_name):
    """Apply multiscale preset values"""
    if preset_name == "None":
        return None
    
    try:
        from src.sample.multiscale_presets import get_preset_parameters
        return get_preset_parameters(preset_name)
    except Exception as e:
        st.error(f"Error applying preset {preset_name}: {e}")
        return None


def generate_images(settings, progress_placeholder, status_placeholder, gallery_placeholder):
    """Generate images with the given settings and live preview support"""
    try:
        # Set preview enabled state
        app_instance.app.previewer_var.set(settings["enable_preview"])
        app_instance.app.cleanup_all_previews()
        
        # Set model persistence
        from src.Device.ModelCache import set_keep_models_loaded
        set_keep_models_loaded(settings["keep_models_loaded"])
        
        # Handle img2img
        img2img_image_path = None
        if settings["img2img_enabled"] and settings.get("img2img_image") is not None:
            img_array = settings["img2img_image"]
            if isinstance(img_array, np.ndarray):
                img_pil = Image.fromarray(img_array)
                img2img_image_path = "temp_img2img.png"
                img_pil.save(img2img_image_path)
        
        status_placeholder.info("🎨 Generating images...")
        
        # Variable to store final images
        final_images = []
        generation_complete = threading.Event()
        
        # Run generation in background thread
        def run_generation():
            nonlocal final_images
            try:
                final_images = pipeline(
                    prompt=settings["prompt"],
                    negative_prompt=settings["negative_prompt"],
                    w=settings["width"],
                    h=settings["height"],
                    number=settings["num_images"],
                    batch=settings["batch_size"],
                    hires_fix=settings["hires_fix"],
                    adetailer=settings["adetailer"],
                    enhance_prompt=settings["enhance_prompt"],
                    img2img=settings["img2img_enabled"],
                    img2img_image=img2img_image_path,
                    stable_fast=settings["stable_fast"],
                    reuse_seed=settings["reuse_seed"],
                    flux_enabled=settings["flux_enabled"],
                    prio_speed=settings["prio_speed"],
                    autohdr=True,
                    realistic_model=settings["realistic_model"],
                    enable_multiscale=settings["multiscale_enabled"],
                    multiscale_intermittent_fullres=settings["multiscale_intermittent"],
                    multiscale_factor=settings["multiscale_factor"],
                    multiscale_fullres_start=settings["multiscale_fullres_start"],
                    multiscale_fullres_end=settings["multiscale_fullres_end"],
                )
            finally:
                generation_complete.set()
        
        # Start generation thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)
        gen_thread.start()
        
        # Monitor for preview updates if enabled
        if settings["enable_preview"]:
            last_preview_time = 0
            preview_container = gallery_placeholder.container()
            
            while not generation_complete.is_set():
                current_previews = app_instance.app.get_latest_previews()
                if current_previews and app_instance.app.last_preview_time > last_preview_time:
                    last_preview_time = app_instance.app.last_preview_time
                    preview_images = []
                    for preview_path in current_previews:
                        try:
                            preview_images.append(Image.open(preview_path))
                        except:
                            pass
                    if preview_images:
                        with preview_container:
                            st.caption("🔄 Preview (TAESD)")
                            st.image(preview_images)
                        status_placeholder.info(f"🎨 Generating... ({len(preview_images)} preview(s))")
                time.sleep(0.5)
        
        # Wait for generation to complete
        gen_thread.join()
        
        # Cleanup
        app_instance.app.cleanup_all_previews()
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")
        
        status_placeholder.success("✅ Generation complete!")
        return load_generated_images()
        
    except Exception as e:
        import traceback
        status_placeholder.error(f"❌ Error: {str(e)}")
        st.error(traceback.format_exc())
        app_instance.app.cleanup_all_previews()
        if os.path.exists("temp_img2img.png"):
            os.remove("temp_img2img.png")
        return []


# Initialize session state
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()
if "show_help" not in st.session_state:
    st.session_state.show_help = False

# Header with help button
col_title, col_help = st.columns([6, 1])
with col_title:
    st.markdown('<h1 class="main-header">LightDiffusion</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fast AI image generation</p>', unsafe_allow_html=True)
with col_help:
    if st.button("?", help="Show help"):
        st.session_state.show_help = not st.session_state.show_help

# Help dialog
if st.session_state.show_help:
    with st.container():
        st.info("""
        **Quick Start Guide**
        
        1. **Prompt**: Describe what you want to generate
        2. **Dimensions**: Set width and height (512x512 recommended)
        3. **Images**: Choose how many to generate
        4. Click **Generate**
        
        **Tips:**
        - Use descriptive prompts for better results
        - Enable **HiRes Fix** for 2x quality boost
        - **Auto Enhance** improves faces automatically
        - **Fast** mode prioritizes speed over quality
        - **Flux** mode uses advanced Flux model
        - **Img2Img** generates from an uploaded image
        
        **Multi-Scale Presets:**
        - **Quality**: Best results, slower
        - **Performance**: Fastest, good results
        - **Balanced**: Good speed and quality
        - **None**: Disable optimization
        
        **Advanced:**
        - **Reuse Seed**: Same results with same prompt
        - **Keep in VRAM**: Faster subsequent generations
        - **Live Preview**: See progress during generation
        """)
        if st.button("Close Help"):
            st.session_state.show_help = False
            st.rerun()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Generate", "History", "Settings"])

# ============================================================================
# TAB 1: GENERATE
# ============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Prompt
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.settings["prompt"],
            height=100,
            placeholder="Describe what you want to generate...",
            key="prompt_input"
        )
        
        # Negative prompt
        with st.expander("Negative Prompt", expanded=False):
            negative_prompt = st.text_area(
                "What to avoid",
                value=st.session_state.settings["negative_prompt"],
                height=60,
                key="negative_prompt_input"
            )
        
        st.markdown("**Dimensions**")
        dim_col1, dim_col2 = st.columns(2)
        with dim_col1:
            width = st.slider("Width", 64, 2048, st.session_state.settings["width"], 64)
        with dim_col2:
            height = st.slider("Height", 64, 2048, st.session_state.settings["height"], 64)
        
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            num_images = st.slider("Images", 1, 10, st.session_state.settings["num_images"])
        with gen_col2:
            batch_size = st.slider("Batch", 1, 4, st.session_state.settings["batch_size"])
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Generation Modes")
        
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            flux_enabled = st.checkbox("🌊 Flux Mode", st.session_state.settings["flux_enabled"], help="Use Flux model for generation")
            realistic_model = st.checkbox("� Realistic Model", st.session_state.settings["realistic_model"], help="Use model optimized for realistic images")
        with mode_col2:
            img2img_enabled = st.checkbox("🖼️ Image to Image", st.session_state.settings["img2img_enabled"], help="Generate from an existing image")
            prio_speed = st.checkbox("🚀 Speed Priority", st.session_state.settings["prio_speed"], help="Optimize for faster generation")
        
        # Image to Image
        img2img_image = None
        if img2img_enabled:
            img2img_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if img2img_image:
                img2img_image = Image.open(img2img_image)
                st.image(img2img_image)
                img2img_image = np.array(img2img_image)
        
        # Enhancement features
        with st.expander("✨ Enhancement Features", expanded=False):
            feat_col1, feat_col2 = st.columns(2)
            with feat_col1:
                hires_fix = st.checkbox("🔍 HiRes Fix", st.session_state.settings["hires_fix"], help="2x upscale with refinement pass")
                adetailer = st.checkbox("� Auto Enhance", st.session_state.settings["adetailer"], help="Automatically enhance faces and bodies")
            with feat_col2:
                enhance_prompt = st.checkbox("✨ Enhance Prompt", st.session_state.settings["enhance_prompt"], help="Use AI to improve your prompt")
                stable_fast = st.checkbox("⚡ Stable Fast", st.session_state.settings["stable_fast"], help="Compile model for 70% speed boost (first run slower)")
        
        # Advanced settings
        with st.expander("Advanced", expanded=False):
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                reuse_seed = st.checkbox("Reuse Seed", st.session_state.settings["reuse_seed"])
                keep_models_loaded = st.checkbox("Keep in VRAM", st.session_state.settings["keep_models_loaded"])
            with adv_col2:
                enable_preview = st.checkbox("Live Preview", st.session_state.settings["enable_preview"])
        
        # Multi-scale settings
        with st.expander("Multi-Scale", expanded=False):
            multiscale_preset = st.selectbox(
                "Preset",
                ["None", "quality", "performance", "balanced"],
                index=["None", "quality", "performance", "balanced"].index(st.session_state.settings.get("multiscale_preset", "quality")) if st.session_state.settings.get("multiscale_preset", "quality") in ["None", "quality", "performance", "balanced"] else 0,
                key="multiscale_preset_selector"
            )
            
            # Apply preset values if changed
            if st.session_state.settings.get("multiscale_preset") != multiscale_preset:
                if multiscale_preset == "None":
                    # Disable multi-scale completely
                    st.session_state.settings["multiscale_enabled"] = False
                    st.session_state.settings["multiscale_preset"] = multiscale_preset
                    save_settings(st.session_state.settings)
                    st.rerun()
                else:
                    # Apply preset values
                    preset_params = apply_multiscale_preset(multiscale_preset)
                    if preset_params:
                        # Update session state with preset values
                        st.session_state.settings["multiscale_enabled"] = preset_params["enable_multiscale"]
                        st.session_state.settings["multiscale_factor"] = preset_params["multiscale_factor"]
                        st.session_state.settings["multiscale_fullres_start"] = preset_params["multiscale_fullres_start"]
                        st.session_state.settings["multiscale_fullres_end"] = preset_params["multiscale_fullres_end"]
                        st.session_state.settings["multiscale_intermittent"] = preset_params["multiscale_intermittent_fullres"]
                        st.session_state.settings["multiscale_preset"] = multiscale_preset
                        save_settings(st.session_state.settings)
                        st.rerun()
            
            multiscale_enabled = st.checkbox("Enable", st.session_state.settings["multiscale_enabled"])
            
            if multiscale_enabled:
                multiscale_factor = st.slider(
                    "Factor", 
                    0.1, 1.0, 
                    st.session_state.settings["multiscale_factor"], 
                    0.1
                )
                
                ms_col1, ms_col2 = st.columns(2)
                with ms_col1:
                    multiscale_fullres_start = st.number_input(
                        "Start", 
                        0, 20, 
                        st.session_state.settings["multiscale_fullres_start"]
                    )
                with ms_col2:
                    multiscale_fullres_end = st.number_input(
                        "End", 
                        0, 20, 
                        st.session_state.settings["multiscale_fullres_end"]
                    )
                
                multiscale_intermittent = st.checkbox(
                    "Intermittent", 
                    st.session_state.settings["multiscale_intermittent"]
                )
            else:
                # Set defaults when disabled
                multiscale_factor = st.session_state.settings["multiscale_factor"]
                multiscale_fullres_start = st.session_state.settings["multiscale_fullres_start"]
                multiscale_fullres_end = st.session_state.settings["multiscale_fullres_end"]
                multiscale_intermittent = st.session_state.settings["multiscale_intermittent"]
        
        # Generate button
        generate_btn = st.button("Generate", type="primary", use_container_width=True)
    
    with col_right:
        # Quick stats
        if st.session_state.settings.get("last_gen_time"):
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Resolution", f"{st.session_state.settings.get('last_width', width)}x{st.session_state.settings.get('last_height', height)}")
            with stat_col2:
                st.metric("Batch", f"{st.session_state.settings.get('last_num_images', num_images)}")
            with stat_col3:
                st.metric("Time", f"{st.session_state.settings.get('last_gen_time', 0):.1f}s")
        
        # Placeholders for status and results
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        gallery_placeholder = st.empty()
        
        # Show existing images if any
        existing_images = load_generated_images()
        if existing_images:
            gallery_placeholder.image(existing_images)
        else:
            gallery_placeholder.info("No images yet. Generate some to see them here!")
        
        # Handle generation
        if generate_btn:
            if not prompt.strip():
                status_placeholder.warning("⚠️ Please enter a prompt!")
            else:
                # Track generation time
                import time
                start_time = time.time()
                
                # Update settings
                current_settings = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_images": num_images,
                    "batch_size": batch_size,
                    "hires_fix": hires_fix,
                    "adetailer": adetailer,
                    "enhance_prompt": enhance_prompt,
                    "img2img_enabled": img2img_enabled,
                    "img2img_image": img2img_image,
                    "stable_fast": stable_fast,
                    "reuse_seed": reuse_seed,
                    "flux_enabled": flux_enabled,
                    "prio_speed": prio_speed,
                    "realistic_model": realistic_model,
                    "multiscale_enabled": multiscale_enabled,
                    "multiscale_intermittent": multiscale_intermittent,
                    "multiscale_factor": multiscale_factor,
                    "multiscale_fullres_start": multiscale_fullres_start,
                    "multiscale_fullres_end": multiscale_fullres_end,
                    "keep_models_loaded": keep_models_loaded,
                    "multiscale_preset": multiscale_preset,
                    "enable_preview": enable_preview,
                }
                
                st.session_state.settings.update(current_settings)
                save_settings(st.session_state.settings)
                
                # Generate
                with st.spinner("Generating images..."):
                    generated_images = generate_images(current_settings, progress_placeholder, status_placeholder, gallery_placeholder)
                    
                    # Calculate generation time
                    gen_time = time.time() - start_time
                    
                    if generated_images:
                        # Save stats BEFORE displaying images
                        st.session_state.settings.update({
                            "last_gen_time": gen_time,
                            "last_width": width,
                            "last_height": height,
                            "last_num_images": num_images,
                        })
                        save_settings(st.session_state.settings)
                        
                        # Force rerun to update stats display
                        st.rerun()
                    else:
                        gallery_placeholder.warning("No images were generated. Check the error messages above.")

# ============================================================================
# TAB 2: HISTORY
# ============================================================================
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Clear All", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_delete_all"):
                try:
                    image_files = glob.glob("./output/**/*.png", recursive=True)
                    for file_path in image_files:
                        os.remove(file_path)
                    st.success(f"Deleted {len(image_files)} images")
                    st.session_state.confirm_delete_all = False
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.session_state.confirm_delete_all = True
                st.warning("Click again to confirm")
    
    # Load all images
    all_images = load_all_generated_images()
    
    if not all_images:
        st.info("No images found")
    else:
        st.caption(f"Total: {len(all_images)}")
        
        # Display images in grid
        cols_per_row = 4
        for idx in range(0, len(all_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, col in enumerate(cols):
                img_idx = idx + col_idx
                if img_idx < len(all_images):
                    img_data = all_images[img_idx]
                    with col:
                        st.image(img_data["image"])
                        with st.expander("Info"):
                            st.caption(img_data['filename'])
                            st.caption(f"{img_data['size']} • {img_data['modified']}")
                            if st.button("Delete", key=f"delete_{img_idx}"):
                                try:
                                    os.remove(img_data['path'])
                                    st.success("Deleted")
                                    time.sleep(0.3)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

# ============================================================================
# TAB 3: SETTINGS
# ============================================================================
with tab3:
    # VRAM Info
    st.markdown("**VRAM**")
    if st.button("Check VRAM"):
        vram_info = get_vram_info()
        if "error" in vram_info:
            st.error(vram_info['error'])
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", f"{vram_info.get('total_vram', 0):.1f} GB")
            with col2:
                st.metric("Used", f"{vram_info.get('used_vram', 0):.1f} GB")
            with col3:
                st.metric("Free", f"{vram_info.get('free_vram', 0):.1f} GB")
            
            st.caption(f"Models cached: {vram_info.get('has_cached_checkpoint', False)}")
    
    st.divider()
    
    # Model Cache Management
    st.markdown("**Cache**")
    if st.button("Clear Cache", type="secondary"):
        if clear_model_cache_ui():
            st.success("Cache cleared")
        else:
            st.error("Failed to clear cache")
    
    st.divider()
    
    # About
    st.markdown("**About**")
    st.caption("LightDiffusion - Fast AI image generation")
    st.caption("[GitHub](https://github.com/LightDiffusion/LightDiffusion-Next)")

# Auto-save settings on every interaction
st.session_state.settings.update({
    "prompt": prompt if 'prompt_input' in locals() else st.session_state.settings["prompt"],
    "negative_prompt": negative_prompt if 'negative_prompt_input' in locals() else st.session_state.settings["negative_prompt"],
})
save_settings(st.session_state.settings)
