"""
LightDiffusion Streamlit UI
A minimalistic, single-mode web interface focused on the generated image.
All controls are in a togglable sidebar, main canvas displays preview/final at same size.
"""

import base64
import io
import streamlit as st
import os
import json
import glob
import threading
import time
import numpy as np
from PIL import Image

# Core Pipeline Integration
from src.user.pipeline import pipeline
from src.user import app_instance
from src.Device.ModelCache import (
    set_keep_models_loaded,
    get_memory_info,
    clear_model_cache
)
from src.FileManaging.Downloader import CheckAndDownload

# Page Configuration
st.set_page_config(
    page_title="LightDiffusion",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Default Settings Schema
# ============================================================================

def get_default_settings():
    """Returns default settings for the UI"""
    return {
        # Prompt & Text
        "prompt": "",
        "negative_prompt": "",
        
        # Dimensions & Batch
        "width": 512,
        "height": 512,
        "num_images": 1,
        
        # Generation Modes
        "flux_mode": False,
        "realistic_mode": False,
        "img2img_mode": False,
        "speed_mode": False,
        
        # Image Input (Img2Img)
        "input_image_path": "",
        
        # Enhancement Features
        "hiresfix": False,
        "adetailer": False,
        "enhance_prompt": False,
        "stable_fast": False,
        
        # Advanced Settings
        "reuse_seed": False,
        "keep_models_loaded": True,
        "enable_preview": True,
        
        # Multi-scale
        "multiscale_preset": "balanced",
        "multiscale_custom": False,
        "multiscale_factor": 0.5,
        "multiscale_fullres_start": 3,
        "multiscale_fullres_end": 8,
        "multiscale_intermittent_fullres": False,
        
        # UI Settings
        "dark_mode": True,
        "verbose_mode": False,
        "sidebar_collapsed_by_default": False,
    }

# ============================================================================
# Settings Persistence
# ============================================================================

SETTINGS_FILE = "./webui_settings.json"

def load_settings():
    """Load settings from disk, merge with defaults"""
    defaults = get_default_settings()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception as e:
            st.warning(f"Could not load settings: {e}")
    return defaults

def save_settings(settings):
    """Save settings to disk"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Could not save settings: {e}")

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize all required session state variables"""
    
    # Load settings
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    
    # UI State
    if "show_help" not in st.session_state:
        st.session_state.show_help = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = st.session_state.settings.get("dark_mode", True)
    if "verbose_mode" not in st.session_state:
        st.session_state.verbose_mode = st.session_state.settings.get("verbose_mode", False)
    
    # Generation State
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "interrupt_generation" not in st.session_state:
        st.session_state.interrupt_generation = False
    if "lightdiffusion_ready" not in st.session_state:
        st.session_state.lightdiffusion_ready = False
    if "enhanced_prompt_preview" not in st.session_state:
        st.session_state.enhanced_prompt_preview = {}
    if "generation_job" not in st.session_state:
        st.session_state.generation_job = {}
    if "start_generation" not in st.session_state:
        st.session_state.start_generation = False
    
    # Setup State
    if "setup_progress" not in st.session_state:
        st.session_state.setup_progress = 0.0
    if "setup_message" not in st.session_state:
        st.session_state.setup_message = "Initializing..."
    if "setup_thread" not in st.session_state:
        st.session_state.setup_thread = None
    if "setup_status" not in st.session_state:
        st.session_state.setup_status = {
            "progress": 0.0,
            "message": "Initializing...",
            "complete": False,
            "error": None,
            "pipeline": None,
            "app_instance": None
        }
    
    # Display State
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "display_size" not in st.session_state:
        st.session_state.display_size = (512, 512)

# ============================================================================
# Background Initialization
# ============================================================================

def initialize_lightdiffusion(status_dict, verbose=False):
    """Initialize LightDiffusion in background thread"""
    try:
        status_dict["message"] = "Checking and downloading models..."
        status_dict["progress"] = 0.1
        
        # Download models
        CheckAndDownload()
        
        status_dict["message"] = "Loading pipeline..."
        status_dict["progress"] = 0.5
        
        # Import pipeline (already done at top)
        status_dict["pipeline"] = pipeline
        status_dict["app_instance"] = app_instance
        
        status_dict["message"] = "Ready!"
        status_dict["progress"] = 1.0
        status_dict["complete"] = True
        
    except Exception as e:
        status_dict["error"] = str(e)
        status_dict["complete"] = True
        if verbose:
            import traceback
            traceback.print_exc()

def start_initialization():
    """Start background initialization if not already started"""
    if st.session_state.setup_thread is None or not st.session_state.setup_thread.is_alive():
        status_dict = st.session_state.setup_status
        status_dict["complete"] = False
        status_dict["error"] = None
        
        thread = threading.Thread(
            target=initialize_lightdiffusion,
            args=(status_dict, st.session_state.verbose_mode),
            daemon=True
        )
        thread.start()
        st.session_state.setup_thread = thread

# ============================================================================
# CSS Styling
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for theming and responsive images"""
    theme = "dark" if st.session_state.dark_mode else "light"
    
    css = f"""
    <style>
    /* Main theming */
    :root {{
        --ld-bg-primary: {'#0e1117' if theme == 'dark' else '#ffffff'};
        --ld-bg-secondary: {'#262730' if theme == 'dark' else '#f0f2f6'};
        --ld-text-primary: {'#fafafa' if theme == 'dark' else '#262730'};
        --ld-text-secondary: {'#a3a8b4' if theme == 'dark' else '#6c757d'};
        --ld-accent: {'#ff4b4b' if theme == 'dark' else '#ff4b4b'};
    }}
    
    /* Responsive image container */
    .ld-responsive-image {{
        width: var(--ld-display-width, 100%);
        height: var(--ld-display-height, auto);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
        overflow: hidden;
    }}
    
    .ld-responsive-image img {{
        width: var(--ld-display-width, auto) !important;
        height: var(--ld-display-height, auto) !important;
        max-width: 100% !important;
        max-height: 100% !important;
        object-fit: contain;
        display: block;
        margin: 0 auto;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.38);
        border-radius: 8px;
    }}
    
    /* Status indicators */
    .status-generating {{
        color: #ffa500;
        animation: pulse 1.5s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    
    /* Sidebar styling */
    .sidebar .sidebar-content {{
        background-color: var(--ld-bg-secondary);
    }}
    
    /* Compact expanders */
    .streamlit-expanderHeader {{
        font-size: 0.95rem;
        font-weight: 500;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# Image Display Helpers
# ============================================================================

def compute_display_size(image_width, image_height, max_width=800, max_height=600):
    """Compute display size that fits viewport while preserving aspect ratio"""
    aspect_ratio = image_width / image_height
    
    if aspect_ratio > max_width / max_height:
        # Width-constrained
        display_w = max_width
        display_h = int(max_width / aspect_ratio)
    else:
        # Height-constrained
        display_h = max_height
        display_w = int(max_height * aspect_ratio)
    
    return (display_w, display_h)

@st.cache_data
def image_to_base64(image, format="PNG"):
    """Convert PIL Image to base64 string (cached)"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def render_responsive_image(image, target_display_size, placeholder=None):
    """Render image at exact display size with CSS variables"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    display_w, display_h = target_display_size
    
    # Create display copy (downscaled if needed)
    if image.size != target_display_size:
        display_image = image.resize(target_display_size, Image.Resampling.LANCZOS)
    else:
        display_image = image
    
    # Convert to base64
    img_b64 = image_to_base64(display_image)
    
    # Render with CSS variables
    html = f"""
    <div class="ld-responsive-image" style="--ld-display-width: {display_w}px; --ld-display-height: {display_h}px;">
        <img src="data:image/png;base64,{img_b64}" alt="Generated Image">
    </div>
    """
    
    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)

# ============================================================================
# Generation Functions
# ============================================================================

def generate_images(settings, status_placeholder, gallery_placeholder):
    """Generate images with live preview support"""
    
    # Pre-generation checks
    if not st.session_state.lightdiffusion_ready:
        status_placeholder.error("❌ LightDiffusion is not initialized yet!")
        return []
    
    if not settings["prompt"].strip():
        status_placeholder.warning("⚠️ Please enter a prompt!")
        return []
    
    # Setup generation state
    st.session_state.interrupt_generation = False
    st.session_state.is_generating = True
    app_instance.app.cleanup_all_previews()
    app_instance.app.clear_interrupt()
    set_keep_models_loaded(settings["keep_models_loaded"])
    
    # Create output directories
    os.makedirs("./output/preview_display", exist_ok=True)
    
    # Background generation thread
    pipeline_result = None
    generation_complete = threading.Event()
    generation_error = None
    
    def run_generation():
        nonlocal pipeline_result, generation_error
        try:
            # Prepare multiscale parameters
            if settings["multiscale_custom"]:
                # Use custom settings
                multiscale_params = {
                    "multiscale_preset": None,
                    "enable_multiscale": True,
                    "multiscale_factor": settings["multiscale_factor"],
                    "multiscale_fullres_start": settings["multiscale_fullres_start"],
                    "multiscale_fullres_end": settings["multiscale_fullres_end"],
                    "multiscale_intermittent_fullres": settings["multiscale_intermittent_fullres"],
                }
            else:
                # Use preset
                multiscale_params = {
                    "multiscale_preset": settings["multiscale_preset"],
                }
            
            result = pipeline(
                prompt=settings["prompt"],
                negative_prompt=settings["negative_prompt"],
                w=settings["width"],
                h=settings["height"],
                number=settings["num_images"],
                batch=1,
                hires_fix=settings["hiresfix"],
                adetailer=settings["adetailer"],
                enhance_prompt=settings["enhance_prompt"],
                img2img=settings["img2img_mode"],
                stable_fast=settings["stable_fast"],
                reuse_seed=settings["reuse_seed"],
                flux_enabled=settings["flux_mode"],
                prio_speed=settings["speed_mode"],
                autohdr=True,
                realistic_model=settings["realistic_mode"],
                img2img_image=settings.get("input_image_path") if settings["img2img_mode"] else None,
                **multiscale_params,
            )
            pipeline_result = result
        except Exception as e:
            generation_error = e
            if settings["verbose_mode"]:
                import traceback
                traceback.print_exc()
        finally:
            generation_complete.set()
    
    # Start generation thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()
    
    # Track in session state
    st.session_state.generation_job = {
        "thread": gen_thread,
        "complete_event": generation_complete,
        "start_time": time.time()
    }
    
    # Compute display size
    display_size = compute_display_size(
        settings["width"], 
        settings["height"],
        max_width=800,
        max_height=600
    )
    st.session_state.display_size = display_size
    
    # Preview update loop
    preview_container = gallery_placeholder.empty() if settings["enable_preview"] else None
    last_preview_time = 0
    
    while not generation_complete.wait(0.2):
        # Check for interrupt
        if st.session_state.interrupt_generation:
            app_instance.app.request_interrupt()
            status_placeholder.warning("⏹️ Stopping generation...")
            break
        
        # Update preview
        if preview_container and settings["enable_preview"]:
            current_time = time.time()
            if current_time - last_preview_time > 0.5:  # Update every 0.5s
                previews = app_instance.app.get_latest_previews()
                if previews:
                    try:
                        latest_preview = Image.open(previews[-1])
                        render_responsive_image(latest_preview, display_size, preview_container)
                        last_preview_time = current_time
                    except Exception:
                        pass
        
        # Update status
        elapsed = time.time() - st.session_state.generation_job["start_time"]
        status_placeholder.info(f"🎨 Generating... ({elapsed:.1f}s)")
    
    # Cleanup previews
    try:
        app_instance.app.cleanup_all_previews()
    except Exception:
        pass
    
    # Small delay to ensure files are fully written
    time.sleep(0.3)
    
    # Handle results
    st.session_state.is_generating = False
    
    if generation_error:
        status_placeholder.error(f"❌ Generation failed: {generation_error}")
        return []
    
    # Find generated images (check all possible output directories)
    generated_images = []
    
    # Determine which directory to check based on generation mode
    if settings["flux_mode"]:
        primary_dirs = ["./output/Flux"]
    elif settings["img2img_mode"]:
        primary_dirs = ["./output/Img2Img"]
    elif settings["adetailer"]:
        primary_dirs = ["./output/Adetailer", "./output/Classic", "./output/HiresFix"]
    elif settings["hiresfix"]:
        primary_dirs = ["./output/HiresFix"]
    else:
        primary_dirs = ["./output/Classic"]
    
    # Collect all PNG files from primary directories
    all_outputs = []
    for output_dir in primary_dirs:
        if os.path.exists(output_dir):
            files = glob.glob(f"{output_dir}/*.png")
            all_outputs.extend(files)
    
    # Debug: print what we found
    if settings["verbose_mode"]:
        status_placeholder.info(f"Searching in: {primary_dirs}, Found {len(all_outputs)} files")
    
    # Sort by modification time and get the most recent
    if all_outputs:
        all_outputs = sorted(all_outputs, key=os.path.getmtime, reverse=True)
        for f in all_outputs[:settings["num_images"]]:
            try:
                img = Image.open(f)
                generated_images.append((img, f))
            except Exception as e:
                if settings["verbose_mode"]:
                    status_placeholder.warning(f"Error loading {f}: {e}")
    
    if generated_images:
        # Save to session state first
        st.session_state.generated_images = generated_images
        st.session_state.display_size = display_size
        
        # Display first image
        img, path = generated_images[0]
        render_responsive_image(img, display_size, gallery_placeholder)
        
        elapsed = time.time() - st.session_state.generation_job["start_time"]
        status_placeholder.success(f"✅ Generated {len(generated_images)} image(s) in {elapsed:.1f}s")
        return [img for img, _ in generated_images]
    else:
        # Show a helpful message with debug info
        if pipeline_result:
            checked_dirs = ", ".join(primary_dirs)
            status_placeholder.warning(f"⚠️ Generation completed but no images found. Checked: {checked_dirs}")
        else:
            status_placeholder.info("Generation stopped")
        return []

def stop_generation():
    """Request generation stop"""
    st.session_state.interrupt_generation = True


def prepare_generation():
    """Prepare state so controls become disabled on the next rerun,
    then start the generation flow from the rerun. This ensures the
    UI is re-rendered with disabled widgets before heavy work starts,
    preventing accidental extra clicks."""
    st.session_state.start_generation = True
    st.session_state.is_generating = True

# ============================================================================
# Main UI
# ============================================================================

def main():
    # Initialize session state
    init_session_state()
    
    # Inject CSS
    inject_custom_css()
    
    # Start initialization if needed
    if not st.session_state.lightdiffusion_ready and not st.session_state.setup_status.get("complete"):
        start_initialization()
    
    # Check initialization status
    if not st.session_state.lightdiffusion_ready:
        setup_status = st.session_state.setup_status
        
        if setup_status.get("error"):
            st.error(f"❌ Initialization failed: {setup_status['error']}")
            if st.button("Retry Initialization"):
                st.session_state.setup_thread = None
                st.rerun()
            return
        
        if setup_status.get("complete"):
            st.session_state.lightdiffusion_ready = True
            st.rerun()
        else:
            st.title("🎨 LightDiffusion")
            st.info(setup_status.get("message", "Initializing..."))
            progress = setup_status.get("progress", 0.0)
            st.progress(progress)
            time.sleep(0.5)
            st.rerun()
            return
    
    # ========================================================================
    # Header
    # ========================================================================
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🎨 LightDiffusion")
    with col2:
        if st.button("❓ Help"):
            st.session_state.show_help = not st.session_state.show_help
    
    if st.session_state.show_help:
        st.info("""
        **LightDiffusion Quick Guide:**
        - Enter your prompt in the sidebar
        - Adjust settings in the expandable sections
        - Click Generate to create images
        - Enable Live Preview to see progress
        - All settings auto-save
        """)
    
    # ========================================================================
    # Sidebar Controls
    # ========================================================================
    
    settings = st.session_state.settings
    
    with st.sidebar:
        st.header("⚙️ Settings")
        # Disable controls while a generation is active to avoid
        # user-driven reruns that can desync the UI thread and the
        # background generation thread. The Stop button remains active
        # so the user can interrupt generation.
        controls_disabled = st.session_state.is_generating
        if controls_disabled:
            st.warning("⚠️ Generation in progress — settings are locked until the current job finishes or you press Stop.")
        
        # Prompt & Text
        with st.expander("📝 Prompt & Text", expanded=True):
            prompt = st.text_area(
                "Prompt",
                value=settings["prompt"],
                height=100,
                key="prompt_input",
                disabled=controls_disabled,
            )
            settings["prompt"] = prompt

            negative_prompt = st.text_area(
                "Negative Prompt",
                value=settings["negative_prompt"],
                height=80,
                key="negative_prompt_input",
                disabled=controls_disabled,
            )
            settings["negative_prompt"] = negative_prompt
        
        # Dimensions & Batch
        with st.expander("📐 Dimensions & Batch", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                settings["width"] = st.number_input(
                    "Width",
                    min_value=64,
                    max_value=2048,
                    value=settings["width"],
                    step=64,
                    disabled=controls_disabled,
                )
            with col2:
                settings["height"] = st.number_input(
                    "Height",
                    min_value=64,
                    max_value=2048,
                    value=settings["height"],
                    step=64,
                    disabled=controls_disabled,
                )

            settings["num_images"] = st.number_input(
                "Number of Images",
                min_value=1,
                max_value=10,
                value=settings["num_images"],
                disabled=controls_disabled,
            )

            # Presets
            preset = st.selectbox(
                "Presets",
                ["Custom", "512x512", "768x768", "1024x1024", "512x768 (Portrait)", "768x512 (Landscape)"],
                disabled=controls_disabled,
            )
            if preset == "512x512":
                settings["width"], settings["height"] = 512, 512
            elif preset == "768x768":
                settings["width"], settings["height"] = 768, 768
            elif preset == "1024x1024":
                settings["width"], settings["height"] = 1024, 1024
            elif preset == "512x768 (Portrait)":
                settings["width"], settings["height"] = 512, 768
            elif preset == "768x512 (Landscape)":
                settings["width"], settings["height"] = 768, 512
        
        # Generation Modes
        with st.expander("🎯 Generation Modes", expanded=False):
            settings["flux_mode"] = st.checkbox("Flux Mode", value=settings["flux_mode"], disabled=controls_disabled)
            settings["realistic_mode"] = st.checkbox("Realistic Mode", value=settings["realistic_mode"], disabled=controls_disabled)
            settings["speed_mode"] = st.checkbox("Speed Mode", value=settings["speed_mode"], disabled=controls_disabled)
            settings["img2img_mode"] = st.checkbox("Img2Img Mode", value=settings["img2img_mode"], disabled=controls_disabled)

            # File uploader is not interactive while generating. Show current image
            # preview (if any) and an explanatory note instead of accepting new uploads.
            if settings["img2img_mode"]:
                if controls_disabled:
                    st.info("Image upload is disabled while generation is running. Stop the job to change the input image.")
                    if settings.get("input_image_path") and os.path.exists(settings.get("input_image_path")):
                        try:
                            st.image(settings.get("input_image_path"), caption="Current Input Image", use_column_width=True)
                        except Exception:
                            pass
                else:
                    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
                    if uploaded_file:
                        img_path = f"./output/uploaded_{uploaded_file.name}"
                        with open(img_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        settings["input_image_path"] = img_path
                        st.image(uploaded_file, caption="Input Image", use_column_width=True)
        
        # Enhancements
        with st.expander("✨ Enhancements", expanded=False):
            settings["hiresfix"] = st.checkbox("HiRes Fix", value=settings["hiresfix"], disabled=controls_disabled)
            settings["adetailer"] = st.checkbox("ADetailer", value=settings["adetailer"], disabled=controls_disabled)
            settings["enhance_prompt"] = st.checkbox("Enhance Prompt", value=settings["enhance_prompt"], disabled=controls_disabled)
            settings["stable_fast"] = st.checkbox("Stable Fast", value=settings["stable_fast"], disabled=controls_disabled)
        
        # Advanced Settings
        with st.expander("🔧 Advanced", expanded=False):
            settings["reuse_seed"] = st.checkbox("Reuse Seed", value=settings["reuse_seed"], disabled=controls_disabled)
            settings["enable_preview"] = st.checkbox("Live Preview", value=settings["enable_preview"], disabled=controls_disabled)
        
        # Multi-scale
        with st.expander("🔬 Multi-scale", expanded=False):
            preset_options = {
                "quality": "Quality - Best image quality with intermittent full-res",
                "balanced": "Balanced - Good quality and performance",
                "performance": "Performance - Maximum speed with aggressive downscaling",
                "disabled": "Disabled - Full resolution throughout",
                "custom": "Custom - Configure all settings manually"
            }
            
            # Determine current selection
            if settings.get("multiscale_custom", False):
                current_preset = "custom"
            else:
                current_preset = settings.get("multiscale_preset", "balanced")
            
            selected_preset = st.selectbox(
                "Preset",
                options=list(preset_options.keys()),
                format_func=lambda x: preset_options[x],
                index=list(preset_options.keys()).index(current_preset),
                disabled=controls_disabled,
            )
            
            if selected_preset == "custom":
                settings["multiscale_custom"] = True
                settings["multiscale_factor"] = st.slider(
                    "Scale Factor",
                    min_value=0.1,
                    max_value=1.0,
                    value=settings.get("multiscale_factor", 0.5),
                    step=0.05,
                    help="Scale factor for intermediate steps",
                    disabled=controls_disabled,
                )
                settings["multiscale_fullres_start"] = st.number_input(
                    "Full-res Start Steps",
                    min_value=0,
                    max_value=20,
                    value=settings.get("multiscale_fullres_start", 3),
                    help="Number of first steps at full resolution",
                    disabled=controls_disabled,
                )
                settings["multiscale_fullres_end"] = st.number_input(
                    "Full-res End Steps",
                    min_value=0,
                    max_value=20,
                    value=settings.get("multiscale_fullres_end", 8),
                    help="Number of last steps at full resolution",
                    disabled=controls_disabled,
                )
                settings["multiscale_intermittent_fullres"] = st.checkbox(
                    "Intermittent Full-res",
                    value=settings.get("multiscale_intermittent_fullres", False),
                    help="Enable intermittent full-res rendering in low-res region",
                    disabled=controls_disabled,
                )
            else:
                settings["multiscale_custom"] = False
                settings["multiscale_preset"] = selected_preset
        
        # VRAM & Cache
        with st.expander("💾 VRAM & Cache", expanded=False):
            settings["keep_models_loaded"] = st.checkbox(
                "Keep Models in VRAM",
                value=settings["keep_models_loaded"],
                disabled=controls_disabled,
            )

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
        
        # Verbose Mode
        st.divider()
        settings["verbose_mode"] = st.checkbox("Verbose Logging", value=settings["verbose_mode"], disabled=controls_disabled)
        st.session_state.verbose_mode = settings["verbose_mode"]
    
    # Save settings
    st.session_state.settings = settings
    save_settings(settings)
    
    # ========================================================================
    # Main Canvas
    # ========================================================================
    
    # Generate / Stop buttons in main layout
    col1, col2 = st.columns([1, 1])
    with col1:
        # Use an on_click callback to set state and trigger a rerun so the
        # sidebar widgets are re-rendered as disabled before starting the
        # generation work. The actual generation is started below when
        # `start_generation` is detected.
        st.button(
            "🎨 Generate",
            use_container_width=True,
            disabled=st.session_state.is_generating,
            type="primary",
            on_click=prepare_generation,
        )
    with col2:
        stop_clicked = st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_generating)
    
    if stop_clicked:
        stop_generation()
    
    st.divider()
    
    status_placeholder = st.empty()
    gallery_placeholder = st.empty()
    
    # Show existing images if any
    if st.session_state.generated_images and not st.session_state.is_generating:
        img, path = st.session_state.generated_images[0]
        display_size = st.session_state.display_size
        render_responsive_image(img, display_size, gallery_placeholder)
        
        # Download button
        with open(path, "rb") as f:
            st.download_button(
                label="💾 Download Image",
                data=f,
                file_name=os.path.basename(path),
                mime="image/png"
            )
        
        # Show all images if multiple
        if len(st.session_state.generated_images) > 1:
            st.subheader("All Generated Images")
            cols = st.columns(min(3, len(st.session_state.generated_images)))
            for idx, (img, path) in enumerate(st.session_state.generated_images):
                with cols[idx % 3]:
                    st.image(img, caption=f"Image {idx+1}", use_column_width=True)
    else:
        # Show placeholder
        gallery_placeholder.info("👈 Configure settings and click Generate to create images")
    
    # If the user just clicked Generate, the on_click callback set
    # `start_generation=True` and `is_generating=True` and triggered a
    # rerun. Now start the real generation work in this run so the UI
    # renders with disabled controls before heavy processing begins.
    if st.session_state.get("start_generation", False):
        # Reset the trigger so we don't start again on subsequent reruns
        st.session_state.start_generation = False
        # Clear previous images but keep display size
        st.session_state.generated_images = []
        # Start generation (this will update placeholders/live preview)
        generate_images(settings, status_placeholder, gallery_placeholder)
        # Rerun to refresh the UI and show the final image properly
        st.rerun()

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
