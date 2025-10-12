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
import hashlib
import gc
# (no additional top-level imports required)

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
        # How many images to generate per internal batch (affects VRAM)
        "batch_size": 1,
        
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
    "ui_scale": 1.0,
        "sidebar_collapsed_by_default": False,
    }

# ============================================================================
# Settings Persistence
# ============================================================================

SETTINGS_FILE = "./webui_settings.json"
HISTORY_FILE = "./webui_history.json"

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

def load_history():
    """Load generation history from disk"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # Sanitize any saved seed fields that look like tensor dumps.
                changed = False
                for e in saved:
                    if isinstance(e, dict):
                        if 'seed' in e:
                            sanitized = sanitize_seed_for_display(e.get('seed'))
                            if sanitized != e.get('seed'):
                                e['seed'] = sanitized
                                changed = True
                        png_meta = e.get('png_metadata') or {}
                        if isinstance(png_meta, dict) and 'seed' in png_meta:
                            sanitized_png_seed = sanitize_seed_for_display(png_meta.get('seed'))
                            if sanitized_png_seed != png_meta.get('seed'):
                                png_meta['seed'] = sanitized_png_seed
                                e['png_metadata'] = png_meta
                                changed = True
                # If we cleaned anything, persist the cleaned history back to disk
                if changed:
                    try:
                        save_history(saved)
                    except Exception:
                        pass
                return saved
        except Exception as e:
            st.warning(f"Could not load history: {e}")
    return []

def save_history(history):
    """Save generation history to disk"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Could not save history: {e}")

def sanitize_seed_for_display(seed_value):
    """Return a safe seed string or None if the value looks like a tensor/image dump.

    This avoids storing or displaying very large tensor representations that
    were accidentally written by earlier versions of the adetailer pipeline.
    """
    if seed_value is None:
        return None
    if isinstance(seed_value, (int, float)):
        return str(int(seed_value))
    if isinstance(seed_value, str):
        s = seed_value.strip()
        # Heuristics: if it contains literal 'tensor' or multiline dumps/brackets,
        # treat it as invalid.
        if "tensor(" in s.lower() or "[" in s or "\n" in s or len(s) > 240:
            return None
        return s
    return None

def add_to_history(image_paths, settings):
    """Add generated images to history"""
    history = load_history()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            # Read any embedded PNG metadata
            png_meta = {}
            try:
                with Image.open(img_path) as _img:
                    png_meta = getattr(_img, "info", {}) or {}
            except Exception:
                png_meta = {}

            # Ensure seed_meta is always defined
            seed_meta = None
            # Sanitize seed metadata to avoid storing huge tensor dumps
            seed_meta = sanitize_seed_for_display(png_meta.get("seed"))
            # Also sanitize the nested png_meta so the stored history file
            # does not contain enormous tensor dumps in the png_metadata field.
            try:
                png_meta["seed"] = sanitize_seed_for_display(png_meta.get("seed"))
            except Exception:
                pass

            # Prefer prompt text embedded in the image's PNG metadata when
            # available (this is the authoritative prompt used by the
            # pipeline). Fall back to the provided settings prompt.
            png_prompt = None
            png_negative = None
            try:
                png_prompt = png_meta.get("prompt")
            except Exception:
                png_prompt = None
            try:
                png_negative = png_meta.get("negative_prompt")
            except Exception:
                png_negative = None

            entry = {
                "timestamp": timestamp,
                "image_path": img_path,
                "prompt": png_prompt if png_prompt not in (None, "") else settings.get("prompt", ""),
                "negative_prompt": png_negative if png_negative not in (None, "") else settings.get("negative_prompt", ""),
                "width": settings.get("width", None),
                "height": settings.get("height", None),
                "batch_size": settings.get("batch_size"),
                "flux_mode": settings.get("flux_mode", False),
                "realistic_mode": settings.get("realistic_mode", False),
                # Add extracted PNG metadata fields if present (sanitized)
                "seed": seed_meta,
                "sampler": png_meta.get("sampler"),
                "steps": png_meta.get("steps"),
                # Generation timing metadata (may be added after generation)
                "generation_duration": None,
                "avg_iters_per_s": None,
                "cfg": png_meta.get("cfg"),
                "scheduler": png_meta.get("scheduler"),
                "denoise": png_meta.get("denoise"),
                "png_metadata": png_meta,
            }
            # Parse timing metadata from PNG chunk if present
            try:
                gd = png_meta.get("generation_duration")
                if gd is not None:
                    try:
                        entry["generation_duration"] = float(gd)
                    except Exception:
                        # If it has a trailing 's' or is a string, try to clean
                        try:
                            entry["generation_duration"] = float(str(gd).rstrip('s'))
                        except Exception:
                            entry["generation_duration"] = None
            except Exception:
                pass
            try:
                ai = png_meta.get("avg_iters_per_s")
                if ai is not None:
                    try:
                        entry["avg_iters_per_s"] = float(ai)
                    except Exception:
                        try:
                            entry["avg_iters_per_s"] = float(str(ai).rstrip('s'))
                        except Exception:
                            entry["avg_iters_per_s"] = None
            except Exception:
                pass
            history.insert(0, entry)  # Add to beginning
    
    # Keep only last 100 entries
    history = history[:100]
    save_history(history)

def clear_history():
    """Clear all history and delete all tracked image files"""
    history = load_history()
    
    # Delete all image files
    for entry in history:
        img_path = entry.get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                st.warning(f"Could not delete {os.path.basename(img_path)}: {e}")
    
    # Clear history
    save_history([])

def scan_output_folders():
    """Scan all output folders and build history from existing images"""
    output_dirs = [
        "./output/Classic",
        "./output/Flux",
        "./output/HiresFix",
        "./output/Img2Img",
        "./output/Adetailer"
    ]
    
    all_images = []
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            images = glob.glob(f"{output_dir}/*.png")
            all_images.extend(images)
    
    # Sort by modification time (newest first)
    all_images = sorted(all_images, key=os.path.getmtime, reverse=True)
    
    # Load existing history to preserve metadata
    existing_history = load_history()
    existing_paths = {entry["image_path"]: entry for entry in existing_history}
    
    # Build new history
    new_history = []
    for img_path in all_images[:100]:  # Keep only last 100
        if img_path in existing_paths:
            # Use existing entry
            new_history.append(existing_paths[img_path])
        else:
            # Create new entry with minimal info
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(img_path)))
                with Image.open(img_path) as img:
                    width, height = img.size
                    png_meta = getattr(img, "info", {}) or {}

                entry = {
                    "timestamp": timestamp,
                    "image_path": img_path,
                    "prompt": png_meta.get("prompt", "(prompt not available)"),
                    "negative_prompt": png_meta.get("negative_prompt", ""),
                    "width": width,
                    "height": height,
                    "batch_size": png_meta.get("batch_size"),
                    "flux_mode": png_meta.get("flux_enabled", False) or ("Flux" in img_path),
                    "realistic_mode": png_meta.get("realistic_model", False),
                    # Add commonly useful png metadata fields
                    "seed": sanitize_seed_for_display(png_meta.get("seed")),
                    "sampler": png_meta.get("sampler"),
                    "steps": png_meta.get("steps"),
                    "generation_duration": None,
                    "avg_iters_per_s": None,
                    "cfg": png_meta.get("cfg"),
                    "scheduler": png_meta.get("scheduler"),
                    "denoise": png_meta.get("denoise"),
                    "png_metadata": png_meta,
                }
                # Parse timing metadata if present
                try:
                    gd = png_meta.get("generation_duration")
                    if gd is not None:
                        try:
                            entry["generation_duration"] = float(gd)
                        except Exception:
                            try:
                                entry["generation_duration"] = float(str(gd).rstrip('s'))
                            except Exception:
                                entry["generation_duration"] = None
                except Exception:
                    pass
                try:
                    ai = png_meta.get("avg_iters_per_s")
                    if ai is not None:
                        try:
                            entry["avg_iters_per_s"] = float(ai)
                        except Exception:
                            try:
                                entry["avg_iters_per_s"] = float(str(ai).rstrip('s'))
                            except Exception:
                                entry["avg_iters_per_s"] = None
                except Exception:
                    pass
                new_history.append(entry)
            except Exception:
                pass
    
    save_history(new_history)
    return new_history

def delete_history_entry(entry_index):
    """Delete a history entry and its associated image file"""
    history = load_history()
    if 0 <= entry_index < len(history):
        entry = history[entry_index]
        img_path = entry["image_path"]
        
        # Delete the actual image file
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                st.error(f"Could not delete image file: {e}")
        
        # Remove from history
        history.pop(entry_index)
        save_history(history)
        return True
    return False

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize all required session state variables"""
    
    # Load settings
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    
    # UI State
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
        # Backwards compatibility: keep field but prefer paths below
        st.session_state.generated_images = []
    if "generated_image_paths" not in st.session_state:
        st.session_state.generated_image_paths = []
    if "display_size" not in st.session_state:
        st.session_state.display_size = (512, 512)
    if "ui_display_size" not in st.session_state:
        default_scale = load_settings().get("ui_scale", 1.0)
        st.session_state.ui_display_size = (min(int(512 * default_scale), 1400), min(int(512 * default_scale), 1000))
    
    # Page State
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Generate"

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
    /* Status bar background and text chosen for strong contrast */
    --ld-status-bg: {'rgba(0,0,0,0.35)' if theme == 'dark' else 'rgba(0,0,0,0.28)'};
    --ld-status-text: {'#ffffff' if theme == 'dark' else '#ffffff'};
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
    /* Compact fixed status bar shown during generation (small, non-intrusive) */
    .ld-status-bar {{
        position: fixed;
        bottom: 12px;
        left: 50%;
        transform: translateX(-50%);
        background-color: var(--ld-status-bg);
        color: var(--ld-status-text);
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 0.92rem;
        z-index: 9999;
        box-shadow: 0 6px 18px rgba(0,0,0,0.28);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }}
    /* Auto-hide animation for final/status messages */
    .ld-status-bar.auto-hide {{
        animation: ld-fadeout 0.9s ease-in-out forwards;
        animation-delay: 8s; /* visible for 8 seconds, then fade */
    }}
    @keyframes ld-fadeout {{
        0% {{ opacity: 1; transform: translateY(0px); visibility: visible; }}
        100% {{ opacity: 0; transform: translateY(8px); visibility: hidden; }}
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

def generate_images(settings, status_placeholder, gallery_placeholder, status_bar=None):
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
    # If the user is attempting a large job, give them a heads-up that
    # the app may temporarily unload models between chunks to avoid
    # running out of VRAM. The actual unload happens inside the
    # background worker so we only display the notice here.
    try:
        if settings.get("num_images", 1) >= 10 and settings.get("keep_models_loaded", True):
            try:
                status_placeholder.info("⚠️ Large job detected — the app will unload models between chunks to reduce VRAM usage.")
            except Exception:
                pass
    except Exception:
        pass
    
    # Create output directories
    os.makedirs("./output/preview_display", exist_ok=True)
    
    # Background generation thread
    pipeline_result = None
    generation_complete = threading.Event()
    generation_error = None
    
    def run_generation():
        """Run the pipeline in small chunks to avoid VRAM accumulation.

        This routine intentionally does not call Streamlit APIs (they are
        not thread-safe). It will set pipeline_result/generation_error and
        signal the completion event so the main thread can react.
        """
        nonlocal pipeline_result, generation_error
        # Keep a local copy of the user's preference so we can restore it.
        original_keep_models = bool(settings.get("keep_models_loaded", True))
        forced_unload_for_large_job = False

        try:
            total_images = int(settings.get("num_images", 1))
            configured_batch = int(settings.get("batch_size", 1))

            # For very large jobs, force unloading models between chunks to
            # avoid gradual VRAM growth in some backends. We do not persist
            # this change to the saved settings; it only applies during this
            # generation run.
            LARGE_JOB_MIN_IMAGES = 10
            if total_images >= LARGE_JOB_MIN_IMAGES and original_keep_models:
                try:
                    set_keep_models_loaded(False)
                    forced_unload_for_large_job = True
                except Exception:
                    # If the backend refuses, continue but try other mitigations
                    forced_unload_for_large_job = False

            # Prepare multiscale params once
            if settings.get("multiscale_custom"):
                multiscale_params = {
                    "multiscale_preset": None,
                    "enable_multiscale": True,
                    "multiscale_factor": settings.get("multiscale_factor", 0.5),
                    "multiscale_fullres_start": settings.get("multiscale_fullres_start", 3),
                    "multiscale_fullres_end": settings.get("multiscale_fullres_end", 8),
                    "multiscale_intermittent_fullres": settings.get("multiscale_intermittent_fullres", False),
                }
            else:
                multiscale_params = {"multiscale_preset": settings.get("multiscale_preset", "balanced")}

            images_generated = 0
            last_error = None

            def _attempt_memory_recovery(force_clear=False):
                """Try a few cheap memory recovery steps: clear previews, GC,
                clear model cache (optionally) and empty CUDA cache if available.
                If force_clear=True the model cache will be cleared unconditionally;
                otherwise we consult VRAM usage and only clear when it looks high.
                """
                try:
                    # Close any preview files the app created
                    try:
                        app_instance.app.cleanup_all_previews()
                    except Exception:
                        pass

                    gc.collect()

                    # Clear model cache only when explicitly requested or when
                    # VRAM utilization is high. This avoids unnecessary reloads
                    # between small chunks while still protecting long runs.
                    try:
                        if force_clear:
                            clear_model_cache()
                        else:
                            try:
                                mem_info = get_memory_info()
                                used = float(mem_info.get('used_vram', 0) or 0)
                                total = float(mem_info.get('total_vram', 0) or 0)
                                if total > 0 and (used / total) >= 0.85:
                                    clear_model_cache()
                            except Exception:
                                # If we can't read memory info, avoid clearing
                                # cache unless explicitly forced.
                                pass
                    except Exception:
                        pass

                    try:
                        import torch
                        if getattr(torch, "cuda", None) and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        # Torch not available or cuda not present
                        pass

                    # Small delay to give drivers time to reclaim memory
                    time.sleep(0.15)
                except Exception:
                    pass

            # Generate in chunks to avoid holding onto intermediate RAM across
            # many generated images. We try progressively smaller chunk sizes
            # if an OOM occurs.
            while images_generated < total_images and not stop_event.is_set():
                remaining = total_images - images_generated
                target_chunk = min(configured_batch, remaining)
                attempt_chunk = target_chunk

                # If the user explicitly asked to keep models loaded, we still
                # try using smaller chunks for stability.
                while attempt_chunk > 0 and not stop_event.is_set():
                    try:
                        result = pipeline(
                            prompt=settings.get("prompt", ""),
                            negative_prompt=settings.get("negative_prompt", ""),
                            w=settings.get("width"),
                            h=settings.get("height"),
                            number=attempt_chunk,
                            batch=min(attempt_chunk, configured_batch),
                            hires_fix=settings.get("hiresfix", False),
                            adetailer=settings.get("adetailer", False),
                            enhance_prompt=settings.get("enhance_prompt", False),
                            img2img=settings.get("img2img_mode", False),
                            stable_fast=settings.get("stable_fast", False),
                            reuse_seed=settings.get("reuse_seed", False),
                            flux_enabled=settings.get("flux_mode", False),
                            prio_speed=settings.get("speed_mode", False),
                            autohdr=True,
                            realistic_model=settings.get("realistic_mode", False),
                            img2img_image=settings.get("input_image_path") if settings.get("img2img_mode", False) else None,
                            **multiscale_params,
                        )

                        # We don't keep large pipeline return values around —
                        # just note that a result existed and then free it so
                        # memory can be reclaimed.
                        pipeline_result = True
                        try:
                            del result
                        except Exception:
                            pass

                        images_generated += attempt_chunk

                        # Aggressively try to free any leftover memory between
                        # chunks so long-running generation runs remain stable.
                        # For routine between-chunk cleanup only force a
                        # full model unload when the user selected the large
                        # job mitigation; otherwise be conservative.
                        _attempt_memory_recovery(force_clear=forced_unload_for_large_job)
                        break
                    except Exception as e:
                        last_error = e
                        # Heuristic: treat messages mentioning OOM/CUDA as memory
                        # exhaustion and try to recover by shrinking chunk size.
                        msg = str(e).lower() if e is not None else ""
                        is_oom = (
                            "out of memory" in msg
                            or "cuda" in msg and ("out" in msg or "oom" in msg)
                            or isinstance(e, MemoryError)
                        )

                        if not is_oom:
                            # Not a memory error — rethrow and let outer handler
                            # capture it and end the run.
                            raise

                        # Try to recover from OOM: shrink the attempted chunk
                        # size and clear caches before retrying.
                        new_attempt = max(1, attempt_chunk // 2)
                        if new_attempt >= attempt_chunk:
                            new_attempt = 1
                        attempt_chunk = new_attempt

                        # On OOM we want to aggressively clear caches before
                        # retrying smaller batches.
                        _attempt_memory_recovery(force_clear=True)

                        # If we've reduced to single-image chunks and still OOM,
                        # try one last time with a full cache clear and then
                        # propagate the failure if it still fails.
                        if attempt_chunk == 1:
                            try:
                                clear_model_cache()
                            except Exception:
                                pass
                            gc.collect()
                            try:
                                import torch as _torch
                                if getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                                    _torch.cuda.empty_cache()
                            except Exception:
                                pass

                            # Next loop iteration will try a single-image run.

                # End inner attempt loop

                # If we exited the inner loop due to an interrupt, break outer loop
                if stop_event.is_set():
                    break

            # If the inner logic ended with an exception, surface it
            if last_error is not None and images_generated < total_images:
                raise last_error

        except Exception as e:
            generation_error = e
            if settings.get("verbose_mode"):
                import traceback
                traceback.print_exc()
        finally:
            # Restore user's preferred "keep models loaded" flag if we
            # temporarily overrode it for large jobs.
            try:
                if forced_unload_for_large_job and original_keep_models:
                    set_keep_models_loaded(original_keep_models)
            except Exception:
                pass

            generation_complete.set()
    
    # Create a thread-safe stop event for the background worker so the
    # main thread can request a clean stop without touching Streamlit
    # session state from the worker thread.
    stop_event = threading.Event()

    # Start generation thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    # Track in session state
    st.session_state.generation_job = {
        "thread": gen_thread,
        "complete_event": generation_complete,
        "stop_event": stop_event,
        "start_time": time.time()
    }
    
    # Compute display size (base) and a larger UI display size used for both
    # live previews and final output. The UI size is independent from the
    # raw image pixels and intentionally scaled up for improved visibility.
    display_size = compute_display_size(
        settings["width"], 
        settings["height"],
        max_width=800,
        max_height=600
    )
    st.session_state.display_size = display_size

    # UI scale factor (user-configurable) and clamping maximums
    UI_SCALE = float(settings.get("ui_scale", 1.0))
    UI_MAX_WIDTH = 1400
    UI_MAX_HEIGHT = 1000
    ui_full_w = min(int(display_size[0] * UI_SCALE), UI_MAX_WIDTH)
    ui_full_h = min(int(display_size[1] * UI_SCALE), UI_MAX_HEIGHT)
    # Store for potential use elsewhere
    st.session_state.ui_display_size = (ui_full_w, ui_full_h)
    
    # Preview update loop
    preview_container = gallery_placeholder.empty() if settings["enable_preview"] else None
    last_preview_time = 0
    
    while not generation_complete.wait(0.2):
        # Check for interrupt
        if st.session_state.interrupt_generation:
            app_instance.app.request_interrupt()
            try:
                if status_bar is not None:
                    status_bar.markdown("<div class=\"ld-status-bar auto-hide\">⏹️ Stopping generation...</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.warning("⏹️ Stopping generation...")
            except Exception:
                status_placeholder.warning("⏹️ Stopping generation...")
            break
        
        # Update preview
        if preview_container and settings["enable_preview"]:
            current_time = time.time()
            if current_time - last_preview_time > 0.5:  # Update every 0.5s
                previews = app_instance.app.get_latest_previews()
                if previews:
                    try:
                        # Show a tiled set of recent previews (up to 6)
                        recent = previews[-6:]
                        cols_count = min(3, len(recent)) or 1
                        with preview_container.container():
                            cols = st.columns(cols_count)
                            for i, pth in enumerate(recent):
                                try:
                                    img_prev = Image.open(pth)
                                    # Compute per-column tile size so previews and
                                    # outputs render at the same visual size.
                                    cols_for_preview = cols_count or 1
                                    tile_w = max(64, int(ui_full_w / cols_for_preview))
                                    # maintain aspect ratio from the base display_size
                                    tile_h = max(64, int(tile_w * (display_size[1] / (display_size[0] or 1))))
                                    render_responsive_image(img_prev, (tile_w, tile_h), cols[i % cols_count])
                                except Exception:
                                    # Ignore preview rendering errors
                                    pass
                        last_preview_time = current_time
                    except Exception:
                        pass
        
        # Update compact status bar (avoid large banners)
        elapsed = time.time() - st.session_state.generation_job["start_time"]
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar\">🎨 Generating — {elapsed:.1f}s</div>", unsafe_allow_html=True)
            else:
                status_placeholder.info(f"🎨 Generating... ({elapsed:.1f}s)")
        except Exception:
            # Fallback to the original banner if status bar update fails
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
        short_err = str(generation_error)
        if len(short_err) > 120:
            short_err = short_err[:117] + "..."
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">❌ Generation failed: {short_err}</div>", unsafe_allow_html=True)
            else:
                status_placeholder.error(f"❌ Generation failed: {generation_error}")
        except Exception:
            status_placeholder.error(f"❌ Generation failed: {generation_error}")
        return []
    
    # Find generated images (check all possible output directories)
    generated_image_paths = []
    
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
    
    # Select outputs produced during this generation job by using the
    # recorded generation start time. This reliably collects all files
    # produced by the job (including multi-file outputs like adetailer
    # head/body) without depending on filename heuristics.
    if all_outputs:
        all_outputs = sorted(all_outputs, key=os.path.getmtime, reverse=True)
        job_start = None
        try:
            job_start = st.session_state.generation_job.get("start_time")
        except Exception:
            job_start = None

        now_time = time.time()
        if job_start:
            # Gather files modified during the job window. Use a small
            # tolerance before start in case some outputs were written
            # immediately at the start timestamp.
            job_window = [f for f in all_outputs if os.path.getmtime(f) >= (job_start - 2.0) and os.path.getmtime(f) <= (now_time + 2.0)]
        else:
            job_window = []

        # If we found files that match the job window, use them; otherwise
        # fall back to taking the most recent N files.
        if job_window:
            generated_image_paths.extend(sorted(job_window, key=os.path.getmtime))
        else:
            # Fallback: show the most recent files up to the requested count
            for f in all_outputs[: settings.get("num_images", 1)]:
                if os.path.exists(f):
                    generated_image_paths.append(f)
                else:
                    if settings["verbose_mode"]:
                        status_placeholder.warning(f"File disappeared before collection: {f}")
    
    if generated_image_paths:
        # Save to session state as paths (safe to persist)
        st.session_state.generated_image_paths = generated_image_paths
        st.session_state.display_size = display_size

        # Compute elapsed time for the job
        job_elapsed = time.time() - st.session_state.generation_job.get("start_time", time.time())

        # Attempt to write generation duration and average iterations/s into
        # each generated image's PNG metadata so history and files contain
        # this useful diagnostic information.
        try:
            from PIL.PngImagePlugin import PngInfo
            for path in generated_image_paths:
                try:
                    with Image.open(path) as im:
                        png_meta = getattr(im, "info", {}) or {}

                        # Try to extract steps from existing metadata
                        steps_val = png_meta.get("steps") or png_meta.get("step")
                        steps = None
                        try:
                            if steps_val is not None:
                                steps = int(float(steps_val))
                        except Exception:
                            steps = None

                        avg_iters = None
                        if steps is not None and job_elapsed > 0:
                            avg_iters = steps / job_elapsed

                        # Build new PNG info while preserving existing text keys
                        pnginfo = PngInfo()
                        for k, v in (png_meta.items() if isinstance(png_meta, dict) else []):
                            try:
                                pnginfo.add_text(str(k), str(v))
                            except Exception:
                                pass

                        pnginfo.add_text("generation_duration", f"{job_elapsed:.3f}")
                        if avg_iters is not None:
                            pnginfo.add_text("avg_iters_per_s", f"{avg_iters:.3f}")
                        else:
                            pnginfo.add_text("avg_iters_per_s", "unknown")

                        # Re-save image with the updated metadata
                        try:
                            im.save(path, pnginfo=pnginfo)
                        except Exception:
                            # Some images may be read-only or locked — ignore write errors
                            if settings.get("verbose_mode"):
                                status_placeholder.warning(f"Could not persist metadata to {os.path.basename(path)}")
                except Exception:
                    if settings.get("verbose_mode"):
                        status_placeholder.warning(f"Failed updating metadata for {path}")
        except Exception:
            # PIL PngInfo not available or another failure — skip metadata writing
            if settings.get("verbose_mode"):
                status_placeholder.warning("Could not attach generation metadata to images")

        # Add to history (reads metadata back from files)
        add_to_history(generated_image_paths, settings)

        # Display tiled gallery of all images in the same placeholder so the
        # user immediately sees the whole batch (no extra rerun required).
        try:
            # Render final images using the same per-column tile sizing so
            # previews and final output visually match. For a single image
            # we use the full UI size for maximum visibility.
            with gallery_placeholder.container():
                cols_count = min(3, len(generated_image_paths)) or 1
                cols = st.columns(cols_count)
                for idx, path in enumerate(generated_image_paths):
                    try:
                        img = Image.open(path)
                        # If we only have one image, render at the full UI size.
                        if len(generated_image_paths) == 1:
                            tile_w, tile_h = ui_full_w, ui_full_h
                        else:
                            tile_w = max(64, int(ui_full_w / cols_count))
                            tile_h = max(64, int(tile_w * (display_size[1] / (display_size[0] or 1))))

                        # Use the responsive renderer so final images match preview size
                        try:
                            render_responsive_image(img, (tile_w, tile_h), cols[idx % cols_count])
                            cols[idx % cols_count].caption(f"Image {idx+1}")
                        except Exception:
                            # Fallback to Streamlit image if something goes wrong
                            with cols[idx % cols_count]:
                                st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                    except Exception as e:
                        if settings["verbose_mode"]:
                            status_placeholder.warning(f"Error rendering {path}: {e}")
        except Exception:
            # Fall back to rendering only the first image if tiled gallery fails
            try:
                first_img = Image.open(generated_image_paths[0])
                # If only one image, use the full UI size; otherwise use per-column
                if len(generated_image_paths) == 1:
                    render_responsive_image(first_img, (ui_full_w, ui_full_h), gallery_placeholder)
                else:
                    fallback_tile_w = max(64, int(ui_full_w / min(3, len(generated_image_paths))))
                    fallback_tile_h = max(64, int(fallback_tile_w * (display_size[1] / (display_size[0] or 1))))
                    render_responsive_image(first_img, (fallback_tile_w, fallback_tile_h), gallery_placeholder)
            except Exception:
                pass

        elapsed = time.time() - st.session_state.generation_job["start_time"]
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">✅ Generated {len(generated_image_paths)} image(s) — {elapsed:.1f}s</div>", unsafe_allow_html=True)
            else:
                status_placeholder.success(f"✅ Generated {len(generated_image_paths)} image(s) in {elapsed:.1f}s")
        except Exception:
            status_placeholder.success(f"✅ Generated {len(generated_image_paths)} image(s) in {elapsed:.1f}s")
        # Return paths for callers that might use them (not used currently)
        return generated_image_paths
    else:
        # Show a helpful message with debug info
        if pipeline_result:
            checked_dirs = ", ".join(primary_dirs)
            try:
                if status_bar is not None:
                    status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">⚠️ Completed but no images found (checked: {checked_dirs})</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.warning(f"⚠️ Generation completed but no images found. Checked: {checked_dirs}")
            except Exception:
                status_placeholder.warning(f"⚠️ Generation completed but no images found. Checked: {checked_dirs}")
        else:
            try:
                if status_bar is not None:
                    status_bar.markdown("<div class=\"ld-status-bar auto-hide\">Generation stopped</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.info("Generation stopped")
            except Exception:
                status_placeholder.info("Generation stopped")
        return []

def stop_generation():
    """Request generation stop"""
    # Signal via session state (used by the UI loop) and also immediately
    # notify the app-level interrupt event so sampling loops can react
    # without waiting for another rerun / preview update.
    st.session_state.interrupt_generation = True
    try:
        app_instance.app.request_interrupt()
    except Exception:
        # If app instance isn't available for some reason, ignore and
        # rely on session state mechanism.
        pass
    # Wake any waiting preview loop and mark generation as stopped so the
    # UI re-enables controls immediately. The background thread may still
    # be cleaning up; we intentionally re-enable the Generate button so the
    # user can start a new job if desired (forceful stop behavior).
    try:
        job = st.session_state.get("generation_job")
        if job and isinstance(job, dict):
            ev = job.get("complete_event")
            if ev is not None and hasattr(ev, "set"):
                ev.set()
            # Signal the worker thread via the thread-safe stop_event if present
            stop_ev = job.get("stop_event")
            if stop_ev is not None and hasattr(stop_ev, "set"):
                try:
                    stop_ev.set()
                except Exception:
                    pass
    except Exception:
        pass

    # Re-enable controls immediately to give the user responsive feedback.
    st.session_state.is_generating = False
    # Also ensure we do not re-trigger generation accidentally and clear any
    # stale job state so the UI shows Generate enabled right away.
    try:
        st.session_state.start_generation = False
        # Mark the current generation job as aborted so other logic can
        # detect that a background cleanup thread may still be running.
        job = st.session_state.get("generation_job")
        if job and isinstance(job, dict):
            job["aborted"] = True
            st.session_state.generation_job = job
    except Exception:
        pass

    # Force a rerun so the UI immediately reflects the new state.
    try:
        st.rerun()
    except Exception:
        # If rerun fails (e.g. called from a context where rerun is not allowed),
        # ignore and let the next user interaction refresh the UI.
        pass


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
            # App title removed during initialization; keep status message only
            st.info(setup_status.get("message", "Initializing..."))
            progress = setup_status.get("progress", 0.0)
            st.progress(progress)
            time.sleep(0.5)
            st.rerun()
            return
    
    # ========================================================================
    # Header & Navigation
    # ========================================================================
    
    # App title removed from main header to provide a cleaner UI
    
    # Page tabs
    tab1, tab2 = st.tabs(["🎨 Generate", "📜 History"])
    
    # ========================================================================
    # Generate Tab
    # ========================================================================
    
    with tab1:
        render_generate_page()
    
    # ========================================================================
    # History Tab
    # ========================================================================
    
    with tab2:
        render_history_page()

def render_generate_page():
    """Render the main generation page"""
    
    settings = st.session_state.settings
    # Flag used to disable interactive controls while generation is running
    controls_disabled = st.session_state.is_generating

    
    with st.sidebar:
        # App title in sidebar with link to GitHub
        st.markdown(
            '<a href="https://github.com/Aatricks/LightDiffusion-Next" target="_blank" style="text-decoration:none;color:inherit;"><h2 style="margin:0 0 8px 0;">LightDiffusion</h2></a>',
            unsafe_allow_html=True,
        )
        st.header("⚙️ Settings")
        # Disable controls while a generation is active to avoid
        # user-driven reruns that can desync the UI thread and the
        # background generation thread. The Stop button remains active
        # so the user can interrupt generation. The `controls_disabled`
        # flag is defined at the top of this function so it can also be
        # used by the Help button in the main layout.
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
                max_value=1000,
                value=settings["num_images"],
                key="num_images_input",
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
            
            # Batch size control (number of images processed per internal batch)
            settings["batch_size"] = st.number_input(
                "Batch Size (images per batch)",
                min_value=1,
                max_value=10,
                value=settings.get("batch_size", 1),
                key="batch_size_input",
                disabled=controls_disabled,
                help="Number of images processed together per internal batch. Higher values use more VRAM but can be faster.",
            )
            # Batch size is independent from Number of Images — users may
            # choose to process multiple images per internal batch even if
            # the total requested images is smaller. The pipeline will
            # respect both values appropriately.
        
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
                        st.image(uploaded_file, caption="Input Image", use_container_width=True)
        
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
        # UI display scale
        settings["ui_scale"] = st.slider(
            "UI Display Scale",
            min_value=0.5,
            max_value=3.0,
            value=settings.get("ui_scale", 1.0),
            step=0.25,
            help="Scale factor applied to preview and output display size (independent of image resolution).",
            disabled=controls_disabled,
        )
        # Update computed ui_display_size when user changes scale in the sidebar
        try:
            scale_val = float(settings["ui_scale"])
            base_display = st.session_state.get("display_size", (512, 512))
            st.session_state.ui_display_size = (min(int(base_display[0] * scale_val), 1400), min(int(base_display[1] * scale_val), 1000))
        except Exception:
            pass
    
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
    # Small fixed status bar placeholder (used for compact timer/messages)
    status_bar = st.empty()
    
    # Show existing images if any (paths stored in session state)
    if st.session_state.generated_image_paths and not st.session_state.is_generating:
        display_size = st.session_state.display_size
        ui_full_w, ui_full_h = st.session_state.get(
            "ui_display_size",
            (
                min(int(display_size[0] * float(settings.get("ui_scale", 1.0))), 1400),
                min(int(display_size[1] * float(settings.get("ui_scale", 1.0))), 1000),
            ),
        )

        # Show tiled gallery of all generated images (tiles sized to match preview)
        paths = st.session_state.generated_image_paths
        cols_count = min(3, len(paths)) or 1
        cols = st.columns(cols_count)
        for idx, path in enumerate(paths):
            try:
                with open(path, "rb") as f:
                    img = Image.open(f)
                    if len(paths) == 1:
                        tile_w, tile_h = ui_full_w, ui_full_h
                    else:
                        tile_w = max(64, int(ui_full_w / cols_count))
                        tile_h = max(64, int(tile_w * (display_size[1] / (display_size[0] or 1))))
                    with cols[idx % cols_count]:
                        render_responsive_image(img, (tile_w, tile_h))
            except Exception as e:
                with cols[idx % cols_count]:
                    st.warning(f"Could not load image: {e}")
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

        # If a previous generation thread is still cleaning up, refuse to
        # start a new generation immediately to avoid running two jobs
        # concurrently. The Stop action sets `is_generating=False` to
        # re-enable the Generate button, but the underlying thread may
        # still be terminating; detect and warn instead of starting.
        prev_job = st.session_state.get("generation_job") or {}
        prev_thread = prev_job.get("thread") if isinstance(prev_job, dict) else None
        if prev_thread is not None and hasattr(prev_thread, "is_alive") and prev_thread.is_alive():
            status_placeholder.warning(
                "⚠️ Previous generation is still stopping. Please wait a moment and try Generate again."
            )
            # Don't start a new generation while the old thread is alive
            st.session_state.start_generation = False
        else:
            # Clear previous images but keep display size
            st.session_state.generated_images = []
            st.session_state.generated_image_paths = []
            # Start generation (this will update placeholders/live preview)
            generate_images(settings, status_placeholder, gallery_placeholder, status_bar)
            # Rerun to refresh the UI and show the final image properly
            st.rerun()

def render_history_page():
    """Render the history page with past generations"""
    
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
    
    # Load history
    history = load_history()
    
    with col3:
        st.text(f"Total: {len(history)}")
    
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
                        st.image(img, use_container_width=True)
                        
                        # Compact info
                        with st.expander("ℹ️ Details", expanded=False):
                            st.text(f"🕒 {entry.get('timestamp')}")
                            st.text(f"📐 {entry.get('width')}x{entry.get('height')}")
                            batch = entry.get("batch_size")
                            if batch is not None:
                                st.text(f"🔁 Batch: {batch}")
                            
                            # Key metadata
                            seed = entry.get("seed")
                            sampler = entry.get("sampler")
                            steps = entry.get("steps")
                            cfg = entry.get("cfg")
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
                            
                            if entry.get('flux_mode'):
                                st.text("⚡ Flux Mode")
                            if entry.get('realistic_mode'):
                                st.text("📸 Realistic")
                            
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
                                        # Create a stable unique widget key using the
                                        # entry index and a short hash of the path so
                                        # keys cannot collide with other download
                                        # buttons elsewhere in the app.
                                        hist_key = hashlib.md5(img_path.encode('utf-8')).hexdigest()[:8]
                                        st.download_button(
                                            label="💾",
                                            data=f,
                                            file_name=os.path.basename(img_path),
                                            mime="image/png",
                                            key=f"download_history_{entry_idx}_{hist_key}",
                                            use_container_width=True
                                        )
                            with col_del:
                                if st.button("🗑️", key=f"delete_{entry_idx}", use_container_width=True):
                                    if delete_history_entry(entry_idx):
                                        st.rerun()
                            # All metadata expander (minimalistic)
                            with st.expander("🧾 All metadata", expanded=False):
                                # Combine top-level and png metadata for inspection
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

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
