"""
LightDiffusion Streamlit UI
A minimalistic, single-mode web interface focused on the generated image.
All controls are in a togglable sidebar, main canvas displays preview/final at same size.
"""

import streamlit as st
import threading
import time
from ui import settings as ui_settings
from ui import history as ui_history
from ui import helpers as ui_helpers
from ui.pages import render_history_page, render_generate_page

# Core Pipeline Integration
from src.user.pipeline import pipeline
from src.user import app_instance
# Model cache helpers are imported by the UI modules that need them.
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
    """Default settings (delegates to ui.settings).

    The full schema lives in `ui.settings.get_default_settings`.
    """
    return ui_settings.get_default_settings()

# ============================================================================
# Settings Persistence
# ============================================================================

# SETTINGS_FILE and HISTORY_FILE have been moved to ui.settings/ui.history

def load_settings():
    """Load settings from disk (delegates to ui.settings)."""
    return ui_settings.load_settings()

def save_settings(settings):
    """Persist settings (delegates to ui.settings)."""
    return ui_settings.save_settings(settings)

def load_history():
    """Load generation history (delegates to ui.history)."""
    return ui_history.load_history()

def save_history(history):
    """Persist generation history (delegates to ui.history)."""
    return ui_history.save_history(history)

def sanitize_seed_for_display(seed_value):
    """Delegate sanitization to ui.history.sanitize_seed_for_display."""
    return ui_history.sanitize_seed_for_display(seed_value)

def add_to_history(image_paths, settings):
    """Delegate add_to_history to ui.history.add_to_history."""
    return ui_history.add_to_history(image_paths, settings)

def clear_history():
    """Delegate clear_history to ui.history.clear_history."""
    return ui_history.clear_history()

def scan_output_folders():
    """Delegate scanning of output folders to ui.history.scan_output_folders."""
    return ui_history.scan_output_folders()

def delete_history_entry(entry_index):
    """Delegate deletion to ui.history.delete_history_entry."""
    return ui_history.delete_history_entry(entry_index)

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
    """Inject custom CSS for theming and responsive images.

    Delegates to `ui.helpers.inject_custom_css` so styling is defined in
    a single place and easier to test.
    """
    try:
        ui_helpers.inject_custom_css(st.session_state.dark_mode)
    except Exception:
        # Best-effort: styling is non-critical so swallow errors here.
        pass

# ============================================================================
# Image Display Helpers
# ============================================================================

def compute_display_size(image_width, image_height, max_width=800, max_height=600):
    """Delegate computation to ui.helpers.compute_display_size."""
    return ui_helpers.compute_display_size(image_width, image_height, max_width=max_width, max_height=max_height)

@st.cache_data
def image_to_base64(image, format="PNG"):
    """Delegate to ui.helpers.image_to_base64 (cached)."""
    return ui_helpers.image_to_base64(image, format=format)

def render_responsive_image(image, target_display_size, placeholder=None):
    """Render an image via the ui.helpers helper (keeps caller signature)."""
    return ui_helpers.render_responsive_image(image, target_display_size, placeholder=placeholder)

# ============================================================================
# Generation Functions
# ============================================================================
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

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
