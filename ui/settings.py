import os
import json
import streamlit as st

SETTINGS_FILE = "./webui_settings.json"


def get_default_settings():
    """Return the default UI settings dictionary.

    Keeping the defaults here makes it easy to unit test and to keep
    the main app file small and focused on layout.
    """
    return {
        # Prompt & Text
        "prompt": "",
        "negative_prompt": "",

        # Dimensions & Batch
        "width": 512,
        "height": 512,
        "num_images": 1,
        "batch_size": 1,

        # Generation Modes
        "flux_mode": False,
        "realistic_mode": False,
        "img2img_mode": False,

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

        # DeepCache
        "deepcache_enabled": False,
        "deepcache_interval": 3,
        "deepcache_depth": 2,
        "deepcache_start_step": 0,
        "deepcache_end_step": 1000,
        
        # Scheduler & Sampling
        "scheduler": "ays",  # Options: normal, karras, simple, beta, ays
        "sampler": "dpmpp_sde_cfgpp",  # Options: euler, euler_ancestral, dpmpp_2m, etc.
        "steps": 20,
        
        # Optimizations
        "prompt_cache_enabled": True,  # Cache CLIP embeddings (5-15% speedup)

        # UI Settings
        "dark_mode": True,
        "verbose_mode": False,
        "ui_scale": 1.0,
        "sidebar_collapsed_by_default": False,
    }


def load_settings():
    """Load settings from disk and merge them with defaults."""
    defaults = get_default_settings()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    defaults.update(saved)
        except Exception as e:
            try:
                st.warning(f"Could not load settings: {e}")
            except Exception:
                pass
    return defaults


def save_settings(settings: dict):
    """Persist settings to disk.

    Errors are surfaced to the user via Streamlit when possible.
    """
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        try:
            st.error(f"Could not save settings: {e}")
        except Exception:
            pass
