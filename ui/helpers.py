"""Small UI helper functions: CSS injection and responsive image rendering."""
import base64
import io
import streamlit as st
from PIL import Image
import numpy as np


def inject_custom_css(dark_mode: bool):
    """Inject a small set of CSS rules used by the app.

    The function accepts the current theme (dark_mode) so the
    CSS can be rendered deterministically for testing.
    """
    theme = "dark" if dark_mode else "light"
    css = f"""
    <style>
    :root {{
        --ld-bg-primary: {'#0e1117' if theme == 'dark' else '#ffffff'};
        --ld-bg-secondary: {'#262730' if theme == 'dark' else '#f0f2f6'};
        --ld-text-primary: {'#fafafa' if theme == 'dark' else '#262730'};
        --ld-text-secondary: {'#a3a8b4' if theme == 'dark' else '#6c757d'};
        --ld-accent: {'#ff4b4b' if theme == 'dark' else '#ff4b4b'};
        --ld-status-bg: {'rgba(0,0,0,0.35)' if theme == 'dark' else 'rgba(0,0,0,0.28)'};
        --ld-status-text: {'#ffffff' if theme == 'dark' else '#ffffff'};
    }}

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
    .ld-status-bar.auto-hide {{
        animation: ld-fadeout 0.9s ease-in-out forwards;
        animation-delay: 8s;
    }}
    @keyframes ld-fadeout {{
        0% {{ opacity: 1; transform: translateY(0px); visibility: visible; }}
        100% {{ opacity: 0; transform: translateY(8px); visibility: hidden; }}
    }}
    </style>
    """
    try:
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        # In contexts where Streamlit markup isn't available the caller
        # will simply not see the styling. Don't raise.
        pass


def compute_display_size(image_width, image_height, max_width=800, max_height=600):
    """Compute a display size that fits into a given viewport while maintaining ratio."""
    aspect_ratio = image_width / (image_height or 1)
    if aspect_ratio > (max_width / max_height):
        display_w = max_width
        display_h = int(max_width / aspect_ratio)
    else:
        display_h = max_height
        display_w = int(max_height * aspect_ratio)
    return (display_w, display_h)


@st.cache_data
def image_to_base64(image: Image.Image, format="PNG") -> str:
    """Convert a PIL image to a base64 data URL (cached)."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def render_responsive_image(image, target_display_size, placeholder=None):
    """Render a PIL image (or numpy array) into a streamlit placeholder using CSS variables."""
    if isinstance(image, (list, tuple)) or hasattr(image, 'shape') and not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(np.array(image))
        except Exception:
            # If conversion fails, try to let PIL handle it
            image = Image.fromarray(image)

    display_w, display_h = target_display_size
    # Choose resampling constant depending on Pillow version
    resample = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    # Resize the image to fit inside the target display box while preserving
    # the original aspect ratio. This avoids stretching images when the
    # UI-specified width/height don't match the actual image's aspect.
    try:
        orig_w, orig_h = image.size
    except Exception:
        orig_w, orig_h = display_w, display_h

    if orig_w <= 0 or orig_h <= 0:
        # Fallback: if image reports invalid size use the target box.
        resized_w, resized_h = display_w, display_h
    else:
        # Compute the largest size that fits within the target box
        scale = min(float(display_w) / orig_w, float(display_h) / orig_h)
        # Allow scaling up or down but preserve ratio. If you prefer to avoid
        # upscaling, clamp scale = min(scale, 1.0).
        resized_w = max(1, int(round(orig_w * scale)))
        resized_h = max(1, int(round(orig_h * scale)))

    if (resized_w, resized_h) != image.size:
        try:
            display_image = image.resize((resized_w, resized_h), resample)
        except Exception:
            display_image = image.copy()
    else:
        display_image = image

    img_b64 = image_to_base64(display_image)
    html = f"""
    <div class="ld-responsive-image" style="--ld-display-width: {display_w}px; --ld-display-height: {display_h}px;">
        <img src="data:image/png;base64,{img_b64}" alt="Generated Image">
    </div>
    """
    if placeholder is not None:
        try:
            placeholder.markdown(html, unsafe_allow_html=True)
        except Exception:
            # If placeholder can't render, swallow the error and let the caller
            # attempt a fallback rendering.
            pass
    else:
        try:
            st.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass
