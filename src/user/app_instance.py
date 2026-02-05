"""
App instance for managing UI state and real-time previews
"""

import os
import threading
import time
from typing import List, Any
from PIL import Image


class AppInstance:
    """Main application instance for managing UI state and previews"""

    def __init__(self):
        self.previewer_var = PreviewerVar()
        self.preview_dir = os.path.join(".", "output", "preview")
        self.preview_lock = threading.Lock()
        self.preview_files = []
        self.preview_base64 = []  # In-memory base64 strings
        self.last_preview_time = 0
        self.current_step = 0
        self.total_steps = 0
        self.progress = ProgressTracker()
        self._interrupt_event = threading.Event()

        # Create preview directory
        os.makedirs(self.preview_dir, exist_ok=True)

    def update_image(self, images: List[Any], step: int = 0, total_steps: int = 0):
        """Update the gallery with preview images in real-time.
        
        Args:
            images: List of PIL.Image or base64 strings
            step: Current step
            total_steps: Total steps
        """
        with self.preview_lock:
            # Update metadata
            self.current_step = step
            self.total_steps = total_steps
            timestamp = int(time.time() * 1000)
            self.last_preview_time = timestamp
            
            # Store in-memory
            new_previews = []
            for img in images:
                if isinstance(img, str) and img.startswith("data:image"):
                    new_previews.append(img)
                elif hasattr(img, "save"): # PIL Image
                    try:
                        import io
                        import base64
                        buffered = io.BytesIO()
                        # Use WEBP for smaller memory footprint if possible, fallback to PNG
                        img.save(buffered, format="WEBP", quality=80)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        new_previews.append(f"data:image/webp;base64,{img_str}")
                    except Exception:
                        try:
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            new_previews.append(f"data:image/png;base64,{img_str}")
                        except Exception:
                            pass
            
            self.preview_base64 = new_previews

    def get_latest_previews(self):
        """Get the latest preview images and metadata"""
        with self.preview_lock:
            try:
                return {
                    "paths": [], # Deprecated path-based previews
                    "base64": self.preview_base64,
                    "step": self.current_step,
                    "total_steps": self.total_steps,
                    "timestamp": self.last_preview_time
                }
            except Exception as e:
                print(f"Error loading preview images: {e}")
                return {"paths": [], "base64": [], "step": 0, "total_steps": 0, "timestamp": 0}

    def clear_preview_files(self):
        """Clear temporary preview data"""
        with self.preview_lock:
            self.preview_base64 = []
            self.preview_files = []

    def cleanup_all_previews(self):
        """Cleanup all preview files in the directory and clear memory"""
        self.clear_preview_files()
        try:
            if os.path.exists(self.preview_dir):
                for filename in os.listdir(self.preview_dir):
                    if filename.startswith("preview_") and filename.endswith((".png", ".webp")):
                        file_path = os.path.join(self.preview_dir, filename)
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error cleaning up preview directory: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.clear_preview_files()
        self.clear_interrupt()

    @property
    def interrupt_flag(self) -> bool:
        """Return True when an interrupt has been requested"""
        return self._interrupt_event.is_set()

    def request_interrupt(self):
        """Signal sampling loops to stop"""
        self._interrupt_event.set()

    def clear_interrupt(self):
        """Reset interrupt state after a run"""
        self._interrupt_event.clear()


class PreviewerVar:
    """Variable to control preview functionality"""

    def __init__(self):
        self._enabled = True

    def get(self) -> bool:
        """Get preview enabled state"""
        return self._enabled

    def set(self, value: bool):
        """Set preview enabled state"""
        self._enabled = value


class ProgressTracker:
    """Simple progress tracker for sampling"""

    def __init__(self):
        self._progress = 0.0

    def set(self, value: float):
        """Set progress value (0.0 to 1.0)"""
        self._progress = max(0.0, min(1.0, value))

    def get(self) -> float:
        """Get current progress value"""
        return self._progress


# Global app instance
app = AppInstance()
