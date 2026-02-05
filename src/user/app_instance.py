"""
App instance for managing UI state and real-time previews
"""

import os
import threading
import time
from typing import List
from PIL import Image


class AppInstance:
    """Main application instance for managing UI state and previews"""

    def __init__(self):
        self.previewer_var = PreviewerVar()
        self.preview_dir = os.path.join(".", "output", "preview")
        self.preview_lock = threading.Lock()
        self.preview_files = []
        self.last_preview_time = 0
        self.current_step = 0
        self.total_steps = 0
        self.progress = ProgressTracker()
        self._interrupt_event = threading.Event()

        # Create preview directory
        os.makedirs(self.preview_dir, exist_ok=True)

    def update_image(self, images: List[Image.Image], step: int = 0, total_steps: int = 0):
        """Update the gallery with preview images in real-time"""
        with self.preview_lock:
            # Clear old preview files
            self.clear_preview_files()

            # Save new preview images with timestamp
            self.preview_files = []
            self.current_step = step
            self.total_steps = total_steps
            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness

            for i, img in enumerate(images):
                preview_path = os.path.join(
                    self.preview_dir, f"preview_{timestamp}_{i}.png"
                )
                # Resize to a reasonable preview size for performance
                preview_img = img.copy()
                preview_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                preview_img.save(preview_path)
                self.preview_files.append(preview_path)

            self.last_preview_time = timestamp

    def get_latest_previews(self):
        """Get the latest preview images and metadata"""
        with self.preview_lock:
            try:
                paths = []
                if self.preview_files:
                    # Return existing file paths if they exist
                    paths = [path for path in self.preview_files if os.path.exists(path)]
                
                return {
                    "paths": paths,
                    "step": self.current_step,
                    "total_steps": self.total_steps,
                    "timestamp": self.last_preview_time
                }
            except Exception as e:
                print(f"Error loading preview images: {e}")
                return {"paths": [], "step": 0, "total_steps": 0, "timestamp": 0}

    def clear_preview_files(self):
        """Clear temporary preview files"""
        for file_path in list(self.preview_files):
            if not os.path.exists(file_path):
                try:
                    self.preview_files.remove(file_path)
                except Exception:
                    pass
                continue
            # Try a few times to remove the file in case an Image handle
            # is still being closed by another thread/process (common on
            # Windows). If we cannot remove after retries, skip it.
            removed = False
            for attempt in range(3):
                try:
                    os.remove(file_path)
                    removed = True
                    break
                except PermissionError:
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Error removing preview file {file_path}: {e}")
                    break
            try:
                if removed:
                    self.preview_files.remove(file_path)
            except Exception:
                pass

    def cleanup_all_previews(self):
        """Cleanup all preview files in the directory"""
        # Remove all preview files with a short retry on PermissionError to
        # reduce noisy 'file is used by another process' errors on Windows.
        try:
            for filename in os.listdir(self.preview_dir):
                if filename.startswith("preview_") and filename.endswith(".png"):
                    file_path = os.path.join(self.preview_dir, filename)
                    for attempt in range(3):
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            break
                        except PermissionError:
                            time.sleep(0.05)
                        except Exception as e:
                            print(f"Error cleaning up preview file {file_path}: {e}")
                            break
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
