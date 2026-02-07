"""
HistoryManager - Centralized image history and metadata management.

This module provides a unified API for managing generation history across
both Streamlit and Gradio UIs with features including:
- Type-safe HistoryEntry dataclass
- In-memory caching with invalidation
- Deduplication by image path
- Backup rotation for corrupt JSON recovery
- Search and filtering capabilities
"""

import os
import re
import json
import glob
import time
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from PIL import Image


# Configuration
HISTORY_FILE = "./webui_history.json"
BACKUP_DIR = "./.history_backups"
MAX_HISTORY_ENTRIES = 100
MAX_BACKUPS = 3


@dataclass
class HistoryEntry:
    """Type-safe representation of a history entry."""
    timestamp: str
    image_path: str
    prompt: str = ""
    negative_prompt: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    batch_size: Optional[int] = None
    model_type: Optional[str] = None
    model_path: Optional[str] = None
    seed: Optional[str] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    generation_duration: Optional[float] = None
    avg_iters_per_s: Optional[float] = None
    cfg: Optional[float] = None
    scheduler: Optional[str] = None
    denoise: Optional[float] = None
    png_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        """Create entry from dictionary, handling missing fields gracefully."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def sanitize_seed_for_display(seed_value: Any) -> Optional[str]:
    """
    Return a safe seed string or None if the value looks like a tensor/image dump.
    
    Handles various edge cases:
    - Numeric values (int/float)
    - String representations with tensor dumps
    - Very long strings that indicate binary/array data
    - Extracts numeric tokens from mixed content
    """
    if seed_value is None:
        return None
    
    if isinstance(seed_value, (int, float)):
        return str(int(seed_value))
    
    if isinstance(seed_value, str):
        s = seed_value.strip()
        
        # Detect and reject tensor/array dumps
        if any(pattern in s.lower() for pattern in ["tensor(", "array(", "[[", "]]"]):
            # Try to extract numeric token
            m = re.search(r"(\d{4,})", s)
            return m.group(0) if m else None
        
        # Reject multiline or excessively long strings
        if "\n" in s or len(s) > 240:
            m = re.search(r"(\d{4,})", s)
            return m.group(0) if m else None
        
        # Reject bracket-heavy content (likely JSON/list dumps)
        if s.count("[") > 2 or s.count("{") > 2:
            m = re.search(r"(\d{4,})", s)
            return m.group(0) if m else None
        
        # Reject array-like content (starts with [ and contains commas)
        if s.startswith("[") and "," in s:
            m = re.search(r"(\d{4,})", s)
            return m.group(0) if m else None
        
        return s if s else None
    
    return None



def _parse_float_safe(value: Any) -> Optional[float]:
    """Safely parse a float value, handling string suffixes like 's'."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return float(str(value).rstrip('s'))
        except (ValueError, TypeError):
            return None


def _parse_int_safe(value: Any) -> Optional[int]:
    """Safely parse an integer value."""
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return int(value)
    except (ValueError, TypeError):
        return None


class HistoryManager:
    """
    Centralized manager for image generation history.
    
    Features:
    - In-memory caching for fast access
    - Automatic deduplication by image path
    - Backup rotation for data safety
    - Search and filter capabilities
    """
    
    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self._cache: Optional[List[HistoryEntry]] = None
        self._cache_mtime: float = 0
    
    def _create_backup(self) -> None:
        """Create a backup of the current history file."""
        if not os.path.exists(self.history_file):
            return
        
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        # Rotate existing backups
        for i in range(MAX_BACKUPS - 1, 0, -1):
            old_backup = os.path.join(BACKUP_DIR, f"history_backup_{i}.json")
            new_backup = os.path.join(BACKUP_DIR, f"history_backup_{i + 1}.json")
            if os.path.exists(old_backup):
                if i + 1 > MAX_BACKUPS:
                    os.remove(old_backup)
                else:
                    shutil.move(old_backup, new_backup)
        
        # Create new backup
        backup_path = os.path.join(BACKUP_DIR, "history_backup_1.json")
        try:
            shutil.copy2(self.history_file, backup_path)
        except Exception:
            pass  # Best effort backup
    
    def _restore_from_backup(self) -> List[Dict[str, Any]]:
        """Attempt to restore history from the most recent valid backup."""
        if not os.path.exists(BACKUP_DIR):
            return []
        
        for i in range(1, MAX_BACKUPS + 1):
            backup_path = os.path.join(BACKUP_DIR, f"history_backup_{i}.json")
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Restore successful - copy backup to main file
                            shutil.copy2(backup_path, self.history_file)
                            return data
                except Exception:
                    continue
        
        return []
    
    def _invalidate_cache(self) -> None:
        """Invalidate the in-memory cache."""
        self._cache = None
        self._cache_mtime = 0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid based on file modification time."""
        if self._cache is None:
            return False
        
        try:
            current_mtime = os.path.getmtime(self.history_file)
            return current_mtime == self._cache_mtime
        except OSError:
            return False
    
    def load(self, use_cache: bool = True) -> List[HistoryEntry]:
        """
        Load history entries from disk with caching.
        
        Args:
            use_cache: If True, return cached data if available and valid.
            
        Returns:
            List of HistoryEntry objects, deduplicated by image_path.
        """
        if use_cache and self._is_cache_valid():
            return self._cache
        
        raw_data = []
        
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except json.JSONDecodeError:
                # Attempt restore from backup
                raw_data = self._restore_from_backup()
            except Exception:
                raw_data = []
        
        # Convert to HistoryEntry and deduplicate
        seen_paths = set()
        entries = []
        
        for item in raw_data:
            if not isinstance(item, dict):
                continue
            
            path = item.get("image_path")
            if path and path in seen_paths:
                continue  # Skip duplicate
            
            if path:
                seen_paths.add(path)
            
            # Normalize fields
            entry = self._normalize_entry(item)
            entries.append(entry)
        
        # Enforce max limit
        entries = entries[:MAX_HISTORY_ENTRIES]
        
        # Update cache
        self._cache = entries
        try:
            self._cache_mtime = os.path.getmtime(self.history_file)
        except OSError:
            self._cache_mtime = 0
        
        return entries
    
    def _normalize_entry(self, data: Dict[str, Any]) -> HistoryEntry:
        """Normalize a raw dictionary into a HistoryEntry with sanitized fields."""
        png_meta = data.get("png_metadata") or {}
        
        # Normalize seed
        seed = sanitize_seed_for_display(data.get("seed"))
        if not seed and isinstance(png_meta, dict):
            seed = sanitize_seed_for_display(png_meta.get("seed"))
        
        # Normalize dimensions
        width = _parse_int_safe(data.get("width"))
        height = _parse_int_safe(data.get("height"))
        
        # Try to get dimensions from image if missing
        if (width is None or height is None) and data.get("image_path"):
            img_path = data.get("image_path")
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception:
                    pass
        
        # Normalize numeric fields
        steps = _parse_int_safe(data.get("steps") or png_meta.get("steps"))
        cfg = _parse_float_safe(data.get("cfg") or png_meta.get("cfg"))
        generation_duration = _parse_float_safe(
            data.get("generation_duration") or png_meta.get("generation_duration")
        )
        avg_iters_per_s = _parse_float_safe(
            data.get("avg_iters_per_s") or png_meta.get("avg_iters_per_s")
        )
        denoise = _parse_float_safe(data.get("denoise") or png_meta.get("denoise"))
        
        return HistoryEntry(
            timestamp=data.get("timestamp", ""),
            image_path=data.get("image_path", ""),
            prompt=data.get("prompt") or png_meta.get("prompt", ""),
            negative_prompt=data.get("negative_prompt") or png_meta.get("negative_prompt", ""),
            width=width,
            height=height,
            batch_size=_parse_int_safe(data.get("batch_size")),
            model_type=data.get("model_type") or png_meta.get("model_type"),
            model_path=data.get("model_path") or png_meta.get("model_path"),
            seed=seed,
            sampler=data.get("sampler") or png_meta.get("sampler"),
            steps=steps,
            generation_duration=generation_duration,
            avg_iters_per_s=avg_iters_per_s,
            cfg=cfg,
            scheduler=data.get("scheduler") or png_meta.get("scheduler"),
            denoise=denoise,
            png_metadata=png_meta if isinstance(png_meta, dict) else {},
        )
    
    def save(self, entries: List[HistoryEntry]) -> bool:
        """
        Save history entries to disk with backup rotation.
        
        Args:
            entries: List of HistoryEntry objects to save.
            
        Returns:
            True if save was successful, False otherwise.
        """
        # Create backup before overwriting
        self._create_backup()
        
        # Enforce limit
        entries = entries[:MAX_HISTORY_ENTRIES]
        
        try:
            data = [e.to_dict() for e in entries]
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Invalidate cache to force reload
            self._invalidate_cache()
            return True
        except Exception:
            return False
    
    def add_entry(self, entry: HistoryEntry) -> bool:
        """
        Add a new entry to the history (at the beginning).
        
        Args:
            entry: The HistoryEntry to add.
            
        Returns:
            True if successful, False otherwise.
        """
        entries = self.load(use_cache=False)
        
        # Remove any existing entry with the same path
        entries = [e for e in entries if e.image_path != entry.image_path]
        
        # Insert at beginning
        entries.insert(0, entry)
        
        return self.save(entries)
    
    def add_from_image_paths(
        self,
        image_paths: List[str],
        settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add entries from a list of image paths, extracting PNG metadata.
        
        Args:
            image_paths: List of paths to PNG images.
            settings: Optional settings dict to supplement PNG metadata.
            
        Returns:
            True if all entries were added successfully.
        """
        settings = settings or {}
        entries = self.load(use_cache=False)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            
            # Read PNG metadata and dimensions
            png_meta = {}
            width, height = None, None
            try:
                with Image.open(img_path) as img:
                    png_meta = getattr(img, "info", {}) or {}
                    width, height = img.size
            except Exception:
                continue
            
            # Create normalized entry
            entry_data = {
                "timestamp": timestamp,
                "image_path": img_path,
                "prompt": png_meta.get("prompt") or settings.get("prompt", ""),
                "negative_prompt": png_meta.get("negative_prompt") or settings.get("negative_prompt", ""),
                "width": width,
                "height": height,
                "batch_size": settings.get("batch_size"),
                "model_type": png_meta.get("model_type"),
                "model_path": png_meta.get("model_path"),
                "seed": png_meta.get("seed"),
                "sampler": png_meta.get("sampler"),
                "steps": png_meta.get("steps"),
                "generation_duration": png_meta.get("generation_duration"),
                "avg_iters_per_s": png_meta.get("avg_iters_per_s"),
                "cfg": png_meta.get("cfg"),
                "scheduler": png_meta.get("scheduler"),
                "denoise": png_meta.get("denoise"),
                "png_metadata": png_meta,
            }
            
            entry = self._normalize_entry(entry_data)
            
            # Remove existing entry with same path
            entries = [e for e in entries if e.image_path != entry.image_path]
            entries.insert(0, entry)
        
        return self.save(entries)
    
    def delete_entry(self, index: int) -> bool:
        """
        Delete an entry by index and remove the associated image file.
        
        Args:
            index: The index of the entry to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        entries = self.load(use_cache=False)
        
        if not (0 <= index < len(entries)):
            return False
        
        entry = entries[index]
        
        # Delete image file
        if entry.image_path and os.path.exists(entry.image_path):
            try:
                os.remove(entry.image_path)
            except Exception:
                pass  # Continue even if file deletion fails
        
        # Remove from list
        entries.pop(index)
        return self.save(entries)
    
    def clear(self, delete_files: bool = True) -> bool:
        """
        Clear all history entries.
        
        Args:
            delete_files: If True, also delete the associated image files.
            
        Returns:
            True if successful, False otherwise.
        """
        if delete_files:
            entries = self.load(use_cache=False)
            for entry in entries:
                if entry.image_path and os.path.exists(entry.image_path):
                    try:
                        os.remove(entry.image_path)
                    except Exception:
                        pass
        
        return self.save([])
    
    def scan_output_folders(
        self,
        output_dirs: Optional[List[str]] = None
    ) -> List[HistoryEntry]:
        """
        Scan output folders for PNG images and build/update history.
        
        Args:
            output_dirs: List of directories to scan. Defaults to standard output dirs.
            
        Returns:
            Updated list of history entries.
        """
        if output_dirs is None:
            output_dirs = [
                "./output/Classic",
                "./output/HiresFix",
                "./output/Img2Img",
                "./output/Adetailer",
                "./output/ControlNet",
                "./output/Flux",
            ]
        
        # Collect all PNG files
        all_images = []
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                images = glob.glob(f"{output_dir}/*.png")
                all_images.extend(images)
        
        # Sort by modification time (newest first)
        all_images = sorted(all_images, key=os.path.getmtime, reverse=True)
        
        # Get existing entries as a lookup
        existing = self.load(use_cache=False)
        existing_map = {e.image_path: e for e in existing}
        
        # Build new history preserving existing metadata
        new_entries = []
        seen_paths = set()
        
        for img_path in all_images[:MAX_HISTORY_ENTRIES]:
            if img_path in seen_paths:
                continue
            seen_paths.add(img_path)
            
            if img_path in existing_map:
                new_entries.append(existing_map[img_path])
            else:
                # Create new entry from image
                try:
                    mtime = os.path.getmtime(img_path)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                    
                    with Image.open(img_path) as img:
                        width, height = img.size
                        png_meta = getattr(img, "info", {}) or {}
                    
                    entry_data = {
                        "timestamp": timestamp,
                        "image_path": img_path,
                        "prompt": png_meta.get("prompt", "(prompt not available)"),
                        "negative_prompt": png_meta.get("negative_prompt", ""),
                        "width": width,
                        "height": height,
                        "png_metadata": png_meta,
                    }
                    
                    new_entries.append(self._normalize_entry(entry_data))
                except Exception:
                    continue
        
        self.save(new_entries)
        return new_entries
    
    # =========================================================================
    # Search and Filter Methods
    # =========================================================================
    
    def search(
        self,
        keyword: Optional[str] = None,
        model_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ) -> List[HistoryEntry]:
        """
        Search and filter history entries.
        
        Args:
            keyword: Search in prompt and negative_prompt (case-insensitive).
            model_type: Filter by model type (SD15, SDXL, Flux, etc.).
            date_from: Filter entries from this date (YYYY-MM-DD format).
            date_to: Filter entries until this date (YYYY-MM-DD format).
            min_width: Minimum image width.
            min_height: Minimum image height.
            
        Returns:
            Filtered list of HistoryEntry objects.
        """
        entries = self.load()
        results = []
        
        keyword_lower = keyword.lower() if keyword else None
        
        for entry in entries:
            # Keyword search
            if keyword_lower:
                prompt_match = keyword_lower in (entry.prompt or "").lower()
                neg_match = keyword_lower in (entry.negative_prompt or "").lower()
                if not (prompt_match or neg_match):
                    continue
            
            # Model type filter
            if model_type and entry.model_type:
                if model_type.lower() not in entry.model_type.lower():
                    continue
            elif model_type and not entry.model_type:
                continue
            
            # Date range filter
            if date_from and entry.timestamp < date_from:
                continue
            if date_to and entry.timestamp > date_to + " 23:59:59":
                continue
            
            # Dimension filters
            if min_width and (entry.width is None or entry.width < min_width):
                continue
            if min_height and (entry.height is None or entry.height < min_height):
                continue
            
            results.append(entry)
        
        return results
    
    def get_model_types(self) -> List[str]:
        """Get a list of unique model types in the history."""
        entries = self.load()
        types = {e.model_type for e in entries if e.model_type}
        return sorted(types)
    
    def get_date_range(self) -> tuple:
        """Get the date range of entries in history."""
        entries = self.load()
        if not entries:
            return None, None
        
        dates = [e.timestamp[:10] for e in entries if e.timestamp]
        if not dates:
            return None, None
        
        return min(dates), max(dates)


# Global singleton instance for convenience
_default_manager: Optional[HistoryManager] = None


def get_history_manager() -> HistoryManager:
    """Get the default HistoryManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = HistoryManager()
    return _default_manager
