"""
History module - thin wrapper delegating to HistoryManager.

Maintains backward compatibility with existing UI code while
using the centralized HistoryManager for all operations.
"""

import os
import streamlit as st
from typing import Optional, List, Dict, Any

from src.FileManaging.HistoryManager import (
    HistoryManager,
    HistoryEntry,
    sanitize_seed_for_display,
    get_history_manager,
)

# Legacy file path - keep for reference
HISTORY_FILE = "./webui_history.json"

# Singleton manager instance
_manager: Optional[HistoryManager] = None


def _get_manager() -> HistoryManager:
    """Get or create the HistoryManager singleton."""
    global _manager
    if _manager is None:
        _manager = get_history_manager()
    return _manager


def load_history() -> List[Dict[str, Any]]:
    """
    Load generation history from disk.
    
    Returns:
        List of history entry dictionaries (for backward compatibility).
    """
    manager = _get_manager()
    try:
        entries = manager.load(use_cache=True)
        return [e.to_dict() for e in entries]
    except Exception as e:
        try:
            st.warning(f"Could not load history: {e}")
        except Exception:
            pass
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    """
    Save generation history to disk.
    
    Args:
        history: List of history entry dictionaries.
    """
    manager = _get_manager()
    try:
        entries = [manager._normalize_entry(item) for item in history]
        manager.save(entries)
    except Exception as e:
        try:
            st.error(f"Could not save history: {e}")
        except Exception:
            pass


def add_to_history(image_paths: List[str], settings: Dict[str, Any]) -> None:
    """
    Add generated image file paths to the persistent history store.
    
    Args:
        image_paths: List of paths to generated images.
        settings: Generation settings dictionary.
    """
    manager = _get_manager()
    try:
        manager.add_from_image_paths(image_paths, settings)
    except Exception as e:
        try:
            st.warning(f"Could not add to history: {e}")
        except Exception:
            pass


def clear_history() -> None:
    """Remove all history and delete associated image files."""
    manager = _get_manager()
    try:
        manager.clear(delete_files=True)
    except Exception as e:
        try:
            st.error(f"Could not clear history: {e}")
        except Exception:
            pass


def scan_output_folders() -> List[Dict[str, Any]]:
    """
    Scan common output folders and build a history list of found PNGs.
    
    Returns:
        List of history entry dictionaries.
    """
    manager = _get_manager()
    try:
        entries = manager.scan_output_folders()
        return [e.to_dict() for e in entries]
    except Exception as e:
        try:
            st.warning(f"Could not scan output folders: {e}")
        except Exception:
            pass
        return []


def delete_history_entry(entry_index: int) -> bool:
    """
    Delete a history entry by index.
    
    Args:
        entry_index: Index of the entry to delete.
        
    Returns:
        True if deletion was successful.
    """
    manager = _get_manager()
    try:
        return manager.delete_entry(entry_index)
    except Exception as e:
        try:
            st.error(f"Could not delete history entry: {e}")
        except Exception:
            pass
        return False


# =========================================================================
# New Search and Filter API
# =========================================================================

def search_history(
    keyword: Optional[str] = None,
    model_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search and filter history entries.
    
    Args:
        keyword: Search in prompt/negative_prompt (case-insensitive).
        model_type: Filter by model type (SD15, SDXL, Flux, etc.).
        date_from: Filter entries from this date (YYYY-MM-DD).
        date_to: Filter entries until this date (YYYY-MM-DD).
        
    Returns:
        Filtered list of history entry dictionaries.
    """
    manager = _get_manager()
    try:
        entries = manager.search(
            keyword=keyword,
            model_type=model_type,
            date_from=date_from,
            date_to=date_to,
        )
        return [e.to_dict() for e in entries]
    except Exception:
        return []


def get_available_model_types() -> List[str]:
    """
    Get list of model types present in history.
    
    Returns:
        Sorted list of unique model type strings.
    """
    manager = _get_manager()
    try:
        return manager.get_model_types()
    except Exception:
        return []


def get_history_date_range() -> tuple:
    """
    Get the date range of entries in history.
    
    Returns:
        Tuple of (min_date, max_date) as YYYY-MM-DD strings, or (None, None).
    """
    manager = _get_manager()
    try:
        return manager.get_date_range()
    except Exception:
        return (None, None)


# Re-export for backward compatibility
__all__ = [
    "sanitize_seed_for_display",
    "load_history",
    "save_history",
    "add_to_history",
    "clear_history",
    "scan_output_folders",
    "delete_history_entry",
    "search_history",
    "get_available_model_types",
    "get_history_date_range",
    "HISTORY_FILE",
]
