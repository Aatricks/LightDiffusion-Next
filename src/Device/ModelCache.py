"""
Model Persistence Manager for LightDiffusion
Keeps models loaded in VRAM for instant reuse between generations
"""

from typing import Dict, Optional, Any, Tuple, List
import logging
from src.Device import Device


class ModelCache:
    """Global model cache to keep models loaded in VRAM"""

    def __init__(self):
        self._cached_checkpoints: Dict[str, Tuple[Any, Any, Any]] = {}
        self._cached_taesd: Dict[Tuple[int, bool], Any] = {}
        self._cached_conditions: Dict[str, Any] = {}
        self._last_checkpoint_path: Optional[str] = None
        self._keep_models_loaded: bool = True
        self._loaded_models_list: List[Any] = []
        self._max_cached_checkpoints: int = 3
        
        # Prefetching support
        self._prefetched_state_dict: Optional[dict] = None
        self._prefetched_path: Optional[str] = None

    def cache_taesd(self, channels: int, flux: bool, model: Any) -> None:
        """Cache a TAESD model instance"""
        self._cached_taesd[(channels, flux)] = model

    def get_taesd(self, channels: int, flux: bool) -> Optional[Any]:
        """Get a cached TAESD model instance"""
        return self._cached_taesd.get((channels, flux))

    def set_prefetched_model(self, path: str, state_dict: dict) -> None:
        """Store a prefetched state dict in CPU RAM"""
        self._prefetched_path = path
        self._prefetched_state_dict = state_dict
        logging.info(f"ModelCache: Stored prefetched model: {path}")

    def get_prefetched_model(self, path: str) -> Optional[dict]:
        """Get prefetched state dict if path matches"""
        if self._prefetched_path == path:
            logging.info(f"ModelCache: Using prefetched state dict for {path}")
            return self._prefetched_state_dict
        return None

    def clear_prefetch(self) -> None:
        """Clear prefetched data from RAM"""
        self._prefetched_state_dict = None
        self._prefetched_path = None

    def set_keep_models_loaded(self, keep_loaded: bool) -> None:
        """Enable or disable keeping models loaded in VRAM"""
        self._keep_models_loaded = keep_loaded
        if not keep_loaded:
            self.clear_cache()

    def get_keep_models_loaded(self) -> bool:
        """Check if models should be kept loaded"""
        return self._keep_models_loaded

    def cache_checkpoint(
        self, checkpoint_path: str, model_patcher: Any, clip: Any, vae: Any
    ) -> None:
        """Cache a loaded checkpoint"""
        if not self._keep_models_loaded:
            return

        # Limit cache size
        if len(self._cached_checkpoints) >= self._max_cached_checkpoints and checkpoint_path not in self._cached_checkpoints:
            # Remove oldest (first) entry
            oldest_path = next(iter(self._cached_checkpoints))
            old_patcher, _, _ = self._cached_checkpoints.pop(oldest_path)
            try:
                if oldest_path != checkpoint_path:
                    logging.info(f"ModelCache: Evicting {oldest_path} to make room")
                    if hasattr(old_patcher, "model_unload"):
                        old_patcher.model_unload()
            except Exception:
                pass

        self._last_checkpoint_path = checkpoint_path
        self._cached_checkpoints[checkpoint_path] = (model_patcher, clip, vae)
        logging.info(f"Cached checkpoint: {checkpoint_path} (Total cached: {len(self._cached_checkpoints)})")

    def get_cached_checkpoint(
        self, checkpoint_path: str
    ) -> Optional[Tuple[Any, Any, Any]]:
        """Get cached checkpoint if available"""
        if not self._keep_models_loaded:
            return None

        if checkpoint_path in self._cached_checkpoints:
            logging.info(f"Using cached checkpoint: {checkpoint_path}")
            self._last_checkpoint_path = checkpoint_path
            return self._cached_checkpoints[checkpoint_path]
        return None

    def cache_sampling_models(self, models: List[Any]) -> None:
        """Cache models used during sampling"""
        if not self._keep_models_loaded:
            return

        self._loaded_models_list = models.copy()

    def get_cached_sampling_models(self) -> List[Any]:
        """Get cached sampling models"""
        if not self._keep_models_loaded:
            return []
        return self._loaded_models_list

    def prevent_model_cleanup(self, conds: Dict[str, Any], models: List[Any]) -> None:
        """Prevent models from being cleaned up if caching is enabled"""
        if not self._keep_models_loaded:
            # Original cleanup behavior
            from src.cond import cond_util

            cond_util.cleanup_additional_models(models)

            control_cleanup = []
            for k in conds:
                control_cleanup += cond_util.get_models_from_cond(conds[k], "control")
            cond_util.cleanup_additional_models(set(control_cleanup))
        else:
            # Keep models loaded - only cleanup control models that aren't main models
            control_cleanup = []
            for k in conds:
                from src.cond import cond_util

                control_cleanup += cond_util.get_models_from_cond(conds[k], "control")

            # Only cleanup control models, not the main models
            from src.cond import cond_util

            cond_util.cleanup_additional_models(set(control_cleanup))
            logging.info("Kept main models loaded in VRAM for reuse")

    def clear_cache(self) -> None:
        """Clear all cached models"""
        for path, (model_patcher, _, _) in self._cached_checkpoints.items():
            try:
                if hasattr(model_patcher, "model_unload"):
                    model_patcher.model_unload()
            except Exception as e:
                logging.warning(f"Error unloading cached model {path}: {e}")

        self._cached_checkpoints.clear()
        self._cached_taesd.clear()
        self._cached_conditions.clear()
        self._last_checkpoint_path = None
        self._loaded_models_list.clear()

        # Force cleanup
        Device.cleanup_models(keep_clone_weights_loaded=False)
        Device.soft_empty_cache(force=True)
        logging.info("Cleared model cache and freed VRAM")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        device = Device.get_torch_device()
        total_mem = Device.get_total_memory(device)
        free_mem = Device.get_free_memory(device)
        used_mem = total_mem - free_mem

        return {
            "total_vram": total_mem / (1024 * 1024 * 1024),  # GB
            "used_vram": used_mem / (1024 * 1024 * 1024),  # GB
            "free_vram": free_mem / (1024 * 1024 * 1024),  # GB
            "cached_models": len(self._cached_checkpoints),
            "keep_loaded": self._keep_models_loaded,
            "has_cached_checkpoint": len(self._cached_checkpoints) > 0,
        }


# Global model cache instance
model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """Get the global model cache instance"""
    return model_cache


def set_keep_models_loaded(keep_loaded: bool) -> None:
    """Global function to enable/disable model persistence"""
    model_cache.set_keep_models_loaded(keep_loaded)


def get_keep_models_loaded() -> bool:
    """Global function to check if models should be kept loaded"""
    return model_cache.get_keep_models_loaded()


def clear_model_cache() -> None:
    """Global function to clear model cache"""
    model_cache.clear_cache()


def get_memory_info() -> Dict[str, Any]:
    """Global function to get memory info"""
    return model_cache.get_memory_info()
