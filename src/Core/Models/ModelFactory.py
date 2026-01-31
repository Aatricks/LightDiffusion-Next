"""Model factory for LightDiffusion-Next.

Provides automatic model type detection and instantiation.
Simplified to a single function with a registry for extensibility.
"""

import logging
import os
from typing import Optional, Type

from src.Core.AbstractModel import AbstractModel

logger = logging.getLogger(__name__)

# Model type registry - maps type names to model classes
_MODEL_REGISTRY: dict[str, Type[AbstractModel]] = {}

# SDXL detection keywords
_SDXL_INDICATORS = frozenset(["sdxl", "refiner", "hassaku", "juggernaut", "xl"])


def _ensure_registry_populated():
    """Lazily populate registry to avoid circular imports."""
    if not _MODEL_REGISTRY:
        from src.Core.Models.SD15Model import SD15Model
        from src.Core.Models.SDXLModel import SDXLModel
        _MODEL_REGISTRY["SD15"] = SD15Model
        _MODEL_REGISTRY["SDXL"] = SDXLModel


def detect_model_type(model_path: Optional[str]) -> str:
    """Detect model type from file path.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        'SD15' or 'SDXL'
        
    Raises:
        ValueError: If GGUF file provided (unsupported)
    """
    if not model_path:
        return "SD15"
    
    lp = model_path.lower()
    
    if lp.endswith(".gguf"):
        raise ValueError(f"GGUF files not supported: {model_path}")
    
    base = os.path.basename(lp)
    if any(ind in base for ind in _SDXL_INDICATORS):
        return "SDXL"
    
    return "SD15"


def create_model(
    model_path: Optional[str] = None,
    model_type: Optional[str] = None,
) -> AbstractModel:
    """Create a model instance with automatic type detection.
    
    Args:
        model_path: Path to checkpoint file
        model_type: Explicit type ('SD15', 'SDXL'), or None to auto-detect
        
    Returns:
        Configured model instance (not yet loaded)
        
    Example:
        model = create_model("./checkpoints/dreamer.safetensors")
        model.load()
        positive, negative = model.encode_prompt("a cat")
        latents = model.generate(ctx, positive, negative)
        image = model.decode(latents["samples"])
    """
    _ensure_registry_populated()
    
    if model_type is None:
        model_type = detect_model_type(model_path)
    
    if model_type not in _MODEL_REGISTRY:
        logger.warning(f"Unknown model type '{model_type}', using SD15")
        model_type = "SD15"
    
    logger.info(f"Creating {model_type} model: {model_path}")
    return _MODEL_REGISTRY[model_type](model_path=model_path)


def register_model_type(type_name: str, model_class: Type[AbstractModel]) -> None:
    """Register a custom model type.
    
    Args:
        type_name: Identifier for the model type
        model_class: Class inheriting from AbstractModel
    """
    _ensure_registry_populated()
    
    if not issubclass(model_class, AbstractModel):
        raise TypeError(f"{model_class} must inherit from AbstractModel")
    
    _MODEL_REGISTRY[type_name] = model_class
    logger.info(f"Registered model type: {type_name}")


def list_model_types() -> list[str]:
    """List registered model types."""
    _ensure_registry_populated()
    return list(_MODEL_REGISTRY.keys())


def list_available_models(
    checkpoint_dir: str = "./include/checkpoints/",
    return_mapping: bool = False,
) -> list:
    """List available model files in the checkpoints directory.
    
    Args:
        checkpoint_dir: Directory to scan for models
        return_mapping: If True, return list of (display_name, full_path) tuples
        
    Returns:
        List of model names, or list of (name, path) tuples if return_mapping=True
    """
    import glob
    
    valid_extensions = (".safetensors", ".pt", ".pth")
    results = []
    
    if not os.path.isdir(checkpoint_dir):
        return results
    
    for ext in valid_extensions:
        pattern = os.path.join(checkpoint_dir, f"*{ext}")
        for filepath in glob.glob(pattern):
            basename = os.path.basename(filepath)
            if return_mapping:
                results.append((basename, filepath))
            else:
                results.append(basename)
    
    # Sort alphabetically
    results.sort(key=lambda x: x[0].lower() if isinstance(x, tuple) else x.lower())
    return results
