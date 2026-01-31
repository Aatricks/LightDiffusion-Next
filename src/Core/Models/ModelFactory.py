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

# Flux2 Klein detection keywords
_FLUX2_KLEIN_INDICATORS = frozenset(["flux2", "klein", "flux_klein", "flux-klein", "flux2_klein", "flux-2"])

# Default paths for Flux2 components
FLUX2_DIFFUSION_MODEL_DIR = "./include/diffusion_model"
FLUX2_TEXT_ENCODER_DIR = "./include/text_encoder"
FLUX2_VAE_DIR = "./include/vae"


def _ensure_registry_populated():
    """Lazily populate registry to avoid circular imports."""
    if not _MODEL_REGISTRY:
        from src.Core.Models.SD15Model import SD15Model
        from src.Core.Models.SDXLModel import SDXLModel
        from src.Core.Models.Flux2KleinModel import Flux2KleinModel
        _MODEL_REGISTRY["SD15"] = SD15Model
        _MODEL_REGISTRY["SDXL"] = SDXLModel
        _MODEL_REGISTRY["Flux2Klein"] = Flux2KleinModel


def _find_flux2_components() -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Auto-detect Flux2 components in default directories.
    
    Returns:
        Tuple of (diffusion_model_path, text_encoder_path, vae_path)
    """
    diffusion_path = None
    text_encoder_path = None
    vae_path = None
    
    # Find diffusion model
    if os.path.exists(FLUX2_DIFFUSION_MODEL_DIR):
        for f in os.listdir(FLUX2_DIFFUSION_MODEL_DIR):
            f_lower = f.lower()
            if ("flux" in f_lower or "klein" in f_lower) and f.endswith((".safetensors", ".pt", ".pth")):
                diffusion_path = os.path.join(FLUX2_DIFFUSION_MODEL_DIR, f)
                break
    
    # Find text encoder
    if os.path.exists(FLUX2_TEXT_ENCODER_DIR):
        for f in os.listdir(FLUX2_TEXT_ENCODER_DIR):
            f_lower = f.lower()
            if ("qwen" in f_lower or "klein" in f_lower) and f.endswith((".safetensors", ".pt", ".pth")):
                text_encoder_path = os.path.join(FLUX2_TEXT_ENCODER_DIR, f)
                break
    
    # Find VAE
    if os.path.exists(FLUX2_VAE_DIR):
        for f in os.listdir(FLUX2_VAE_DIR):
            if f.endswith((".safetensors", ".pt", ".pth")):
                vae_path = os.path.join(FLUX2_VAE_DIR, f)
                break
    
    return diffusion_path, text_encoder_path, vae_path


def detect_model_type(model_path: Optional[str]) -> str:
    """Detect model type from file path.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        'SD15', 'SDXL', or 'Flux2Klein'
        
    Raises:
        ValueError: If GGUF file provided (unsupported)
    """
    if not model_path:
        # Check if Flux2 components exist in default dirs
        diffusion_path, _, _ = _find_flux2_components()
        if diffusion_path:
            return "Flux2Klein"
        return "SD15"
    
    lp = model_path.lower()
    
    if lp.endswith(".gguf"):
        raise ValueError(f"GGUF files not supported: {model_path}")
    
    base = os.path.basename(lp)
    
    # Check for Flux2 Klein first (more specific)
    if any(ind in base for ind in _FLUX2_KLEIN_INDICATORS):
        return "Flux2Klein"
    
    # Check for SDXL
    if any(ind in base for ind in _SDXL_INDICATORS):
        return "SDXL"
    
    return "SD15"


def detect_model_type_from_state_dict(state_dict: dict) -> str:
    """Detect model type by inspecting state dict keys.
    
    This is more accurate than filename-based detection as it
    examines the actual model architecture.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        'SD15', 'SDXL', or 'Flux2Klein'
    """
    keys = set(state_dict.keys())
    
    # Check for Flux2 Klein specific keys
    flux2_indicators = [
        "double_stream_modulation_img.lin.weight",
        "double_stream_modulation.lin.weight",
    ]
    
    for indicator in flux2_indicators:
        for key in keys:
            if indicator in key:
                return "Flux2Klein"
    
    # Check for double_blocks (Flux architecture)
    if any("double_blocks" in k for k in keys):
        return "Flux2Klein"
    
    # Check for SDXL specific keys
    sdxl_indicators = [
        "conditioner.embedders",
        "model.diffusion_model.label_emb.0.0.weight",
    ]
    
    for indicator in sdxl_indicators:
        if any(indicator in k for k in keys):
            return "SDXL"
    
    return "SD15"


def create_model(
    model_path: Optional[str] = None,
    model_type: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
    vae_path: Optional[str] = None,
) -> AbstractModel:
    """Create a model instance with automatic type detection.
    
    Args:
        model_path: Path to checkpoint file (or diffusion model for Flux2)
        model_type: Explicit type ('SD15', 'SDXL', 'Flux2Klein'), or None to auto-detect
        text_encoder_path: Path to text encoder (Flux2 only)
        vae_path: Path to VAE (Flux2 only)
        
    Returns:
        Configured model instance (not yet loaded)
        
    Example:
        # Auto-detect and load SD1.5/SDXL
        model = create_model("./checkpoints/dreamer.safetensors")
        model.load()
        
        # Flux2 Klein from separate components
        model = create_model(model_type="Flux2Klein")  # auto-detect paths
        model.load()
    """
    _ensure_registry_populated()
    
    if model_type is None:
        model_type = detect_model_type(model_path)
    
    if model_type not in _MODEL_REGISTRY:
        logger.warning(f"Unknown model type '{model_type}', using SD15")
        model_type = "SD15"
    
    # Special handling for Flux2Klein - auto-detect components
    if model_type == "Flux2Klein":
        if model_path is None or text_encoder_path is None:
            diffusion_path, te_path, vae_detected = _find_flux2_components()
            model_path = model_path or diffusion_path
            text_encoder_path = text_encoder_path or te_path  
            vae_path = vae_path or vae_detected
        
        logger.info(f"Creating Flux2Klein model:")
        logger.info(f"  Diffusion model: {model_path}")
        logger.info(f"  Text encoder: {text_encoder_path}")
        logger.info(f"  VAE: {vae_path}")
        
        return _MODEL_REGISTRY[model_type](
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
        )
    
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
