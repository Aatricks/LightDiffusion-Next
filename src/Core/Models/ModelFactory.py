"""Model factory for LightDiffusion-Next.

Provides automatic model type detection and instantiation.
"""

import logging
import os
from typing import Optional

from src.Core.AbstractModel import AbstractModel
from src.Core.Models.SD15Model import SD15Model
from src.Core.Models.SDXLModel import SDXLModel


class ModelFactory:
    """Factory for creating model instances based on type detection.
    
    Automatically detects the model type from the checkpoint file
    and returns the appropriate model adapter.
    """
    
    # Model type registry
    _model_types = {
        "SD15": SD15Model,
        "SDXL": SDXLModel,
    }
    
    @classmethod
    def detect_model_type(cls, model_path: str) -> str:
        """Detect model type from file path/name.
        
        Uses heuristics based on filename and file contents
        to determine the model type.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Model type string ('SD15', 'SDXL')
        """
        if model_path is None:
            return "SD15"
        
        lp = model_path.lower()
        
        # GGUF files are no longer supported
        if lp.endswith(".gguf"):
            raise ValueError(
                f"GGUF files are no longer supported: {model_path}. "
                "Please use .safetensors or .pt models instead."
            )
        
        # Check for SDXL indicators in filename
        base = os.path.basename(lp)
        sdxl_indicators = ["sdxl", "refiner", "hassaku", "juggernaut", "xl"]
        
        for indicator in sdxl_indicators:
            if indicator in base.lower():
                return "SDXL"
        
        # Default to SD1.5
        return "SD15"
    
    @classmethod
    def create(cls, model_path: str = None, model_type: str = None) -> AbstractModel:
        """Create a model instance.
        
        Args:
            model_path: Path to the model checkpoint
            model_type: Explicit model type (if known), otherwise auto-detected
            
        Returns:
            Appropriate model adapter instance
        """
        logger = logging.getLogger(__name__)
        
        # Auto-detect type if not specified
        if model_type is None:
            model_type = cls.detect_model_type(model_path)
        
        # Validate model type
        if model_type not in cls._model_types:
            logger.warning(f"Unknown model type '{model_type}', defaulting to SD15")
            model_type = "SD15"
        
        # Create and return model instance
        model_class = cls._model_types[model_type]
        logger.info(f"Creating {model_type} model from {model_path}")
        
        return model_class(model_path=model_path)
    
    @classmethod
    def register_model_type(cls, type_name: str, model_class: type) -> None:
        """Register a new model type.
        
        Allows extending the factory with custom model implementations.
        
        Args:
            type_name: Name for the model type
            model_class: Class that inherits from AbstractModel
        """
        if not issubclass(model_class, AbstractModel):
            raise TypeError(f"{model_class} must inherit from AbstractModel")
        
        cls._model_types[type_name] = model_class
        logging.getLogger(__name__).info(f"Registered model type: {type_name}")
    
    @classmethod
    def list_model_types(cls) -> list[str]:
        """List all registered model types.
        
        Returns:
            List of registered model type names
        """
        return list(cls._model_types.keys())


def create_model(model_path: str = None, model_type: str = None) -> AbstractModel:
    """Convenience function for creating models.
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Explicit model type (optional)
        
    Returns:
        Model instance
        
    Example:
        model = create_model("./include/checkpoints/my_model.safetensors")
        model.load()
        
        # Or with explicit type:
        model = create_model(model_path, model_type="SDXL")
    """
    return ModelFactory.create(model_path, model_type)
