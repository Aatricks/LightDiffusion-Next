"""Model adapters for LightDiffusion-Next.

This module provides concrete model implementations that inherit from
AbstractModel and wrap the existing model infrastructure.

These adapters provide a clean interface while reusing the existing
heavily-tested model loading and inference code.
"""

from src.Core.Models.SD15Model import SD15Model
from src.Core.Models.SDXLModel import SDXLModel
from src.Core.Models.ModelFactory import (
    create_model,
    detect_model_type,
    register_model_type,
    list_model_types,
    list_available_models,
)

__all__ = [
    "SD15Model",
    "SDXLModel",
    "create_model",
    "detect_model_type",
    "register_model_type",
    "list_model_types",
    "list_available_models",
]
