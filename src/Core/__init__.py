"""Core module for LightDiffusion-Next.

This module provides the foundational abstractions for the modular pipeline:
- AbstractModel: Base class defining the contract for all model types (SD15, SDXL)
- PipelineContext: State container holding all configuration and intermediate results
- Models: Concrete model implementations (SD15Model, SDXLModel)
"""

from src.Core.AbstractModel import AbstractModel, ModelCapabilities
from src.Core.PipelineContext import PipelineContext
from src.Core.Models import SD15Model, SDXLModel, ModelFactory, create_model

__all__ = [
    "AbstractModel",
    "ModelCapabilities",
    "PipelineContext",
    "SD15Model",
    "SDXLModel",
    "ModelFactory",
    "create_model",
]
