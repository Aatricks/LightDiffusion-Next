"""Core module for LightDiffusion-Next.

This module provides the foundational abstractions for the modular pipeline:
- Context: Simplified state container for all pipeline state
- Interfaces: Protocol definitions (ModelProtocol, ProcessorProtocol, etc.)
- Pipeline: Main orchestrator class
- AbstractModel: Base class for model implementations
- Models: Concrete implementations (SD15Model, SDXLModel)
"""

from src.Core.Context import Context, SamplingConfig, GenerationConfig, FeatureFlags
from src.Core.Interfaces import (
    ModelCapabilities,
    ModelProtocol,
    ProcessorProtocol,
    BaseProcessor,
    SamplerProtocol,
    CFGSchedulerProtocol,
    ConstantCFGScheduler,
    CFGFreeScheduler,
    LinearDecayCFGScheduler,
    PipelineProtocol,
)
from src.Core.AbstractModel import AbstractModel
from src.Core.Pipeline import Pipeline, PipelineResult, get_default_pipeline
from src.Core.Models import SD15Model, SDXLModel, create_model, detect_model_type

__all__ = [
    # Context API
    "Context",
    "SamplingConfig",
    "GenerationConfig",
    "FeatureFlags",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "get_default_pipeline",
    # Protocols
    "ModelCapabilities",
    "ModelProtocol",
    "ProcessorProtocol",
    "BaseProcessor",
    "SamplerProtocol",
    "CFGSchedulerProtocol",
    "ConstantCFGScheduler",
    "CFGFreeScheduler",
    "LinearDecayCFGScheduler",
    "PipelineProtocol",
    # Models
    "AbstractModel",
    "SD15Model",
    "SDXLModel",
    "create_model",
    "detect_model_type",
]
