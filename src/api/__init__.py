"""API module for LightDiffusion-Next.

Provides unified service layer for all frontends.
"""

from src.api.GenerationService import (
    GenerationService,
    GenerationRequest,
    GenerationResult,
    get_generation_service,
)

__all__ = [
    "GenerationService",
    "GenerationRequest", 
    "GenerationResult",
    "get_generation_service",
]
