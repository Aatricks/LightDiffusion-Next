"""Protocol definitions for LightDiffusion-Next.

This module defines the contracts (interfaces) that all components must follow.
Using Protocol allows for structural subtyping without requiring explicit inheritance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from src.Core.Context import Context


# ============================================================================
# MODEL PROTOCOLS
# ============================================================================

@dataclass
class ModelCapabilities:
    """Describes what a model implementation can do."""
    min_resolution: int = 256
    max_resolution: int = 2048
    preferred_resolution: int = 512
    requires_resolution_multiple: int = 64
    supports_hires_fix: bool = True
    supports_img2img: bool = True
    supports_inpainting: bool = False
    supports_controlnet: bool = False
    supports_stable_fast: bool = True
    supports_deepcache: bool = True
    supports_tome: bool = True
    uses_dual_clip: bool = False
    requires_size_conditioning: bool = False
    
    def validate_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Clamp and round resolution to model requirements."""
        width = max(self.min_resolution, min(width, self.max_resolution))
        height = max(self.min_resolution, min(height, self.max_resolution))
        width = (width // self.requires_resolution_multiple) * self.requires_resolution_multiple
        height = (height // self.requires_resolution_multiple) * self.requires_resolution_multiple
        return width, height


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol defining the contract for all model implementations."""
    
    model: Any
    clip: Any
    vae: Any
    model_path: str
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
    
    def load(self, model_path: str = None) -> "ModelProtocol":
        """Load model from disk."""
        ...
    
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        clip_skip: int = -2,
    ) -> tuple[Any, Any]:
        """Encode prompts to conditioning."""
        ...
    
    def generate(
        self,
        ctx: "Context",
        positive: Any,
        negative: Any,
    ) -> dict:
        """Generate latents."""
        ...
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        ...
    
    def apply_lora(
        self,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
    ) -> "ModelProtocol":
        """Apply LoRA weights."""
        ...
    
    def apply_stable_fast(self, enable_cuda_graph: bool = True) -> "ModelProtocol":
        """Apply StableFast optimization."""
        ...
    
    def apply_deepcache(
        self,
        cache_interval: int = 3,
        cache_depth: int = 2,
        start_step: int = 0,
        end_step: int = 1000,
    ) -> "ModelProtocol":
        """Apply DeepCache optimization."""
        ...
    
    def unload(self) -> None:
        """Release model resources."""
        ...


# ============================================================================
# PROCESSOR PROTOCOLS
# ============================================================================

@runtime_checkable
class ProcessorProtocol(Protocol):
    """Protocol for pipeline processors (plugins).
    
    Processors are stateless components that can optionally modify
    the pipeline context based on feature flags.
    """
    
    @staticmethod
    def is_enabled(ctx: "Context") -> bool:
        """Check if this processor should run for given context."""
        ...
    
    @staticmethod
    def process(
        ctx: "Context",
        model: ModelProtocol,
        **kwargs
    ) -> "Context":
        """Process the context, potentially modifying latents/images.
        
        Args:
            ctx: Pipeline context (may be modified)
            model: Loaded model for any re-sampling needed
            **kwargs: Processor-specific arguments
            
        Returns:
            Modified context
        """
        ...


class BaseProcessor(ABC):
    """Abstract base class for processors providing common functionality."""
    
    @staticmethod
    @abstractmethod
    def is_enabled(ctx: "Context") -> bool:
        """Check if this processor should run."""
        pass
    
    @staticmethod
    @abstractmethod
    def process(ctx: "Context", model: ModelProtocol, **kwargs) -> "Context":
        """Process the context."""
        pass
    
    @classmethod
    def run_if_enabled(cls, ctx: "Context", model: ModelProtocol, **kwargs) -> "Context":
        """Convenience method to conditionally run processor."""
        if cls.is_enabled(ctx):
            return cls.process(ctx, model, **kwargs)
        return ctx


# ============================================================================
# SAMPLER PROTOCOLS
# ============================================================================

@runtime_checkable
class SamplerProtocol(Protocol):
    """Protocol for diffusion samplers."""
    
    def sample(
        self,
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict = None,
        callback: Any = None,
        disable: bool = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the sampling loop.
        
        Args:
            model: The denoising model
            x: Initial noisy latents
            sigmas: Noise schedule
            extra_args: Model-specific arguments
            callback: Progress callback
            disable: Disable progress bar
            **kwargs: Sampler-specific options
            
        Returns:
            Denoised latents
        """
        ...


# ============================================================================
# CFG SCHEDULER PROTOCOLS
# ============================================================================

@runtime_checkable
class CFGSchedulerProtocol(Protocol):
    """Protocol for CFG scheduling strategies."""
    
    def get_cfg(self, step: int, total_steps: int, base_cfg: float) -> float:
        """Get CFG scale for a given step.
        
        Args:
            step: Current step (0-indexed)
            total_steps: Total number of steps
            base_cfg: Base CFG value
            
        Returns:
            CFG scale to use for this step
        """
        ...


class ConstantCFGScheduler:
    """Default CFG scheduler - constant value throughout."""
    
    def get_cfg(self, step: int, total_steps: int, base_cfg: float) -> float:
        return base_cfg


class CFGFreeScheduler:
    """CFG-free sampling - drops to CFG=1 after a percentage of steps."""
    
    def __init__(self, start_percent: float = 70.0):
        self.start_percent = start_percent
    
    def get_cfg(self, step: int, total_steps: int, base_cfg: float) -> float:
        progress = (step / max(1, total_steps - 1)) * 100
        if progress >= self.start_percent:
            return 1.0
        return base_cfg


class LinearDecayCFGScheduler:
    """Linearly decay CFG from base to target."""
    
    def __init__(self, target_cfg: float = 1.0):
        self.target_cfg = target_cfg
    
    def get_cfg(self, step: int, total_steps: int, base_cfg: float) -> float:
        progress = step / max(1, total_steps - 1)
        return base_cfg + (self.target_cfg - base_cfg) * progress


# ============================================================================
# PIPELINE PROTOCOL
# ============================================================================

@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol for the main generation pipeline."""
    
    def run(self, ctx: "Context") -> "Context":
        """Execute the full generation pipeline.
        
        Args:
            ctx: Configured context with all parameters
            
        Returns:
            Context with generated images in current_image
        """
        ...


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Conditioning tuple type (used throughout the codebase)
Conditioning = list[tuple[torch.Tensor, dict[str, Any]]]

# Latent dict type
LatentDict = dict[str, torch.Tensor]
