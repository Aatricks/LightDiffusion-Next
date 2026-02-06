"""Abstract base class for all model types in LightDiffusion-Next.

This module defines the contract that all model implementations must follow,
enabling a clean, pluggable architecture where SD15, SDXL, and other models
can be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from src.Core.PipelineContext import PipelineContext


@dataclass
class ModelCapabilities:
    """Describes what a model implementation can do.
    
    This allows the pipeline to adapt its behavior based on
    the loaded model's capabilities.
    """
    # Resolution constraints
    min_resolution: int = 256
    max_resolution: int = 2048
    preferred_resolution: int = 512
    requires_resolution_multiple: int = 64
    
    # Feature support
    supports_hires_fix: bool = True
    supports_img2img: bool = True
    supports_inpainting: bool = False
    supports_controlnet: bool = False
    supports_lora: bool = True  # LoRA compatibility
    
    # Performance hints
    supports_stable_fast: bool = True
    supports_deepcache: bool = True
    supports_tome: bool = True
    
    # Model-specific flags
    uses_dual_clip: bool = False  # SDXL uses dual CLIP
    requires_size_conditioning: bool = False  # SDXL needs size embeddings
    is_flux: bool = False
    is_flux2: bool = False
    
    def validate_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Validate and adjust resolution to meet model requirements.
        
        Args:
            width: Requested width
            height: Requested height
            
        Returns:
            Adjusted (width, height) tuple
        """
        # Clamp to min/max
        width = max(self.min_resolution, min(width, self.max_resolution))
        height = max(self.min_resolution, min(height, self.max_resolution))
        
        # Round to required multiple
        width = (width // self.requires_resolution_multiple) * self.requires_resolution_multiple
        height = (height // self.requires_resolution_multiple) * self.requires_resolution_multiple
        
        return width, height


class AbstractModel(ABC):
    """Abstract base class defining the contract for all model implementations.
    
    Every model type (SD15, SDXL, FLUX, etc.) must implement these methods
    to work with the modular pipeline.
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the model.
        
        Args:
            model_path: Optional path to the model checkpoint
        """
        self.model_path = model_path
        self.model = None
        self.clip = None
        self.vae = None
        self._loaded = False
        self._capabilities: Optional[ModelCapabilities] = None
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return the model's capabilities.
        
        Subclasses should override _create_capabilities() to customize.
        """
        if self._capabilities is None:
            self._capabilities = self._create_capabilities()
        return self._capabilities
    
    @abstractmethod
    def _create_capabilities(self) -> ModelCapabilities:
        """Create and return the capabilities for this model type.
        
        Returns:
            ModelCapabilities instance describing this model's features
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded
    
    @abstractmethod
    def load(self, model_path: str = None) -> "AbstractModel":
        """Load the model from disk.
        
        Args:
            model_path: Optional override for the model path
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        clip_skip: int = -2,
    ) -> tuple[Any, Any]:
        """Encode text prompts into conditioning tensors.
        
        Args:
            prompt: Positive prompt(s) to encode
            negative_prompt: Negative prompt(s) to encode
            clip_skip: Number of CLIP layers to skip from the end
            
        Returns:
            Tuple of (positive_conditioning, negative_conditioning)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        ctx: "PipelineContext",
        positive: Any,
        negative: Any,
        latent_image: Optional[Any] = None,
        start_step: Optional[int] = None,
        last_step: Optional[int] = None,
        disable_noise: bool = False,
    ) -> dict:
        """Generate latents using the sampler.
        
        This is the core generation method that runs the diffusion process.
        
        Args:
            ctx: Pipeline context containing all generation parameters
            positive: Positive conditioning from encode_prompt
            negative: Negative conditioning from encode_prompt
            
        Returns:
            Dictionary containing 'samples' key with generated latents
        """
        pass
    
    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space.
        
        Args:
            latents: Latent tensor to decode
            
        Returns:
            Decoded image tensor in [0, 1] range
        """
        pass
    
    def apply_lora(
        self,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
    ) -> "AbstractModel":
        """Apply a LoRA to the model.
        
        Default implementation attempts to use the standard LoRA loader.
        Subclasses can override for model-specific behavior.
        
        Args:
            lora_name: Name/path of the LoRA file
            strength_model: Strength to apply to the model
            strength_clip: Strength to apply to CLIP
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying LoRA")
        
        try:
            from src.Model import LoRas
            loader = LoRas.LoraLoader()
            result = loader.load_lora(
                lora_name=lora_name,
                strength_model=strength_model,
                strength_clip=strength_clip,
                model=self.model,
                clip=self.clip,
            )
            self.model = result[0]
            self.clip = result[1]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to apply LoRA {lora_name}: {e}")
        
        return self
    
    def apply_stable_fast(self, enable_cuda_graph: bool = True) -> "AbstractModel":
        """Apply StableFast optimization to the model.
        
        Args:
            enable_cuda_graph: Whether to enable CUDA graphs
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying StableFast")
        
        if not self.capabilities.supports_stable_fast:
            import logging
            logging.getLogger(__name__).warning(
                f"Model does not support StableFast, skipping"
            )
            return self
        
        try:
            from src.StableFast import StableFast
            applier = StableFast.ApplyStableFastUnet()
            result = applier.apply_stable_fast(
                enable_cuda_graph=enable_cuda_graph,
                model=self.model,
            )
            self.model = result[0]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"StableFast optimization failed: {e}")
        
        return self
    
    def apply_deepcache(
        self,
        cache_interval: int = 3,
        cache_depth: int = 2,
        start_step: int = 0,
        end_step: int = 1000,
    ) -> "AbstractModel":
        """Apply DeepCache optimization to the model.
        
        Args:
            cache_interval: Steps between cache updates
            cache_depth: U-Net depth for caching
            start_step: Start applying at this timestep
            end_step: Stop applying at this timestep
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying DeepCache")
        
        if not self.capabilities.supports_deepcache:
            import logging
            logging.getLogger(__name__).warning(
                f"Model does not support DeepCache, skipping"
            )
            return self
        
        try:
            from src.WaveSpeed import deepcache_nodes
            deepcache = deepcache_nodes.ApplyDeepCacheOnModel()
            # DeepCache returns a tuple
            result = deepcache.patch(
                model=(self.model,),
                object_to_patch="diffusion_model",
                cache_interval=cache_interval,
                cache_depth=cache_depth,
                start_step=start_step,
                end_step=end_step,
            )
            if isinstance(result, tuple) and len(result) > 0:
                self.model = result[0]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"DeepCache optimization failed: {e}")
        
        return self
    
    def apply_hidiff(self, model_type: str = "auto") -> "AbstractModel":
        """Apply HiDiffusion MSW-MSA attention optimization.
        
        Args:
            model_type: Model type hint ('auto', 'sd15', 'sdxl')
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying HiDiffusion")
        
        try:
            from src.hidiffusion import msw_msa_attention
            optimizer = msw_msa_attention.ApplyMSWMSAAttentionSimple()
            result = optimizer.go(model_type=model_type, model=self.model)
            self.model = result[0]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"HiDiffusion optimization failed: {e}")
        
        return self
    
    def unload(self) -> None:
        """Release model resources and free GPU memory."""
        self.model = None
        self.clip = None
        self.vae = None
        self._loaded = False
        
        # Attempt to free GPU memory
        try:
            from src.Device import Device
            Device.soft_empty_cache(force=True)
        except Exception:
            pass
    
    def __enter__(self) -> "AbstractModel":
        """Context manager entry - load the model."""
        if not self._loaded:
            self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - optionally unload the model."""
        # By default we don't unload on context exit to support caching
        # Subclasses can override if they want different behavior
        pass
