"""Abstract base class for all model types in LightDiffusion-Next.

This module defines the contract that all model implementations must follow,
enabling a clean, pluggable architecture where SD15, SDXL, and other models
can be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

if TYPE_CHECKING:
    from src.Core.Context import Context


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
        # Maintain aspect ratio when clamping to max_resolution
        if width > self.max_resolution or height > self.max_resolution:
            scale = min(self.max_resolution / width, self.max_resolution / height)
            width = int(width * scale)
            height = int(height * scale)

        # Clamp to minimum
        width = max(self.min_resolution, width)
        height = max(self.min_resolution, height)
        
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
        ctx: "Context",
        positive: Any,
        negative: Any,
        latent_image: Optional[Any] = None,
        start_step: Optional[int] = None,
        last_step: Optional[int] = None,
        disable_noise: bool = False,
        callback: Optional[Callable] = None,
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
    
    def apply_fp8(self) -> "AbstractModel":
        """Apply FP8 quantization to the diffusion model weights.
        
        Hardware-gated: only applies on supported GPUs (Ada Lovelace 8.9+, Hopper 9.0+).
        Reduces memory usage by ~50% vs FP16 with minimal quality impact.
        
        After casting weights to FP8, enables comfy_cast_weights on all affected
        modules so that forward() uses cast_bias_weight() to upcast FP8 weights
        to the input dtype at runtime, preventing dtype mismatch errors.
        
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying FP8")
        
        try:
            from src.Device import Device
            from src.cond.cast import CastWeightBiasOp
            if not Device.is_fp8_supported():
                import logging
                logging.getLogger(__name__).info(
                    "FP8 not supported on this GPU (requires compute capability 8.9+), skipping"
                )
                return self
            
            inner = getattr(self.model, 'model', self.model)
            # Try common diffusion submodule names, otherwise fall back to top-level module
            diff_model = getattr(inner, 'diffusion_model', None)
            if diff_model is None:
                import torch.nn as nn
                if isinstance(inner, nn.Module):
                    diff_model = inner
                    import logging
                    logging.getLogger(__name__).info(
                        "No 'diffusion_model' submodule found; using top-level model for FP8 quantization"
                    )
                else:
                    import logging
                    logging.getLogger(__name__).warning("No diffusion_model found for FP8 quantization")
                    return self
            
            converted = 0
            cast_enabled = 0
            for name, module in diff_model.named_modules():
                # Quantize weight parameters to FP8
                if hasattr(module, 'weight') and module.weight is not None:
                    w = module.weight
                    if w.dtype in (torch.float16, torch.bfloat16, torch.float32) and w.ndim >= 2:
                        module.weight.data = Device.cast_to_fp8(w.data)
                        converted += 1
                        # Enable runtime casting so forward() upcasts FP8→input dtype
                        if isinstance(module, CastWeightBiasOp):
                            module.comfy_cast_weights = True
                            cast_enabled += 1
            
            import logging
            logging.getLogger(__name__).info(
                f"FP8 quantization applied to {converted} weight tensors, "
                f"runtime casting enabled on {cast_enabled} modules"
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"FP8 quantization failed: {e}")
        
        return self

    def apply_nvfp4(self) -> "AbstractModel":
        """Apply NVFP4 (4-bit) quantization to the diffusion model weights.
        
        Reduces memory usage by ~75% vs FP16 with some quality impact.
        
        After quantizing weights to NVFP4, enables comfy_cast_weights on all affected
        modules so that forward() uses cast_bias_weight() to dequantize NVFP4 weights
        to the input dtype at runtime.
        
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying NVFP4")
        
        try:
            from src.cond.cast import CastWeightBiasOp
            from src.Utilities.Quantization import quantize_nvfp4
            
            inner = getattr(self.model, 'model', self.model)
            diff_model = getattr(inner, 'diffusion_model', None)
            if diff_model is None:
                import torch.nn as nn
                if isinstance(inner, nn.Module):
                    diff_model = inner
                else:
                    import logging
                    logging.getLogger(__name__).warning("No diffusion_model found for NVFP4 quantization")
                    return self
            
            converted = 0
            cast_enabled = 0
            for name, module in diff_model.named_modules():
                # Quantize weight parameters to NVFP4
                if hasattr(module, 'weight') and module.weight is not None:    
                    w = module.weight
                    if w.dtype in (torch.float16, torch.bfloat16, torch.float32) and w.ndim == 2 and w.numel() > 4096:
                        from src.Utilities.Quantization import quantize_nvfp4, from_blocked
                        q_weight, tensor_scale, blocked_scales = quantize_nvfp4(w.data)

                        module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
                        module.quant_format = "nvfp4"
                        
                        # Pre-de-block scales to save compute during inference
                        rows, cols = w.shape
                        block_cols = (cols + 15) // 16
                        deblocked_scales = from_blocked(blocked_scales, rows, block_cols)
                        
                        import torch.nn as nn
                        if isinstance(module, nn.Module):
                            module.register_buffer("weight_scale_2", tensor_scale)
                            module.register_buffer("weight_scale", deblocked_scales)
                        else:
                            module.weight_scale_2 = tensor_scale
                            module.weight_scale = deblocked_scales
                            
                        module.original_shape = w.shape

                        converted += 1
                        # Enable runtime casting so forward() dequantizes NVFP4→input dtype
                        if isinstance(module, CastWeightBiasOp):
                            module.comfy_cast_weights = True
                            cast_enabled += 1            
            import logging
            logging.getLogger(__name__).info(
                f"NVFP4 quantization applied to {converted} weight tensors, "
                f"runtime casting enabled on {cast_enabled} modules"
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"NVFP4 quantization failed: {e}")
        
        return self
    
    def apply_torch_compile(self, mode: str = "max-autotune-no-cudagraphs") -> "AbstractModel":
        """Apply torch.compile optimization to the model.
        
        Uses 'max-autotune-no-cudagraphs' by default to get autotuning benefits
        without CUDA graph fragility (which causes assertion errors with dynamic
        model state like LoRA patches and mixed dtypes).
        
        Args:
            mode: Compilation mode - 'max-autotune-no-cudagraphs' (recommended),
                  'max-autotune', 'default', or 'reduce-overhead'
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying torch.compile")
        
        try:
            from src.Device import Device
            if not hasattr(torch, 'compile'):
                import logging
                logging.getLogger(__name__).warning("torch.compile requires PyTorch 2.0+, skipping")
                return self
            
            Device.enable_torch_compile(True)
            inner = getattr(self.model, 'model', self.model)
            # Try to find a diffusion submodule; if missing, fall back to compiling the top-level module
            diff_model = getattr(inner, 'diffusion_model', None)
            if diff_model is None:
                import torch.nn as nn
                if isinstance(inner, nn.Module):
                    # Compile the top-level module for models without a diffusion wrapper (Flux2, etc.)
                    compiled = Device.compile_model(inner, mode=mode)
                    if compiled is not inner:
                        # If compile returns a Module we can safely replace the module.
                        try:
                            import torch.nn as _nn
                            if isinstance(compiled, _nn.Module):
                                if hasattr(self.model, 'model'):
                                    self.model.model = compiled
                                else:
                                    self.model = compiled
                                import logging
                                logging.getLogger(__name__).info(f"torch.compile applied to top-level model (mode={mode})")
                            elif callable(compiled):
                                # Preserve the original module instance but attach the compiled
                                # callable to its forward method so attribute access (e.g. latent_format)
                                # continues to work while runtime calls go through the compiled code.
                                try:
                                    import types
                                    # attach compiled function to the inner module so forward calls use it
                                    setattr(inner, '_compiled_fn', compiled)
                                    def _compiled_forward(self, *args, **kwargs):
                                        return self._compiled_fn(*args, **kwargs)
                                    inner.forward = types.MethodType(_compiled_forward, inner)
                                    import logging
                                    logging.getLogger(__name__).info(f"torch.compile returned callable; attached compiled forward to top-level module (mode={mode})")
                                except Exception:
                                    import logging
                                    logging.getLogger(__name__).warning("Failed to attach compiled callable to module.forward; leaving original module intact")
                            else:
                                import logging
                                logging.getLogger(__name__).info(f"torch.compile returned unexpected type {type(compiled)}; leaving original model intact")
                        except Exception:
                            import logging
                            logging.getLogger(__name__).info(f"torch.compile returned a new object but could not reassign it; compiled object is available (mode={mode})")
                else:
                    import logging
                    logging.getLogger(__name__).warning("No diffusion_model found for torch.compile")
            else:
                compiled = Device.compile_model(diff_model, mode=mode)
                if compiled is not diff_model:
                    # If compiled returned an nn.Module, replace the diffusion_model.
                    import torch.nn as _nn
                    if isinstance(compiled, _nn.Module):
                        inner.diffusion_model = compiled
                        import logging
                        logging.getLogger(__name__).info(f"torch.compile applied to diffusion model (mode={mode})")
                    elif callable(compiled):
                        # Attach compiled callable to the diffusion_model.forward so callers
                        # (e.g. model.apply_model) continue to operate with the same
                        # argument mapping while using compiled execution.
                        try:
                            import types
                            if hasattr(inner, 'diffusion_model'):
                                dm = inner.diffusion_model
                                setattr(dm, '_compiled_fn', compiled)
                                def _compiled_forward(self, *args, **kwargs):
                                    return self._compiled_fn(*args, **kwargs)
                                dm.forward = types.MethodType(_compiled_forward, dm)
                                import logging
                                logging.getLogger(__name__).info(f"torch.compile returned callable for diffusion_model; attached compiled forward (mode={mode})")
                            else:
                                import logging
                                logging.getLogger(__name__).info(f"torch.compile returned callable but no diffusion_model to attach to; compiled available (mode={mode})")
                        except Exception:
                            import logging
                            logging.getLogger(__name__).warning("Failed to attach compiled callable to diffusion_model.forward")
                    else:
                        import logging
                        logging.getLogger(__name__).info(f"torch.compile returned unexpected type {type(compiled)} for diffusion_model; leaving original module intact")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"torch.compile optimization failed: {e}")
        
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
        
        # Force garbage collection to release tensor references
        import gc
        gc.collect()
        
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
