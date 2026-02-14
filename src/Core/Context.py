"""Simplified Context for LightDiffusion-Next Pipeline.

This module provides a clean, minimal state container that replaces
the verbose PipelineContext with a streamlined dataclass structure.

The Context is the single object passed through the entire pipeline,
holding all configuration and intermediate results.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
import random
import time
import torch

# Settings persistence (replaces legacy include/last_seed.txt)
from src.Core.SettingsStore import get_last_seed, set_last_seed


@dataclass
class SamplingConfig:
    """Sampling parameters - all values have sensible defaults."""
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "dpmpp_sde"
    scheduler: str = "ays"
    denoise: float = 1.0
    
    # Multi-scale diffusion
    enable_multiscale: bool = False
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 3
    multiscale_fullres_end: int = 8
    multiscale_intermittent_fullres: bool = False
    
    # CFG optimizations
    cfg_free_enabled: bool = False
    cfg_free_start_percent: float = 70.0
    batched_cfg: bool = True
    dynamic_cfg_rescaling: bool = False
    dynamic_cfg_method: str = "variance"
    dynamic_cfg_percentile: float = 95.0
    dynamic_cfg_target_scale: float = 7.0
    
    # Adaptive noise
    adaptive_noise_enabled: bool = False
    adaptive_noise_method: str = "complexity"
    
    # DeepCache
    deepcache_enabled: bool = False
    deepcache_interval: int = 3
    deepcache_depth: int = 2
    deepcache_start_step: int = 0
    deepcache_end_step: int = 1000
    
    # Token Merging
    tome_enabled: bool = False
    tome_ratio: float = 0.5
    tome_max_downsample: int = 1


@dataclass
class GenerationConfig:
    """Generation parameters for image output."""
    width: int = 512
    height: int = 512
    batch: int = 1
    number: int = 1
    model_path: Optional[str] = None
    refiner_model_path: Optional[str] = None
    refiner_switch_step: Optional[int] = None
    stable_fast: bool = False
    torch_compile: bool = False
    fp8_inference: bool = False
    weight_quantization: Optional[str] = None  # "fp8", "nvfp4", or None
    autohdr: bool = True


@dataclass  
class FeatureFlags:
    """Feature toggles - all optional enhancements."""
    hires_fix: bool = False
    adetailer: bool = False
    enhance_prompt: bool = False
    img2img: bool = False
    img2img_image: Optional[str] = None
    img2img_denoise: float = 0.75  # Denoising strength: 0=no change, 1=full generation
    reuse_seed: bool = False
    # Server-provided request filename prefix for saving outputs (e.g., 'LD-REQ-<rid>')
    request_filename_prefix: Optional[str] = None
    
    # ControlNet settings
    controlnet_model: Optional[str] = None  # Path to ControlNet model
    controlnet_strength: float = 1.0  # Control strength (0-2)
    controlnet_type: str = "canny"  # Preprocessor type: canny, none


@dataclass
class Context:
    """Central state container for a pipeline run.
    
    Usage:
        ctx = Context(prompt="a landscape", width=512, height=512)
        ctx = Pipeline().run(ctx)
        image = ctx.current_image
    """
    # Core prompts
    prompt: Union[str, list[str]] = ""
    negative_prompt: str = ""
    
    # Configs (using composition)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Runtime state
    # Note: PyTorch generators only support seeds up to 2**63 - 1
    seed: int = field(default_factory=lambda: random.randint(1, 2**63 - 1))
    seeds: list[int] = field(default_factory=list)
    
    # Pipeline state (modified during execution)
    current_latents: Optional[torch.Tensor] = None
    current_image: Optional[Any] = None
    positive_cond: Optional[Any] = None
    negative_cond: Optional[Any] = None
    
    # Timing
    start_time: float = field(default_factory=time.time)
    
    # Callbacks
    callback: Optional[Callable] = None
    
    # Default negative
    DEFAULT_NEGATIVE: str = (
        "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), "
        "(embedding:EasyNegative), (embedding:badhandv4)"
    )
    
    def __post_init__(self):
        """Initialize after creation."""
        if not self.negative_prompt:
            self.negative_prompt = self.DEFAULT_NEGATIVE
        if not self.seeds:
            self._generate_seeds()
    
    def _generate_seeds(self) -> None:
        """Generate seeds for all images."""
        total = len(self.prompt) if isinstance(self.prompt, list) else self.generation.number
        total = max(1, total)
        
        if self.features.reuse_seed:
            try:
                ls = get_last_seed()
                if ls is not None:
                    self.seed = int(ls)
            except Exception:
                pass
            self.seeds = [self.seed] * total
        else:
            self.seeds = [random.randint(1, 2**63 - 1) for _ in range(total)]
            self.seed = self.seeds[0]
    
    def save_seed(self) -> None:
        """Persist seed for reuse."""
        try:
            set_last_seed(int(self.seeds[-1] if self.seeds else self.seed))
        except Exception:
            pass
    
    @property
    def is_batched(self) -> bool:
        """Check if this is multi-prompt generation."""
        return isinstance(self.prompt, list)
    
    @property
    def total_images(self) -> int:
        """Total images to generate."""
        if isinstance(self.prompt, list):
            return len(self.prompt)
        return max(1, self.generation.number)
    
    @property
    def width(self) -> int:
        """Shortcut for generation.width."""
        return self.generation.width
    
    @property
    def height(self) -> int:
        """Shortcut for generation.height."""
        return self.generation.height
    
    @property
    def model_path(self) -> Optional[str]:
        """Shortcut for generation.model_path."""
        return self.generation.model_path
    
    def clone(self) -> "Context":
        """Deep copy this context."""
        import copy
        return copy.deepcopy(self)
    
    def with_hires_settings(self, scale: float = 2.0) -> "Context":
        """Create a new context configured for hires fix pass.
        
        Args:
            scale: Upscale factor
            
        Returns:
            New context with hires-appropriate settings
        """
        hires_ctx = self.clone()
        hires_ctx.generation.width = int(self.generation.width * scale)
        hires_ctx.generation.height = int(self.generation.height * scale)
        hires_ctx.sampling.steps = max(10, int(self.sampling.steps * 0.5))
        hires_ctx.sampling.cfg = 8.0
        hires_ctx.sampling.denoise = 0.45
        return hires_ctx
    
    def build_metadata(self, extra: dict = None) -> dict:
        """Build PNG metadata dictionary."""
        # Detect model type from path
        model_type = "Unknown"
        model_path = self.generation.model_path or "None"
        if model_path and model_path != "None":
            try:
                from src.Core.Models.ModelFactory import detect_model_type
                model_type = detect_model_type(model_path)
            except Exception:
                # Fallback to simple detection
                path_lower = model_path.lower()
                if "xl" in path_lower or "sdxl" in path_lower:
                    model_type = "SDXL"
                elif "flux" in path_lower:
                    model_type = "Flux2Klein"
                else:
                    model_type = "SD15"
        
        # Calculate timing metrics
        elapsed = time.time() - self.start_time
        steps = self.sampling.steps
        avg_iters = steps / elapsed if elapsed > 0 else 0
        
        meta = {
            "prompt": str(self.prompt),
            "negative_prompt": str(self.negative_prompt),
            "seed": str(self.seed),
            "sampler": self.sampling.sampler,
            "steps": str(self.sampling.steps),
            "cfg": str(self.sampling.cfg),
            "scheduler": self.sampling.scheduler,
            "denoise": str(self.sampling.denoise),
            "width": str(self.generation.width),
            "height": str(self.generation.height),
            "model_path": str(model_path),
            "model_type": model_type,
            "weight_quantization": str(self.generation.weight_quantization or "none"),
            "hires_fix": str(self.features.hires_fix),
            "adetailer": str(self.features.adetailer),
            "refiner_model": str(self.generation.refiner_model_path or "None"),
            "refiner_switch": str(self.generation.refiner_switch_step or "None"),
            "generation_duration": f"{elapsed:.3f}",
            "avg_iters_per_s": f"{avg_iters:.3f}",
        }
        if extra:
            meta.update(extra)
        return meta
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> "Context":
        """Create Context from legacy pipeline kwargs.
        
        Maps the old 50+ argument style to structured Context.
        """
        ctx = cls()
        
        # Prompts
        ctx.prompt = kwargs.get("prompt", "")
        ctx.negative_prompt = kwargs.get("negative_prompt", ctx.DEFAULT_NEGATIVE)
        
        # Generation
        ctx.generation.width = kwargs.get("w", kwargs.get("width", 512))
        ctx.generation.height = kwargs.get("h", kwargs.get("height", 512))
        ctx.generation.batch = kwargs.get("batch", 1)
        ctx.generation.number = kwargs.get("number", 1)
        ctx.generation.model_path = kwargs.get("model_path")
        ctx.generation.refiner_model_path = kwargs.get("refiner_model_path")
        ctx.generation.refiner_switch_step = kwargs.get("refiner_switch_step")
        ctx.generation.stable_fast = kwargs.get("stable_fast", False)
        ctx.generation.torch_compile = kwargs.get("torch_compile", False)
        ctx.generation.fp8_inference = kwargs.get("fp8_inference", False)
        ctx.generation.weight_quantization = kwargs.get("weight_quantization")
        ctx.generation.autohdr = kwargs.get("autohdr", True)
        
        # Sampling
        ctx.sampling.steps = kwargs.get("steps", 20)
        ctx.sampling.cfg = kwargs.get("cfg_scale", kwargs.get("cfg", 7.0))  # Accept both cfg_scale and cfg
        ctx.sampling.sampler = kwargs.get("sampler", "dpmpp_sde")
        ctx.sampling.scheduler = kwargs.get("scheduler", "ays")
        ctx.sampling.enable_multiscale = kwargs.get("enable_multiscale", False)
        ctx.sampling.multiscale_factor = kwargs.get("multiscale_factor", 0.5)
        ctx.sampling.multiscale_fullres_start = kwargs.get("multiscale_fullres_start", 3)
        ctx.sampling.multiscale_fullres_end = kwargs.get("multiscale_fullres_end", 8)
        ctx.sampling.multiscale_intermittent_fullres = kwargs.get("multiscale_intermittent_fullres", False)
        ctx.sampling.cfg_free_enabled = kwargs.get("cfg_free_enabled", False)
        ctx.sampling.cfg_free_start_percent = kwargs.get("cfg_free_start_percent", 70.0)
        ctx.sampling.batched_cfg = kwargs.get("batched_cfg", True)
        ctx.sampling.dynamic_cfg_rescaling = kwargs.get("dynamic_cfg_rescaling", False)
        ctx.sampling.dynamic_cfg_method = kwargs.get("dynamic_cfg_method", "variance")
        ctx.sampling.dynamic_cfg_percentile = kwargs.get("dynamic_cfg_percentile", 95.0)
        ctx.sampling.dynamic_cfg_target_scale = kwargs.get("dynamic_cfg_target_scale", 7.0)
        ctx.sampling.adaptive_noise_enabled = kwargs.get("adaptive_noise_enabled", False)
        ctx.sampling.adaptive_noise_method = kwargs.get("adaptive_noise_method", "complexity")
        ctx.sampling.deepcache_enabled = kwargs.get("deepcache_enabled", False)
        ctx.sampling.deepcache_interval = kwargs.get("deepcache_interval", 3)
        ctx.sampling.deepcache_depth = kwargs.get("deepcache_depth", 2)
        ctx.sampling.deepcache_start_step = kwargs.get("deepcache_start_step", 0)
        ctx.sampling.deepcache_end_step = kwargs.get("deepcache_end_step", 1000)
        ctx.sampling.tome_enabled = kwargs.get("tome_enabled", False)
        ctx.sampling.tome_ratio = kwargs.get("tome_ratio", 0.5)
        ctx.sampling.tome_max_downsample = kwargs.get("tome_max_downsample", 1)
        
        # Callbacks
        ctx.callback = kwargs.get("callback")
        
        # Features
        ctx.features.hires_fix = kwargs.get("hires_fix", False)
        ctx.features.adetailer = kwargs.get("adetailer", False)
        ctx.features.enhance_prompt = kwargs.get("enhance_prompt", False)
        ctx.features.img2img = kwargs.get("img2img", False)
        ctx.features.img2img_image = kwargs.get("img2img_image")
        ctx.features.img2img_denoise = kwargs.get("img2img_denoise", 0.75)
        ctx.features.reuse_seed = kwargs.get("reuse_seed", False)
        ctx.features.request_filename_prefix = kwargs.get("request_filename_prefix")
        
        # ControlNet
        ctx.features.controlnet_model = kwargs.get("controlnet_model")
        ctx.features.controlnet_strength = kwargs.get("controlnet_strength", 1.0)
        ctx.features.controlnet_type = kwargs.get("controlnet_type", "canny")
        
        # Handle multiscale preset
        preset = kwargs.get("multiscale_preset")
        if preset and preset != "disabled":
            try:
                from src.sample.multiscale_presets import get_preset_parameters
                params = get_preset_parameters(preset)
                # Only overwrite if explicitly enabled in kwargs or if not specified
                if kwargs.get("enable_multiscale") is not False:
                    ctx.sampling.enable_multiscale = params["enable_multiscale"]
                    ctx.sampling.multiscale_factor = params["multiscale_factor"]
                    ctx.sampling.multiscale_fullres_start = params["multiscale_fullres_start"]
                    ctx.sampling.multiscale_fullres_end = params["multiscale_fullres_end"]
                    ctx.sampling.multiscale_intermittent_fullres = params["multiscale_intermittent_fullres"]
            except Exception:
                pass
        elif preset == "disabled":
            ctx.sampling.enable_multiscale = False
        
        # Regenerate seeds after setting reuse_seed
        ctx._generate_seeds()
        
        return ctx
