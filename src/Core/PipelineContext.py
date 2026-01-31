"""Pipeline context for LightDiffusion-Next.

This module provides a clean state container that replaces passing
50+ arguments through function calls. The PipelineContext holds all
configuration and intermediate results for a generation run.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import random
import time
import torch


@dataclass
class SamplingConfig:
    """Configuration for the sampling process."""
    
    # Core sampling parameters
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "dpmpp_sde_cfgpp"
    scheduler: str = "ays"
    denoise: float = 1.0
    
    # Multi-scale diffusion
    enable_multiscale: bool = True
    multiscale_preset: Optional[str] = "quality"
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 3
    multiscale_fullres_end: int = 8
    multiscale_intermittent_fullres: bool = False
    
    # CFG-free sampling
    cfg_free_enabled: bool = False
    cfg_free_start_percent: float = 70.0
    
    # DeepCache acceleration
    deepcache_enabled: bool = False
    deepcache_interval: int = 3
    deepcache_depth: int = 2
    deepcache_start_step: int = 0
    deepcache_end_step: int = 1000
    
    # Token Merging
    tome_enabled: bool = False
    tome_ratio: float = 0.5
    tome_max_downsample: int = 1
    
    # Advanced CFG optimizations
    batched_cfg: bool = True
    dynamic_cfg_rescaling: bool = False
    dynamic_cfg_method: str = "variance"
    dynamic_cfg_percentile: float = 95.0
    dynamic_cfg_target_scale: float = 7.0
    adaptive_noise_enabled: bool = False
    adaptive_noise_method: str = "complexity"
    
    def apply_multiscale_preset(self) -> None:
        """Apply multiscale preset parameters if specified."""
        if self.multiscale_preset is not None:
            try:
                from src.sample.multiscale_presets import get_preset_parameters
                params = get_preset_parameters(self.multiscale_preset)
                self.enable_multiscale = params["enable_multiscale"]
                self.multiscale_factor = params["multiscale_factor"]
                self.multiscale_fullres_start = params["multiscale_fullres_start"]
                self.multiscale_fullres_end = params["multiscale_fullres_end"]
                self.multiscale_intermittent_fullres = params["multiscale_intermittent_fullres"]
            except Exception:
                pass


@dataclass
class GenerationConfig:
    """Configuration for the overall generation."""
    
    # Image dimensions
    width: int = 512
    height: int = 512
    batch: int = 1
    number: int = 1
    
    # Model selection
    model_path: Optional[str] = None
    realistic_model: bool = False
    
    # Optimizations
    stable_fast: bool = False
    
    # Post-processing
    autohdr: bool = True


@dataclass
class FeatureFlags:
    """Feature toggles for the pipeline."""
    
    # Enhancement features
    hires_fix: bool = False
    adetailer: bool = False
    enhance_prompt: bool = False
    
    # Mode switches
    img2img: bool = False
    img2img_image: Optional[str] = None
    
    # Seed control
    reuse_seed: bool = False


@dataclass
class PipelineContext:
    """Central state container for a pipeline run.
    
    This replaces the 50+ argument passing pattern in the original pipeline
    with a clean, organized state object.
    
    Usage:
        ctx = PipelineContext(prompt="a beautiful landscape", width=512, height=512)
        ctx.sampling.steps = 30
        ctx.features.hires_fix = True
        
        # Generate
        image = model.generate(ctx, positive, negative)
    """
    
    # Prompts
    prompt: str | list[str] = ""
    negative_prompt: str = ""
    
    # Sub-configurations
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Runtime state (set during generation)
    seed: int = field(default_factory=lambda: random.randint(1, 2**64))
    seeds: list[int] = field(default_factory=list)
    
    # Intermediate results
    current_latents: Optional[torch.Tensor] = None
    current_image: Optional[torch.Tensor] = None
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    
    # Default negative prompt
    _default_negative: str = (
        "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), "
        "(embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)"
    )
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        # Apply multiscale preset if specified
        self.sampling.apply_multiscale_preset()
        
        # Set default negative prompt if empty
        if not self.negative_prompt or self.negative_prompt.strip() == "":
            self.negative_prompt = self._default_negative
        
        # Generate seeds if not set
        if not self.seeds:
            self._generate_seeds()
    
    def _generate_seeds(self) -> None:
        """Generate seeds for all images to be created."""
        total = self._calculate_total_images()
        
        if self.features.reuse_seed:
            # Load last used seed
            try:
                with open("./include/last_seed.txt", "r") as f:
                    last_seed = int(f.read().strip())
                self.seeds = [last_seed] * total
                self.seed = last_seed
            except Exception:
                self.seeds = [self.seed] * total
        else:
            self.seeds = [random.randint(1, 2**64) for _ in range(total)]
            self.seed = self.seeds[0] if self.seeds else random.randint(1, 2**64)
    
    def _calculate_total_images(self) -> int:
        """Calculate total number of images to generate."""
        if isinstance(self.prompt, (list, tuple)):
            return len(self.prompt)
        return max(1, self.generation.number)
    
    def save_last_seed(self) -> None:
        """Save the last used seed for potential reuse."""
        try:
            seed_to_save = self.seeds[-1] if self.seeds else self.seed
            with open("./include/last_seed.txt", "w") as f:
                f.write(str(seed_to_save))
        except Exception:
            pass
    
    @property
    def is_batched(self) -> bool:
        """Check if this is a batched (multi-prompt) generation."""
        return isinstance(self.prompt, (list, tuple))
    
    @property
    def total_images(self) -> int:
        """Get total number of images to generate."""
        return self._calculate_total_images()
    
    def get_prompt_at(self, index: int) -> str:
        """Get prompt at a specific index (for batched mode)."""
        if isinstance(self.prompt, (list, tuple)):
            return self.prompt[index] if index < len(self.prompt) else self.prompt[-1]
        return self.prompt
    
    def get_negative_at(self, index: int) -> str:
        """Get negative prompt at a specific index (for batched mode)."""
        if isinstance(self.negative_prompt, (list, tuple)):
            return self.negative_prompt[index] if index < len(self.negative_prompt) else self.negative_prompt[-1]
        return self.negative_prompt
    
    def get_seed_at(self, index: int) -> int:
        """Get seed at a specific index."""
        if index < len(self.seeds):
            return self.seeds[index]
        return self.seed
    
    def build_metadata(self, extra: dict = None) -> dict:
        """Build PNG metadata dictionary for saved images.
        
        Args:
            extra: Additional metadata to include
            
        Returns:
            Metadata dictionary for image saving
        """
        import time
        
        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "prompt": self.prompt if isinstance(self.prompt, str) else str(self.prompt),
            "negative_prompt": self.negative_prompt if isinstance(self.negative_prompt, str) else str(self.negative_prompt),
            "seed": str(self.seed),
            "sampler": self.sampling.sampler,
            "steps": str(self.sampling.steps),
            "cfg": str(self.sampling.cfg),
            "scheduler": self.sampling.scheduler,
            "denoise": str(self.sampling.denoise),
            "width": str(self.generation.width),
            "height": str(self.generation.height),
            "batch_size": str(self.generation.batch),
            "hires_fix": str(self.features.hires_fix),
            "adetailer": str(self.features.adetailer),
            "stable_fast": str(self.generation.stable_fast),
            "realistic_model": str(self.generation.realistic_model),
            "reuse_seed": str(self.features.reuse_seed),
            "multiscale_preset": str(self.sampling.multiscale_preset),
        }
        
        if extra:
            meta.update(extra)
        
        return meta
    
    def clone(self) -> "PipelineContext":
        """Create a deep copy of this context.
        
        Useful for creating modified contexts for sub-operations
        (e.g., hires fix pass with different settings).
        """
        import copy
        return copy.deepcopy(self)
    
    def with_hires_settings(self, scale: float = 2.0) -> "PipelineContext":
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
        hires_ctx.sampling.cfg = 8
        hires_ctx.sampling.denoise = 0.45
        return hires_ctx
    
    @classmethod
    def from_kwargs(cls, **kwargs) -> "PipelineContext":
        """Create a PipelineContext from keyword arguments.
        
        This factory method maps the old-style 50+ arguments to the
        new structured context format.
        
        Args:
            **kwargs: All the old pipeline() arguments
            
        Returns:
            Configured PipelineContext
        """
        ctx = cls()
        
        # Map prompts
        ctx.prompt = kwargs.get("prompt", "")
        ctx.negative_prompt = kwargs.get("negative_prompt", ctx._default_negative)
        
        # Map generation config
        ctx.generation.width = kwargs.get("w", kwargs.get("width", 512))
        ctx.generation.height = kwargs.get("h", kwargs.get("height", 512))
        ctx.generation.batch = kwargs.get("batch", 1)
        ctx.generation.number = kwargs.get("number", 1)
        ctx.generation.model_path = kwargs.get("model_path")
        ctx.generation.realistic_model = kwargs.get("realistic_model", False)
        ctx.generation.stable_fast = kwargs.get("stable_fast", False)
        ctx.generation.autohdr = kwargs.get("autohdr", True)
        
        # Map sampling config
        ctx.sampling.steps = kwargs.get("steps", 20)
        ctx.sampling.cfg = kwargs.get("cfg", 7.0)
        ctx.sampling.sampler = kwargs.get("sampler", "dpmpp_sde_cfgpp")
        ctx.sampling.scheduler = kwargs.get("scheduler", "ays")
        ctx.sampling.multiscale_preset = kwargs.get("multiscale_preset", "quality")
        ctx.sampling.enable_multiscale = kwargs.get("enable_multiscale", True)
        ctx.sampling.multiscale_factor = kwargs.get("multiscale_factor", 0.5)
        ctx.sampling.multiscale_fullres_start = kwargs.get("multiscale_fullres_start", 3)
        ctx.sampling.multiscale_fullres_end = kwargs.get("multiscale_fullres_end", 8)
        ctx.sampling.multiscale_intermittent_fullres = kwargs.get("multiscale_intermittent_fullres", False)
        ctx.sampling.deepcache_enabled = kwargs.get("deepcache_enabled", False)
        ctx.sampling.deepcache_interval = kwargs.get("deepcache_interval", 3)
        ctx.sampling.deepcache_depth = kwargs.get("deepcache_depth", 2)
        ctx.sampling.deepcache_start_step = kwargs.get("deepcache_start_step", 0)
        ctx.sampling.deepcache_end_step = kwargs.get("deepcache_end_step", 1000)
        ctx.sampling.cfg_free_enabled = kwargs.get("cfg_free_enabled", False)
        ctx.sampling.cfg_free_start_percent = kwargs.get("cfg_free_start_percent", 70.0)
        ctx.sampling.tome_enabled = kwargs.get("tome_enabled", False)
        ctx.sampling.tome_ratio = kwargs.get("tome_ratio", 0.5)
        ctx.sampling.tome_max_downsample = kwargs.get("tome_max_downsample", 1)
        ctx.sampling.batched_cfg = kwargs.get("batched_cfg", True)
        ctx.sampling.dynamic_cfg_rescaling = kwargs.get("dynamic_cfg_rescaling", False)
        ctx.sampling.dynamic_cfg_method = kwargs.get("dynamic_cfg_method", "variance")
        ctx.sampling.dynamic_cfg_percentile = kwargs.get("dynamic_cfg_percentile", 95.0)
        ctx.sampling.dynamic_cfg_target_scale = kwargs.get("dynamic_cfg_target_scale", 7.0)
        ctx.sampling.adaptive_noise_enabled = kwargs.get("adaptive_noise_enabled", False)
        ctx.sampling.adaptive_noise_method = kwargs.get("adaptive_noise_method", "complexity")
        
        # Map feature flags
        ctx.features.hires_fix = kwargs.get("hires_fix", False)
        ctx.features.adetailer = kwargs.get("adetailer", False)
        ctx.features.enhance_prompt = kwargs.get("enhance_prompt", False)
        ctx.features.img2img = kwargs.get("img2img", False)
        ctx.features.img2img_image = kwargs.get("img2img_image")
        ctx.features.reuse_seed = kwargs.get("reuse_seed", False)
        
        # Apply multiscale preset
        ctx.sampling.apply_multiscale_preset()
        
        # Generate seeds
        ctx._generate_seeds()
        
        return ctx
