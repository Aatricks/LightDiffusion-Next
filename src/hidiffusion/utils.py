"""Utilities for HiDiffusion patches."""
from __future__ import annotations
import contextlib
import importlib
import itertools
import logging
import math
import sys
from functools import partial
from typing import TYPE_CHECKING, Callable, NamedTuple
from enum import Enum
import torch.nn.functional as F
from src.Utilities import Latent, upscale

# Logger for HiDiffusion modules
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(name, *_): return name.lower()
        def __str__(self): return str(self.value)


UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "nearest", "area")


class TimeMode(StrEnum):
    PERCENT = "percent"
    TIMESTEP = "timestep"
    SIGMA = "sigma"


class ModelType(StrEnum):
    SD15 = "SD15"
    SDXL = "SDXL"


def parse_blocks(name: str, val) -> set[tuple[str, int]]:
    """Parse block definitions."""
    if isinstance(val, (tuple, list)):
        return {(name, item) for item in val if isinstance(item, int) and item >= 0}
    return {(name, int(v.strip())) for v in str(val).split(",") if v.strip()}


def convert_time(ms, time_mode: TimeMode, start: float, end: float) -> tuple[float, float]:
    """Convert time based on mode."""
    if time_mode == TimeMode.SIGMA:
        return start, end
    if time_mode == TimeMode.TIMESTEP:
        start, end = 1.0 - start / 999.0, 1.0 - end / 999.0
    return round(ms.percent_to_sigma(start), 4), round(ms.percent_to_sigma(end), 4)


_sigma_cache, _pct_cache = {}, {}


def get_sigma(options, key="sigmas"):
    """Get sigma value from options."""
    if not isinstance(options, dict) or (sigmas := options.get(key)) is None:
        return None
    if isinstance(sigmas, float):
        return sigmas
    cache_key = id(sigmas)
    if cache_key not in _sigma_cache:
        if len(_sigma_cache) > 4: _sigma_cache.clear()
        _sigma_cache[cache_key] = sigmas.detach().cpu().max().item()
    return _sigma_cache[cache_key]


def check_time(time_arg, start_sigma: float, end_sigma: float) -> bool:
    """Check if time is within sigma range."""
    sigma = get_sigma(time_arg) if not isinstance(time_arg, float) else time_arg
    return sigma is not None and start_sigma >= sigma >= end_sigma


_block_map = {"input": 0, "middle": 1, "output": 2}


def block_to_num(block_type: str, block_id: int) -> tuple[int, int]:
    """Convert block type to numerical representation."""
    if (tid := _block_map.get(block_type)) is None:
        raise ValueError(f"Unexpected block type {block_type}")
    return tid, block_id


def rescale_size(width: int, height: int, target_res: int, tolerance=1) -> tuple[int, int]:
    """Rescale size to fit target resolution."""
    tolerance = min(target_res, tolerance)
    scale = math.sqrt(height * width / target_res)
    hs, ws = height / scale, width / scale
    
    def neighbors(n):
        ni = int(n)
        return [ni + adj for adj in sorted(range(-min(ni-1, tolerance), tolerance+1+math.ceil(n-ni)), key=abs)]
    
    for h, w in itertools.zip_longest(neighbors(hs), neighbors(ws)):
        if w and (ha := target_res / w) % 1 == 0: return w, int(ha)
        if h and (wa := target_res / h) % 1 == 0: return int(wa), h
    raise ValueError(f"Can't rescale {width}x{height} to {target_res}")


def guess_model_type(model) -> ModelType | None:
    """Guess model type from latent format."""
    lf = model.get_model_object("latent_format")
    if lf is None:
        return None

    # 1. Try explicit type checking (most reliable)
    try:
        if isinstance(lf, Latent.SDXL) or isinstance(lf, Latent.SDXL_Playground_2_5):
            return ModelType.SDXL
        if isinstance(lf, Latent.SD15):
            return ModelType.SD15
    except Exception:
        pass

    # 2. Fallback to channel-based heuristics
    ch = getattr(lf, "latent_channels", None)
    if ch == 4:
        # Default to SD15 for 4 channels if not explicitly SDXL
        return ModelType.SD15
    if ch == 8:
        # Some SDXL implementations/VAEs use 8 channels
        return ModelType.SDXL
    
    # 3. Exclude Flux/SD3 (16 or 32 channels) from UNet-specific HiDiffusion
    return None


def sigma_to_pct(ms, sigma):
    """Convert sigma to percentage."""
    if isinstance(sigma, float):
        return (1.0 - ms.timestep(sigma) / 999.0).clamp(0.0, 1.0)
    cache_key = id(sigma)
    if cache_key not in _pct_cache:
        if len(_pct_cache) > 4: _pct_cache.clear()
        _pct_cache[cache_key] = (1.0 - ms.timestep(sigma).detach().cpu() / 999.0).clamp(0.0, 1.0).item()
    return _pct_cache[cache_key]


def fade_scale(pct, start_pct=0.0, end_pct=1.0, fade_start=1.0, fade_cap=0.0):
    """Calculate fade scale."""
    if not (start_pct <= pct <= end_pct) or start_pct > end_pct:
        return 0.0
    if pct < fade_start:
        return 1.0
    return max(fade_cap, 1.0 - (pct - fade_start) / (end_pct - fade_start))


def scale_samples(samples, width, height, mode="bicubic", sigma=None):
    """Scale samples to target size."""
    if mode == "bislerp":
        return upscale.bislerp(samples, width, height)
    return F.interpolate(samples, size=(height, width), mode=mode)


class Integrations:
    """Integration manager."""
    class Integration(NamedTuple):
        key: str
        module_name: str
        handler: Callable | None = None

    def __init__(self):
        self.initialized, self.modules, self.init_handlers, self.handlers = False, {}, [], []

    def __getitem__(self, key): return self.modules[key]
    def __contains__(self, key): return key in self.modules
    def __getattr__(self, key): return self.modules.get(key)

    @staticmethod
    def get_custom_node(name: str):
        module_key = f"custom_nodes.{name}"
        with contextlib.suppress(StopIteration):
            spec = importlib.util.find_spec(module_key)
            if spec:
                return next((v for v in sys.modules.copy().values() 
                            if hasattr(v, "__spec__") and v.__spec__ and v.__spec__.origin == spec.origin), None)
        return None

    def register_init_handler(self, h): self.init_handlers.append(h)
    
    def register_integration(self, key, module_name, handler=None):
        if self.initialized: raise ValueError("Cannot register after init")
        self.handlers.append(self.Integration(key, module_name, handler))

    def initialize(self):
        if self.initialized: return
        self.initialized = True
        for ih in self.handlers:
            if (mod := self.get_custom_node(ih.module_name)):
                mod = ih.handler(mod) if ih.handler else mod
                if mod: self.modules[ih.key] = mod
        for h in self.init_handlers: h(self)


class JHDIntegrations(Integrations):
    """JHD-specific integrations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_integration("bleh", "ComfyUI-bleh", self.bleh_integration)
        self.register_integration("freeu_advanced", "FreeU_Advanced")

    @classmethod
    def bleh_integration(cls, bleh):
        return bleh if getattr(bleh, "BLEH_VERSION", -1) >= 0 else None


MODULES = JHDIntegrations()


class IntegratedNode(type):
    """Metaclass for integrated nodes."""
    @staticmethod
    def wrap_INPUT_TYPES(orig, *args, **kwargs):
        MODULES.initialize()
        return orig(*args, **kwargs)

    def __new__(cls, name, bases, attrs):
        obj = type.__new__(cls, name, bases, attrs)
        if hasattr(obj, "INPUT_TYPES"):
            obj.INPUT_TYPES = partial(cls.wrap_INPUT_TYPES, obj.INPUT_TYPES)
        return obj


def init_integrations(integrations):
    """Initialize integrations."""
    global scale_samples, UPSCALE_METHODS
    if (bleh := integrations.bleh) and (lu := getattr(bleh.py, "latent_utils", None)):
        UPSCALE_METHODS = lu.UPSCALE_METHODS
        if getattr(bleh, "BLEH_VERSION", -1) >= 0:
            scale_samples = lu.scale_samples
        else:
            scale_samples = lambda *a, sigma=None, **k: lu.scale_samples(*a, **k)


MODULES.register_init_handler(init_integrations)

__all__ = ("UPSCALE_METHODS", "check_time", "convert_time", "get_sigma", "guess_model_type",
           "logger", "parse_blocks", "rescale_size", "scale_samples")
