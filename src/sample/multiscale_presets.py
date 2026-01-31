"""Multi-scale diffusion presets for quality/performance optimization."""
from typing import NamedTuple, Dict, Any


class MultiscalePreset(NamedTuple):
    """Multi-scale diffusion preset configuration."""
    name: str
    description: str
    enable_multiscale: bool
    multiscale_factor: float
    multiscale_fullres_start: int
    multiscale_fullres_end: int
    multiscale_intermittent_fullres: bool

    @property
    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ["enable_multiscale", "multiscale_factor", 
                "multiscale_fullres_start", "multiscale_fullres_end", "multiscale_intermittent_fullres"]}


MULTISCALE_PRESETS = {
    "quality": MultiscalePreset("Quality", "High quality with intermittent full-res", True, 0.5, 10, 8, True),
    "performance": MultiscalePreset("Performance", "Aggressive downscaling for max speed", True, 0.25, 5, 8, True),
    "balanced": MultiscalePreset("Balanced", "Good quality and performance", True, 0.5, 5, 8, True),
    "disabled": MultiscalePreset("Disabled", "Full resolution throughout", False, 1.0, 0, 0, False),
}


def get_preset(preset_name: str) -> MultiscalePreset:
    """Get preset by name."""
    if preset_name not in MULTISCALE_PRESETS:
        raise KeyError(f"Preset '{preset_name}' not found. Available: {', '.join(MULTISCALE_PRESETS.keys())}")
    return MULTISCALE_PRESETS[preset_name]


def get_preset_parameters(preset_name: str) -> Dict[str, Any]:
    """Get preset parameters as dict."""
    return get_preset(preset_name).as_dict


def list_presets() -> Dict[str, str]:
    """List preset names and descriptions."""
    return {name: p.description for name, p in MULTISCALE_PRESETS.items()}


def apply_preset_to_kwargs(preset_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Apply preset to kwargs dict."""
    kwargs.update(get_preset_parameters(preset_name))
    return kwargs
