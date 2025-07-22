"""
Multi-scale diffusion presets for quality and performance optimization.

This module provides predefined configurations for multi-scale diffusion that balance
quality and performance based on different use cases.
"""

from typing import NamedTuple, Dict, Any


class MultiscalePreset(NamedTuple):
    """#### Class representing a multi-scale diffusion preset.

    #### Args:
        - `name` (str): The name of the preset.
        - `description` (str): Description of the preset's purpose.
        - `enable_multiscale` (bool): Whether multi-scale diffusion is enabled.
        - `multiscale_factor` (float): Scale factor for intermediate steps (0.1-1.0).
        - `multiscale_fullres_start` (int): Number of first steps at full resolution.
        - `multiscale_fullres_end` (int): Number of last steps at full resolution.
        - `multiscale_intermittent_fullres` (bool): Whether to use intermittent full-res.
    """

    name: str
    description: str
    enable_multiscale: bool
    multiscale_factor: float
    multiscale_fullres_start: int
    multiscale_fullres_end: int
    multiscale_intermittent_fullres: bool

    @property
    def as_dict(self) -> Dict[str, Any]:
        """#### Convert the preset to a dictionary.

        #### Returns:
            - `Dict[str, Any]`: The preset parameters as a dictionary.
        """
        return {
            "enable_multiscale": self.enable_multiscale,
            "multiscale_factor": self.multiscale_factor,
            "multiscale_fullres_start": self.multiscale_fullres_start,
            "multiscale_fullres_end": self.multiscale_fullres_end,
            "multiscale_intermittent_fullres": self.multiscale_intermittent_fullres,
        }


# Predefined multi-scale diffusion presets
MULTISCALE_PRESETS = {
    "quality": MultiscalePreset(
        name="Quality",
        description="High quality preset with intermittent full-res for best image quality",
        enable_multiscale=True,
        multiscale_factor=0.5,
        multiscale_fullres_start=10,
        multiscale_fullres_end=8,
        multiscale_intermittent_fullres=True,
    ),
    "performance": MultiscalePreset(
        name="Performance",
        description="Performance-oriented preset with aggressive downscaling for maximum speed",
        enable_multiscale=True,
        multiscale_factor=0.25,
        multiscale_fullres_start=5,
        multiscale_fullres_end=8,
        multiscale_intermittent_fullres=True,
    ),
    "balanced": MultiscalePreset(
        name="Balanced",
        description="Balanced preset offering good quality and performance",
        enable_multiscale=True,
        multiscale_factor=0.5,
        multiscale_fullres_start=5,
        multiscale_fullres_end=8,
        multiscale_intermittent_fullres=True,
    ),
    "disabled": MultiscalePreset(
        name="Disabled",
        description="Multi-scale diffusion disabled - full resolution throughout",
        enable_multiscale=False,
        multiscale_factor=1.0,
        multiscale_fullres_start=0,
        multiscale_fullres_end=0,
        multiscale_intermittent_fullres=False,
    ),
}


def get_preset(preset_name: str) -> MultiscalePreset:
    """#### Get a multi-scale diffusion preset by name.

    #### Args:
        - `preset_name` (str): The name of the preset to retrieve.

    #### Returns:
        - `MultiscalePreset`: The requested preset.

    #### Raises:
        - `KeyError`: If the preset name is not found.
    """
    if preset_name not in MULTISCALE_PRESETS:
        available_presets = ", ".join(MULTISCALE_PRESETS.keys())
        raise KeyError(
            f"Preset '{preset_name}' not found. Available presets: {available_presets}"
        )

    return MULTISCALE_PRESETS[preset_name]


def get_preset_parameters(preset_name: str) -> Dict[str, Any]:
    """#### Get multi-scale diffusion parameters for a preset.

    #### Args:
        - `preset_name` (str): The name of the preset.

    #### Returns:
        - `Dict[str, Any]`: The preset parameters.
    """
    return get_preset(preset_name).as_dict


def list_presets() -> Dict[str, str]:
    """#### List all available multi-scale diffusion presets.

    #### Returns:
        - `Dict[str, str]`: Dictionary mapping preset names to descriptions.
    """
    return {name: preset.description for name, preset in MULTISCALE_PRESETS.items()}


def apply_preset_to_kwargs(preset_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """#### Apply a multi-scale preset to keyword arguments.

    #### Args:
        - `preset_name` (str): The name of the preset to apply.
        - `kwargs` (Dict[str, Any]): Existing keyword arguments.

    #### Returns:
        - `Dict[str, Any]`: Updated keyword arguments with preset parameters.
    """
    preset_params = get_preset_parameters(preset_name)
    kwargs.update(preset_params)
    return kwargs
