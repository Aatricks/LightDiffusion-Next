from __future__ import annotations

import contextlib
import importlib
import itertools
import logging
import math
import sys
from functools import partial
from typing import TYPE_CHECKING, Callable, NamedTuple
from src.Utilities import Latent, upscale

import torch.nn.functional as torchf

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

try:
    from enum import StrEnum
except ImportError:
    # Compatibility workaround for pre-3.11 Python versions.
    from enum import Enum

    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(name: str, *_unused: list) -> str:
            return name.lower()

        def __str__(self) -> str:
            return str(self.value)


logger = logging.getLogger(__name__)

UPSCALE_METHODS = ("bicubic", "bislerp", "bilinear", "nearest-exact", "nearest", "area")


class TimeMode(StrEnum):
    PERCENT = "percent"
    TIMESTEP = "timestep"
    SIGMA = "sigma"


class ModelType(StrEnum):
    SD15 = "SD15"
    SDXL = "SDXL"


def parse_blocks(name: str, val: str | Sequence[int]) -> set[tuple[str, int]]:
    """#### Parse block definitions.

    #### Args:
        - `name` (str): The name of the block.
        - `val` (Union[str, Sequence[int]]): The block values.

    #### Returns:
        - `set[tuple[str, int]]`: The parsed blocks.
    """
    if isinstance(val, (tuple, list)):
        # Handle a sequence passed in via YAML parameters.
        if not all(isinstance(item, int) and item >= 0 for item in val):
            raise ValueError(
                "Bad blocks definition, must be comma separated string or sequence of positive int",
            )
        return {(name, item) for item in val}
    vals = (rawval.strip() for rawval in val.split(","))
    return {(name, int(val.strip())) for val in vals if val}


def convert_time(
    ms: object,
    time_mode: TimeMode,
    start_time: float,
    end_time: float,
) -> tuple[float, float]:
    """#### Convert time based on the mode.

    #### Args:
        - `ms` (Any): The time object.
        - `time_mode` (TimeMode): The time mode.
        - `start_time` (float): The start time.
        - `end_time` (float): The end time.

    #### Returns:
        - `Tuple[float, float]`: The converted start and end times.
    """
    if time_mode == TimeMode.SIGMA:
        return (start_time, end_time)
    if time_mode == TimeMode.TIMESTEP:
        start_time = 1.0 - (start_time / 999.0)
        end_time = 1.0 - (end_time / 999.0)
    else:
        if start_time > 1.0 or start_time < 0.0:
            raise ValueError(
                "invalid value for start percent",
            )
        if end_time > 1.0 or end_time < 0.0:
            raise ValueError(
                "invalid value for end percent",
            )
    return (
        round(ms.percent_to_sigma(start_time), 4),
        round(ms.percent_to_sigma(end_time), 4),
    )
    raise ValueError("invalid time mode")


def get_sigma(options: dict, key: str = "sigmas") -> float | None:
    """#### Get the sigma value from options.

    #### Args:
        - `options` (dict): The options dictionary.
        - `key` (str, optional): The key to look for. Defaults to "sigmas".

    #### Returns:
        - `Optional[float]`: The sigma value if found, otherwise None.
    """
    if not isinstance(options, dict):
        return None
    sigmas = options.get(key)
    if sigmas is None:
        return None
    if isinstance(sigmas, float):
        return sigmas
    return sigmas.detach().cpu().max().item()


def check_time(time_arg: dict | float, start_sigma: float, end_sigma: float) -> bool:
    """#### Check if the time is within the sigma range.

    #### Args:
        - `time_arg` (Union[dict, float]): The time argument.
        - `start_sigma` (float): The start sigma.
        - `end_sigma` (float): The end sigma.

    #### Returns:
        - `bool`: Whether the time is within the range.
    """
    sigma = get_sigma(time_arg) if not isinstance(time_arg, float) else time_arg
    if sigma is None:
        return False
    return sigma <= start_sigma and sigma >= end_sigma


__block_to_num_map = {"input": 0, "middle": 1, "output": 2}


def block_to_num(block_type: str, block_id: int) -> tuple[int, int]:
    """#### Convert block type and id to numerical representation.

    #### Args:
        - `block_type` (str): The block type.
        - `block_id` (int): The block id.

    #### Returns:
        - `Tuple[int, int]`: The numerical representation of the block.
    """
    type_id = __block_to_num_map.get(block_type)
    if type_id is None:
        errstr = f"Got unexpected block type {block_type}!"
        raise ValueError(errstr)
    return (type_id, block_id)


# Naive and totally inaccurate way to factorize target_res into rescaled integer width/height
def rescale_size(
    width: int,
    height: int,
    target_res: int,
    *,
    tolerance=1,
) -> tuple[int, int]:
    """#### Rescale size to fit target resolution.

    #### Args:
        - `width` (int): The width.
        - `height` (int): The height.
        - `target_res` (int): The target resolution.
        - `tolerance` (int, optional): The tolerance. Defaults to 1.

    #### Returns:
        - `Tuple[int, int]`: The rescaled width and height.
    """
    tolerance = min(target_res, tolerance)

    def get_neighbors(num: float):
        if num < 1:
            return None
        numi = int(num)
        return tuple(
            numi + adj
            for adj in sorted(
                range(
                    -min(numi - 1, tolerance),
                    tolerance + 1 + math.ceil(num - numi),
                ),
                key=abs,
            )
        )

    scale = math.sqrt(height * width / target_res)
    height_scaled, width_scaled = height / scale, width / scale
    height_rounded = get_neighbors(height_scaled)
    width_rounded = get_neighbors(width_scaled)
    for h, w in itertools.zip_longest(height_rounded, width_rounded):
        h_adj = target_res / w if w is not None else 0.1
        if h_adj % 1 == 0:
            return (w, int(h_adj))
        if h is None:
            continue
        w_adj = target_res / h
        if w_adj % 1 == 0:
            return (int(w_adj), h)
    msg = f"Can't rescale {width} and {height} to fit {target_res}"
    raise ValueError(msg)


def guess_model_type(model: object) -> ModelType | None:
    """#### Guess the model type.

    #### Args:
        - `model` (object): The model object.

    #### Returns:
        - `Optional[ModelType]`: The guessed model type.
    """
    latent_format = model.get_model_object("latent_format")
    if isinstance(latent_format, Latent.SD15):
        return ModelType.SD15
    return None


def sigma_to_pct(ms, sigma):
    """#### Convert sigma to percentage.

    #### Args:
        - `ms` (Any): The time object.
        - `sigma` (float): The sigma value.

    #### Returns:
        - `float`: The percentage.
    """
    return (1.0 - (ms.timestep(sigma).detach().cpu() / 999.0)).clamp(0.0, 1.0).item()


def fade_scale(
    pct,
    start_pct=0.0,
    end_pct=1.0,
    fade_start=1.0,
    fade_cap=0.0,
):
    """#### Calculate the fade scale.

    #### Args:
        - `pct` (float): The percentage.
        - `start_pct` (float, optional): The start percentage. Defaults to 0.0.
        - `end_pct` (float, optional): The end percentage. Defaults to 1.0.
        - `fade_start` (float, optional): The fade start. Defaults to 1.0.
        - `fade_cap` (float, optional): The fade cap. Defaults to 0.0.

    #### Returns:
        - `float`: The fade scale.
    """
    if not (start_pct <= pct <= end_pct) or start_pct > end_pct:
        return 0.0
    if pct < fade_start:
        return 1.0
    scaling_pct = 1.0 - ((pct - fade_start) / (end_pct - fade_start))
    return max(fade_cap, scaling_pct)


def scale_samples(
    samples,
    width,
    height,
    mode="bicubic",
    sigma=None,  # noqa: ARG001
):
    """#### Scale samples to the specified width and height.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `width` (int): The target width.
        - `height` (int): The target height.
        - `mode` (str, optional): The scaling mode. Defaults to "bicubic".
        - `sigma` (Optional[float], optional): The sigma value. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The scaled samples.
    """
    if mode == "bislerp":
        return upscale.bislerp(samples, width, height)
    return torchf.interpolate(samples, size=(height, width), mode=mode)


class Integrations:
    """#### Class for managing integrations."""
    class Integration(NamedTuple):
        key: str
        module_name: str
        handler: Callable | None = None

    def __init__(self):
        """#### Initialize the Integrations class."""
        self.initialized = False
        self.modules = {}
        self.init_handlers = []
        self.handlers = []

    def __getitem__(self, key):
        """#### Get a module by key.

        #### Args:
            - `key` (str): The key.

        #### Returns:
            - `ModuleType`: The module.
        """
        return self.modules[key]

    def __contains__(self, key):
        """#### Check if a module is in the integrations.

        #### Args:
            - `key` (str): The key.

        #### Returns:
            - `bool`: Whether the module is in the integrations.
        """
        return key in self.modules

    def __getattr__(self, key):
        """#### Get a module by attribute.

        #### Args:
            - `key` (str): The key.

        #### Returns:
            - `Optional[ModuleType]`: The module if found, otherwise None.
        """
        return self.modules.get(key)

    @staticmethod
    def get_custom_node(name: str) -> ModuleType | None:
        """#### Get a custom node by name.

        #### Args:
            - `name` (str): The name of the custom node.

        #### Returns:
            - `Optional[ModuleType]`: The custom node if found, otherwise None.
        """
        module_key = f"custom_nodes.{name}"
        with contextlib.suppress(StopIteration):
            spec = importlib.util.find_spec(module_key)
            if spec is None:
                return None
            return next(
                v
                for v in sys.modules.copy().values()
                if hasattr(v, "__spec__")
                and v.__spec__ is not None
                and v.__spec__.origin == spec.origin
            )
        return None

    def register_init_handler(self, handler):
        """#### Register an initialization handler.

        #### Args:
            - `handler` (Callable): The handler.
        """
        self.init_handlers.append(handler)

    def register_integration(self, key: str, module_name: str, handler=None) -> None:
        """#### Register an integration.

        #### Args:
            - `key` (str): The key.
            - `module_name` (str): The module name.
            - `handler` (Optional[Callable], optional): The handler. Defaults to None.
        """
        if self.initialized:
            raise ValueError(
                "Internal error: Cannot register integration after initialization",
            )
        if any(item[0] == key or item[1] == module_name for item in self.handlers):
            errstr = (
                f"Module {module_name} ({key}) already in integration handlers list!"
            )
            raise ValueError(errstr)
        self.handlers.append(self.Integration(key, module_name, handler))

    def initialize(self) -> None:
        """#### Initialize the integrations."""
        if self.initialized:
            return
        self.initialized = True
        for ih in self.handlers:
            module = self.get_custom_node(ih.module_name)
            if module is None:
                continue
            if ih.handler is not None:
                module = ih.handler(module)
            if module is not None:
                self.modules[ih.key] = module

        for init_handler in self.init_handlers:
            init_handler(self)


class JHDIntegrations(Integrations):
    """#### Class for managing JHD integrations."""
    def __init__(self, *args: list, **kwargs: dict):
        """#### Initialize the JHDIntegrations class."""
        super().__init__(*args, **kwargs)
        self.register_integration("bleh", "ComfyUI-bleh", self.bleh_integration)
        self.register_integration("freeu_advanced", "FreeU_Advanced")

    @classmethod
    def bleh_integration(cls, bleh: ModuleType) -> ModuleType | None:
        """#### Integrate with BLEH.

        #### Args:
            - `bleh` (ModuleType): The BLEH module.

        #### Returns:
            - `Optional[ModuleType]`: The integrated BLEH module if successful, otherwise None.
        """
        bleh_version = getattr(bleh, "BLEH_VERSION", -1)
        if bleh_version < 0:
            return None
        return bleh


MODULES = JHDIntegrations()


class IntegratedNode(type):
    """#### Metaclass for integrated nodes."""
    @staticmethod
    def wrap_INPUT_TYPES(orig_method: Callable, *args: list, **kwargs: dict) -> dict:
        """#### Wrap the INPUT_TYPES method to initialize modules.

        #### Args:
            - `orig_method` (Callable): The original method.
            - `args` (list): The arguments.
            - `kwargs` (dict): The keyword arguments.

        #### Returns:
            - `dict`: The result of the original method.
        """
        MODULES.initialize()
        return orig_method(*args, **kwargs)

    def __new__(cls: type, name: str, bases: tuple, attrs: dict) -> object:
        """#### Create a new instance of the class.

        #### Args:
            - `name` (str): The name of the class.
            - `bases` (tuple): The base classes.
            - `attrs` (dict): The attributes.

        #### Returns:
            - `object`: The new instance.
        """
        obj = type.__new__(cls, name, bases, attrs)
        if hasattr(obj, "INPUT_TYPES"):
            obj.INPUT_TYPES = partial(cls.wrap_INPUT_TYPES, obj.INPUT_TYPES)
        return obj


def init_integrations(integrations) -> None:
    """#### Initialize integrations.

    #### Args:
        - `integrations` (Integrations): The integrations object.
    """
    global scale_samples, UPSCALE_METHODS  # noqa: PLW0603
    ext_bleh = integrations.bleh
    if ext_bleh is None:
        return
    bleh_latentutils = getattr(ext_bleh.py, "latent_utils", None)
    if bleh_latentutils is None:
        return
    bleh_version = getattr(ext_bleh, "BLEH_VERSION", -1)
    UPSCALE_METHODS = bleh_latentutils.UPSCALE_METHODS
    if bleh_version >= 0:
        scale_samples = bleh_latentutils.scale_samples
        return

    def scale_samples_wrapped(*args: list, sigma=None, **kwargs: dict):  # noqa: ARG001
        """#### Wrap the scale_samples method.

        #### Args:
            - `args` (list): The arguments.
            - `sigma` (Optional[float], optional): The sigma value. Defaults to None.
            - `kwargs` (dict): The keyword arguments.

        #### Returns:
            - `Any`: The result of the scale_samples method.
        """
        return bleh_latentutils.scale_samples(*args, **kwargs)

    scale_samples = scale_samples_wrapped


MODULES.register_init_handler(init_integrations)

__all__ = (
    "UPSCALE_METHODS",
    "check_time",
    "convert_time",
    "get_sigma",
    "guess_model_type",
    "parse_blocks",
    "rescale_size",
    "scale_samples",
)
