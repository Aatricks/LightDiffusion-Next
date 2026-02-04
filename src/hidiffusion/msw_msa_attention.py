from __future__ import annotations
import itertools
import math
from time import time
from typing import Any, NamedTuple
from src.Model import ModelPatcher
import torch
from . import utils
from .utils import IntegratedNode, ModelType, StrEnum, TimeMode, block_to_num, check_time, convert_time, get_sigma, guess_model_type, logger, parse_blocks, rescale_size, scale_samples

F = torch.nn.functional
SCALE_METHODS, REVERSE_SCALE_METHODS = (), ()
DEFAULT_WARN_INTERVAL = 60


def init_integrations(_integrations) -> None:
    global scale_samples, SCALE_METHODS, REVERSE_SCALE_METHODS
    SCALE_METHODS = ("disabled", "skip", *utils.UPSCALE_METHODS)
    REVERSE_SCALE_METHODS = utils.UPSCALE_METHODS
    scale_samples = utils.scale_samples


utils.MODULES.register_init_handler(init_integrations)


class Preset(NamedTuple):
    input_blocks: str = ""
    middle_blocks: str = ""
    output_blocks: str = ""
    time_mode: TimeMode = TimeMode.PERCENT
    start_time: float = 0.2
    end_time: float = 1.0
    scale_mode: str = "nearest-exact"
    reverse_scale_mode: str = "nearest-exact"

    @property
    def as_dict(self):
        return {k: getattr(self, k) for k in self._fields}

    @property
    def pretty_blocks(self):
        return " / ".join(b or "none" for b in (self.input_blocks, self.middle_blocks, self.output_blocks))


SIMPLE_PRESETS = {ModelType.SD15: Preset(input_blocks="1,2", output_blocks="11,10,9"),
                  ModelType.SDXL: Preset(input_blocks="4,5", output_blocks="3,4,5")}


class WindowSize(NamedTuple):
    height: int
    width: int
    @property
    def sum(self):
        return self.height * self.width
    def __neg__(self):
        return self.__class__(-self.height, -self.width)


class ShiftSize(WindowSize):
    pass


class LastShiftMode(StrEnum):
    GLOBAL, BLOCK, BOTH, IGNORE = "global", "block", "both", "ignore"


class LastShiftStrategy(StrEnum):
    INCREMENT, DECREMENT, RETRY = "increment", "decrement", "retry"


class Config(NamedTuple):
    start_sigma: float
    end_sigma: float
    use_blocks: set
    scale_mode: str = "nearest-exact"
    reverse_scale_mode: str = "nearest-exact"
    silent: bool = False
    last_shift_mode: LastShiftMode = LastShiftMode.GLOBAL
    last_shift_strategy: LastShiftStrategy = LastShiftStrategy.INCREMENT
    pre_window_multiplier: float = 1.0
    post_window_multiplier: float = 1.0
    pre_window_reverse_multiplier: float = 1.0
    post_window_reverse_multiplier: float = 1.0
    force_apply_attn2: bool = False
    rescale_search_tolerance: int = 1
    verbose: int = 0

    @classmethod
    def build(cls, *, ms, input_blocks, middle_blocks, output_blocks, time_mode, start_time, end_time, **kwargs):
        time_mode = TimeMode(time_mode)
        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)
        blocks = itertools.starmap(parse_blocks, (("input", input_blocks), ("middle", middle_blocks), ("output", output_blocks)))
        return cls.__new__(cls, start_sigma=start_sigma, end_sigma=end_sigma, use_blocks=set().union(*blocks), **kwargs)

    @staticmethod
    def maybe_multiply(t: torch.Tensor, multiplier: float = 1.0, post: bool = False) -> torch.Tensor:
        return t if multiplier == 1.0 else (t.mul_(multiplier) if post else t * multiplier)


class State:
    __slots__ = ("config", "last_block", "last_shift", "last_shifts", "last_sigma", "last_warned", "window_args")

    def __init__(self, config):
        self.config, self.last_warned = config, None
        self.reset()

    def reset(self):
        self.window_args = self.last_sigma = self.last_block = self.last_shift = None
        self.last_shifts = {}

    @property
    def pretty_last_block(self) -> str:
        if self.last_block is None:
            return "unknown"
        bt, bnum = self.last_block
        return f"{'attn2.' if self.config.force_apply_attn2 else ''}{('in', 'mid', 'out')[bt]}.{bnum}"

    def maybe_warning(self, s):
        if self.config.silent:
            return
        now = time()
        if self.config.verbose >= 2 or self.last_warned is None or now - self.last_warned >= DEFAULT_WARN_INTERVAL:
            logger.warning(f"** jankhidiffusion: MSW-MSA attention({self.pretty_last_block}): {s}")
            self.last_warned = now


class ApplyMSWMSAAttention(metaclass=IntegratedNode):
    RETURN_TYPES, OUTPUT_TOOLTIPS = ("MODEL",), ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION, CATEGORY = "patch", "model_patches/unet"
    DESCRIPTION = "Applies MSW-MSA attention patch. Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_blocks": ("STRING", {"default": "1,2"}),
                "middle_blocks": ("STRING", {"default": ""}),
                "output_blocks": ("STRING", {"default": "9,10,11"}),
                "time_mode": (tuple(str(val) for val in TimeMode), {"default": "percent"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "round": False, "step": 0.01}),
                "end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 999.0, "round": False, "step": 0.01}),
                "model": ("MODEL",),
            },
            "optional": {"yaml_parameters": ("STRING", {"dynamicPrompts": False, "multiline": True, "defaultInput": True})},
        }

    @staticmethod
    def window_partition(x: torch.Tensor, state: State, window_index: int) -> torch.Tensor:
        config = state.config
        x = config.maybe_multiply(x, config.pre_window_multiplier)
        window_size, shift_size, height, width = state.window_args[window_index]
        do_rescale = (height % 2 + width % 2) != 0
        if do_rescale:
            if config.scale_mode == "skip":
                state.maybe_warning("Incompatible latent size - skipping MSW-MSA attention.")
                return x
            if config.scale_mode == "disabled":
                state.maybe_warning("Incompatible latent size - trying to proceed anyway.")
                do_rescale = False
            else:
                state.maybe_warning("Incompatible latent size - applying scaling workaround.")
        batch, _features, channels = x.shape
        wheight, wwidth = window_size
        x = x.view(batch, height, width, channels)
        if do_rescale:
            x = scale_samples(x.permute(0, 3, 1, 2).contiguous(), wwidth * 2, wheight * 2, mode=config.scale_mode, sigma=state.last_sigma).permute(0, 2, 3, 1).contiguous()
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=-shift_size, dims=(1, 2))
        x = x.view(batch, 2, wheight, 2, wwidth, channels)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size.height, window_size.width, channels)
        return config.maybe_multiply(windows.view(-1, window_size.sum, channels), config.post_window_multiplier)

    @staticmethod
    def window_reverse(windows: torch.Tensor, state: State, window_index: int = 0) -> torch.Tensor:
        config = state.config
        windows = config.maybe_multiply(windows, config.pre_window_reverse_multiplier)
        window_size, shift_size, height, width = state.window_args[window_index]
        do_rescale = (height % 2 + width % 2) != 0
        if do_rescale and config.scale_mode == "skip":
            return windows
        if do_rescale and config.scale_mode == "disabled":
            do_rescale = False
        batch, _features, channels = windows.shape
        wheight, wwidth = window_size
        windows = windows.view(-1, wheight, wwidth, channels)
        batch = int(windows.shape[0] / 4)
        x = windows.view(batch, 2, 2, wheight, wwidth, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, wheight * 2, wwidth * 2, -1)
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))
        if do_rescale:
            x = scale_samples(x.permute(0, 3, 1, 2).contiguous(), width, height, mode=config.reverse_scale_mode, sigma=state.last_sigma).permute(0, 2, 3, 1).contiguous()
        return config.maybe_multiply(x.view(batch, height * width, channels), config.post_window_reverse_multiplier)

    @staticmethod
    def get_window_args(config: Config, n: torch.Tensor, orig_shape: tuple, shift: int):
        _batch, features, _channels = n.shape
        width, height = rescale_size(orig_shape[-1], orig_shape[-2], features, tolerance=config.rescale_search_tolerance)
        wheight, wwidth = math.ceil(height / 2), math.ceil(width / 2)
        shifts = [(0, 0), (wheight // 4, wwidth // 4), (wheight // 4 * 2, wwidth // 4 * 2), (wheight // 4 * 3, wwidth // 4 * 3)]
        return WindowSize(wheight, wwidth), ShiftSize(*shifts[shift]), height, width

    @staticmethod
    def get_shift(curr_block: tuple, state: State, *, shift_count=4) -> int:
        mode, strat = state.config.last_shift_mode, state.config.last_shift_strategy
        shift = int(torch.rand(1, device="cpu").item() * shift_count)
        avoid = {state.last_shifts.get(curr_block), state.last_shift} if mode == LastShiftMode.BOTH else \
                {state.last_shifts.get(curr_block)} if mode == LastShiftMode.BLOCK else \
                {state.last_shift} if mode == LastShiftMode.GLOBAL else set()
        while shift in avoid:
            if strat == LastShiftStrategy.DECREMENT:
                shift = (shift - 1) % shift_count
            elif strat == LastShiftStrategy.RETRY:
                shift = int(torch.rand(1, device="cpu").item() * shift_count)
            else:
                shift = (shift + 1) % shift_count
        return shift

    @classmethod
    def patch(cls, *, model: ModelPatcher.ModelPatcher, yaml_parameters: str | None = None, **kwargs) -> tuple:
        if yaml_parameters:
            import yaml
            extra = yaml.safe_load(yaml_parameters)
            if isinstance(extra, dict):
                kwargs |= extra
        config = Config.build(ms=model.get_model_object("model_sampling"), **kwargs)
        if not config.use_blocks:
            return (model,)
        if config.verbose:
            logger.info(f"** jankhidiffusion: MSW-MSA Attention: Using config: {config}")

        model, state = model.clone(), State(config)

        def attn_patch(q, k, v, extra_options):
            state.window_args = None
            sigma, block = get_sigma(extra_options), extra_options.get("block", ("missing", 0))
            curr_block = block_to_num(*block)
            if state.last_sigma is not None and sigma > state.last_sigma:
                state.reset()
            state.last_block, state.last_sigma = curr_block, sigma
            if block not in config.use_blocks or not check_time(sigma, config.start_sigma, config.end_sigma):
                return q, k, v
            shift = cls.get_shift(curr_block, state)
            state.last_shifts[curr_block] = state.last_shift = shift
            try:
                state.window_args = tuple(cls.get_window_args(config, x, extra_options["original_shape"], shift) if x is not None else None for x in (q, k, v))
                attn_parts = (q,) if q is not None and q is k and q is v else (q, k, v)
                result = tuple(cls.window_partition(t, state, i) if t is not None else None for i, t in enumerate(attn_parts))
            except (RuntimeError, ValueError) as exc:
                logger.warning(f"** jankhidiffusion: Exception applying MSW-MSA attention: {exc}")
                state.window_args = None
                return q, k, v
            return result * 3 if len(result) == 1 else result

        def attn_output_patch(n, extra_options):
            if state.window_args is None or state.last_block != block_to_num(*extra_options.get("block", ("missing", 0))):
                state.window_args = None
                return n
            result = cls.window_reverse(n, state)
            state.window_args = None
            return result

        if not config.force_apply_attn2:
            model.set_model_attn1_patch(attn_patch)
            model.set_model_attn1_output_patch(attn_output_patch)
        else:
            model.set_model_attn2_patch(attn_patch)
            model.set_model_attn2_output_patch(attn_output_patch)
        return (model,)


class ApplyMSWMSAAttentionSimple(metaclass=IntegratedNode):
    RETURN_TYPES, OUTPUT_TOOLTIPS = ("MODEL",), ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION, CATEGORY = "go", "model_patches/unet"
    DESCRIPTION = "Simplified MSW-MSA Attention. Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_type": (("auto", "SD15", "SDXL"),), "model": ("MODEL",)}}

    @classmethod
    def go(cls, model_type, model):
        if model_type == "auto":
            model_type = guess_model_type(model)
            if model_type not in SIMPLE_PRESETS:
                raise RuntimeError("Unable to guess model type")
        else:
            model_type = ModelType(model_type)
        preset = SIMPLE_PRESETS.get(model_type)
        if preset is None:
            raise ValueError(f"Unknown model type {model_type!s}")
        logger.info(f"** ApplyMSWMSAAttentionSimple: Using preset {model_type!s}: [{preset.pretty_blocks}], {preset.start_time:.2}/{preset.end_time:.2}")
        return ApplyMSWMSAAttention.patch(model=model, **preset.as_dict)


__all__ = ("ApplyMSWMSAAttention", "ApplyMSWMSAAttentionSimple")
