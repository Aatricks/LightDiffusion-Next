from __future__ import annotations

import itertools
import math
from time import time
from typing import Any, NamedTuple
from src.Model import ModelPatcher

import torch

from . import utils
from .utils import (
    IntegratedNode,
    ModelType,
    StrEnum,
    TimeMode,
    block_to_num,
    check_time,
    convert_time,
    get_sigma,
    guess_model_type,
    logger,
    parse_blocks,
    rescale_size,
    scale_samples,
)

F = torch.nn.functional

SCALE_METHODS = ()
REVERSE_SCALE_METHODS = ()


# Taken from https://github.com/blepping/comfyui_jankhidiffusion


def init_integrations(_integrations) -> None:
    """#### Initialize integrations.

    #### Args:
        - `_integrations` (Any): The integrations object.
    """
    global scale_samples, SCALE_METHODS, REVERSE_SCALE_METHODS  # noqa: PLW0603
    SCALE_METHODS = ("disabled", "skip", *utils.UPSCALE_METHODS)
    REVERSE_SCALE_METHODS = utils.UPSCALE_METHODS
    scale_samples = utils.scale_samples


utils.MODULES.register_init_handler(init_integrations)

DEFAULT_WARN_INTERVAL = 60


class Preset(NamedTuple):
    """#### Class representing a preset configuration.

    #### Args:
        - `input_blocks` (str): The input blocks.
        - `middle_blocks` (str): The middle blocks.
        - `output_blocks` (str): The output blocks.
        - `time_mode` (TimeMode): The time mode.
        - `start_time` (float): The start time.
        - `end_time` (float): The end time.
        - `scale_mode` (str): The scale mode.
        - `reverse_scale_mode` (str): The reverse scale mode.
    """
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
        """#### Convert the preset to a dictionary.

        #### Returns:
            - `Dict[str, Any]`: The preset as a dictionary.
        """
        return {k: getattr(self, k) for k in self._fields}

    @property
    def pretty_blocks(self):
        """#### Get a pretty string representation of the blocks.

        #### Returns:
            - `str`: The pretty string representation of the blocks.
        """
        blocks = (self.input_blocks, self.middle_blocks, self.output_blocks)
        return " / ".join(b or "none" for b in blocks)


SIMPLE_PRESETS = {
    ModelType.SD15: Preset(input_blocks="1,2", output_blocks="11,10,9"),
    ModelType.SDXL: Preset(input_blocks="4,5", output_blocks="3,4,5"),
}


class WindowSize(NamedTuple):
    """#### Class representing the window size.

    #### Args:
        - `height` (int): The height of the window.
        - `width` (int): The width of the window.
    """
    height: int
    width: int

    @property
    def sum(self):
        """#### Get the sum of the height and width.

        #### Returns:
            - `int`: The sum of the height and width.
        """
        return self.height * self.width

    def __neg__(self):
        """#### Negate the window size.

        #### Returns:
            - `WindowSize`: The negated window size.
        """
        return self.__class__(-self.height, -self.width)


class ShiftSize(WindowSize):
    """#### Class representing the shift size."""
    pass


class LastShiftMode(StrEnum):
    """#### Enum for the last shift mode."""
    GLOBAL = "global"
    BLOCK = "block"
    BOTH = "both"
    IGNORE = "ignore"


class LastShiftStrategy(StrEnum):
    """#### Enum for the last shift strategy."""
    INCREMENT = "increment"
    DECREMENT = "decrement"
    RETRY = "retry"


class Config(NamedTuple):
    """#### Class representing the configuration.

    #### Args:
        - `start_sigma` (float): The start sigma.
        - `end_sigma` (float): The end sigma.
        - `use_blocks` (set): The blocks to use.
        - `scale_mode` (str): The scale mode.
        - `reverse_scale_mode` (str): The reverse scale mode.
        - `silent` (bool): Whether to disable log warnings.
        - `last_shift_mode` (LastShiftMode): The last shift mode.
        - `last_shift_strategy` (LastShiftStrategy): The last shift strategy.
        - `pre_window_multiplier` (float): The pre-window multiplier.
        - `post_window_multiplier` (float): The post-window multiplier.
        - `pre_window_reverse_multiplier` (float): The pre-window reverse multiplier.
        - `post_window_reverse_multiplier` (float): The post-window reverse multiplier.
        - `force_apply_attn2` (bool): Whether to force apply attention 2.
        - `rescale_search_tolerance` (int): The rescale search tolerance.
        - `verbose` (int): The verbosity level.
    """
    start_sigma: float
    end_sigma: float
    use_blocks: set
    scale_mode: str = "nearest-exact"
    reverse_scale_mode: str = "nearest-exact"
    # Allows disabling the log warning for incompatible sizes.
    silent: bool = False
    # Mode for trying to avoid using the same window size consecutively.
    last_shift_mode: LastShiftMode = LastShiftMode.GLOBAL
    # Strategy to use when avoiding a duplicate window size.
    last_shift_strategy: LastShiftStrategy = LastShiftStrategy.INCREMENT
    # Allows multiplying the tensor going into/out of the window or window reverse effect.
    pre_window_multiplier: float = 1.0
    post_window_multiplier: float = 1.0
    pre_window_reverse_multiplier: float = 1.0
    post_window_reverse_multiplier: float = 1.0
    force_apply_attn2: bool = False
    rescale_search_tolerance: int = 1
    verbose: int = 0

    @classmethod
    def build(
        cls,
        *,
        ms: object,
        input_blocks: str | list[int],
        middle_blocks: str | list[int],
        output_blocks: str | list[int],
        time_mode: str | TimeMode,
        start_time: float,
        end_time: float,
        **kwargs: dict,
    ) -> object:
        """#### Build a configuration object.

        #### Args:
            - `ms` (object): The model sampling object.
            - `input_blocks` (str | List[int]): The input blocks.
            - `middle_blocks` (str | List[int]): The middle blocks.
            - `output_blocks` (str | List[int]): The output blocks.
            - `time_mode` (str | TimeMode): The time mode.
            - `start_time` (float): The start time.
            - `end_time` (float): The end time.
            - `kwargs` (Dict[str, Any]): Additional keyword arguments.

        #### Returns:
            - `Config`: The configuration object.
        """
        time_mode: TimeMode = TimeMode(time_mode)
        start_sigma, end_sigma = convert_time(ms, time_mode, start_time, end_time)
        input_blocks, middle_blocks, output_blocks = itertools.starmap(
            parse_blocks,
            (
                ("input", input_blocks),
                ("middle", middle_blocks),
                ("output", output_blocks),
            ),
        )
        return cls.__new__(
            cls,
            start_sigma=start_sigma,
            end_sigma=end_sigma,
            use_blocks=input_blocks | middle_blocks | output_blocks,
            **kwargs,
        )

    @staticmethod
    def maybe_multiply(
        t: torch.Tensor,
        multiplier: float = 1.0,
        post: bool = False,
    ) -> torch.Tensor:
        """#### Multiply a tensor by a multiplier.

        #### Args:
            - `t` (torch.Tensor): The input tensor.
            - `multiplier` (float, optional): The multiplier. Defaults to 1.0.
            - `post` (bool, optional): Whether to multiply in-place. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The multiplied tensor.
        """
        if multiplier == 1.0:
            return t
        return t.mul_(multiplier) if post else t * multiplier


class State:
    """#### Class representing the state.

    #### Args:
        - `config` (Config): The configuration object.
    """
    __slots__ = (
        "config",
        "last_block",
        "last_shift",
        "last_shifts",
        "last_sigma",
        "last_warned",
        "window_args",
    )

    def __init__(self, config):
        self.config = config
        self.last_warned = None
        self.reset()

    def reset(self):
        """#### Reset the state."""
        self.window_args = None
        self.last_sigma = None
        self.last_block = None
        self.last_shift = None
        self.last_shifts = {}

    @property
    def pretty_last_block(self) -> str:
        """#### Get a pretty string representation of the last block.

        #### Returns:
            - `str`: The pretty string representation of the last block.
        """
        if self.last_block is None:
            return "unknown"
        bt, bnum = self.last_block
        attstr = "" if not self.config.force_apply_attn2 else "attn2."
        btstr = ("in", "mid", "out")[bt]
        return f"{attstr}{btstr}.{bnum}"

    def maybe_warning(self, s):
        """#### Log a warning if necessary.

        #### Args:
            - `s` (str): The warning message.
        """
        if self.config.silent:
            return
        now = time()
        if (
            self.config.verbose >= 2
            or self.last_warned is None
            or now - self.last_warned >= DEFAULT_WARN_INTERVAL
        ):
            logger.warning(
                f"** jankhidiffusion: MSW-MSA attention({self.pretty_last_block}): {s}",
            )
            self.last_warned = now

    def __repr__(self):
        """#### Get a string representation of the state.

        #### Returns:
            - `str`: The string representation of the state.
        """
        return f"<MSWMSAAttentionState:last_sigma={self.last_sigma}, last_block={self.pretty_last_block}, last_shift={self.last_shift}, last_shifts={self.last_shifts}>"


class ApplyMSWMSAAttention(metaclass=IntegratedNode):
    """#### Class for applying MSW-MSA attention."""
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node applies an attention patch which _may_ slightly improve quality especially when generating at high resolutions. It is a large performance increase on SD1.x, may improve performance on SDXL. This is the advanced version of the node with more parameters, use ApplyMSWMSAAttentionSimple if this seems too complex. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls):
        """#### Get the input types for the class.

        #### Returns:
            - `Dict[str, Any]`: The input types.
        """
        return {
            "required": {
                "input_blocks": (
                    "STRING",
                    {
                        "default": "1,2",
                        "tooltip": "Comma-separated list of input blocks to patch. Default is for SD1.x, you can try 4,5 for SDXL",
                    },
                ),
                "middle_blocks": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated list of middle blocks to patch. Generally not recommended.",
                    },
                ),
                "output_blocks": (
                    "STRING",
                    {
                        "default": "9,10,11",
                        "tooltip": "Comma-separated list of output blocks to patch. Default is for SD1.x, you can try 3,4,5 for SDXL",
                    },
                ),
                "time_mode": (
                    tuple(str(val) for val in TimeMode),
                    {
                        "default": "percent",
                        "tooltip": "Time mode controls how to interpret the values in start_time and end_time.",
                    },
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time the MSW-MSA attention effect starts applying - value is inclusive.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 999.0,
                        "round": False,
                        "step": 0.01,
                        "tooltip": "Time the MSW-MSA attention effect ends - value is inclusive.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch with the MSW-MSA attention effect.",
                    },
                ),
            },
            "optional": {
                "yaml_parameters": (
                    "STRING",
                    {
                        "tooltip": "Allows specifying custom parameters via YAML. You can also override any of the normal parameters by key. This input can be converted into a multiline text widget. See main README for possible options. Note: When specifying paramaters this way, there is very little error checking.",
                        "dynamicPrompts": False,
                        "multiline": True,
                        "defaultInput": True,
                    },
                ),
            },
        }

    # reference: https://github.com/microsoft/Swin-Transformer
    # Window functions adapted from https://github.com/megvii-research/HiDiffusion
    @staticmethod
    def window_partition(
        x: torch.Tensor,
        state: State,
        window_index: int,
    ) -> torch.Tensor:
        """#### Partition a tensor into windows.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `state` (State): The state object.
            - `window_index` (int): The window index.

        #### Returns:
            - `torch.Tensor`: The partitioned tensor.
        """
        config = state.config
        scale_mode = config.scale_mode
        x = config.maybe_multiply(x, config.pre_window_multiplier)
        window_size, shift_size, height, width = state.window_args[window_index]
        do_rescale = (height % 2 + width % 2) != 0
        if do_rescale:
            if scale_mode == "skip":
                state.maybe_warning(
                    "Incompatible latent size - skipping MSW-MSA attention.",
                )
                return x
            if scale_mode == "disabled":
                state.maybe_warning(
                    "Incompatible latent size - trying to proceed anyway. This may result in an error.",
                )
                do_rescale = False
            else:
                state.maybe_warning(
                    "Incompatible latent size - applying scaling workaround. Note: This may reduce quality - use resolutions that are multiples of 64 when possible.",
                )
        batch, _features, channels = x.shape
        wheight, wwidth = window_size
        x = x.view(batch, height, width, channels)
        if do_rescale:
            x = (
                scale_samples(
                    x.permute(0, 3, 1, 2).contiguous(),
                    wwidth * 2,
                    wheight * 2,
                    mode=scale_mode,
                    sigma=state.last_sigma,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=-shift_size, dims=(1, 2))
        x = x.view(batch, 2, wheight, 2, wwidth, channels)
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size.height, window_size.width, channels)
        )
        return config.maybe_multiply(
            windows.view(-1, window_size.sum, channels),
            config.post_window_multiplier,
        )

    @staticmethod
    def window_reverse(
        windows: torch.Tensor,
        state: State,
        window_index: int = 0,
    ) -> torch.Tensor:
        """#### Reverse the window partitioning of a tensor.

        #### Args:
            - `windows` (torch.Tensor): The input windows tensor.
            - `state` (State): The state object.
            - `window_index` (int, optional): The window index. Defaults to 0.

        #### Returns:
            - `torch.Tensor`: The reversed tensor.
        """
        config = state.config
        windows = config.maybe_multiply(windows, config.pre_window_reverse_multiplier)
        window_size, shift_size, height, width = state.window_args[window_index]
        do_rescale = (height % 2 + width % 2) != 0
        if do_rescale:
            if config.scale_mode == "skip":
                return windows
            if config.scale_mode == "disabled":
                do_rescale = False
        batch, _features, channels = windows.shape
        wheight, wwidth = window_size
        windows = windows.view(-1, wheight, wwidth, channels)
        batch = int(windows.shape[0] / 4)
        x = windows.view(batch, 2, 2, wheight, wwidth, -1)
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch, wheight * 2, wwidth * 2, -1)
        )
        if shift_size.sum > 0:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))
        if do_rescale:
            x = (
                scale_samples(
                    x.permute(0, 3, 1, 2).contiguous(),
                    width,
                    height,
                    mode=config.reverse_scale_mode,
                    sigma=state.last_sigma,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        return config.maybe_multiply(
            x.view(batch, height * width, channels),
            config.post_window_reverse_multiplier,
        )

    @staticmethod
    def get_window_args(
        config: Config,
        n: torch.Tensor,
        orig_shape: tuple,
        shift: int,
    ) -> tuple[WindowSize, ShiftSize, int, int]:
        """#### Get window arguments for MSW-MSA attention.

        #### Args:
            - `config` (Config): The configuration object.
            - `n` (torch.Tensor): The input tensor.
            - `orig_shape` (tuple): The original shape of the tensor.
            - `shift` (int): The shift value.

        #### Returns:
            - `tuple[WindowSize, ShiftSize, int, int]`: The window size, shift size, height, and width.
        """
        _batch, features, _channels = n.shape
        orig_height, orig_width = orig_shape[-2:]

        width, height = rescale_size(
            orig_width,
            orig_height,
            features,
            tolerance=config.rescale_search_tolerance,
        )
        # if (height, width) != (orig_height, orig_width):
        #     print(
        #         f"\nRESC: features={features}, orig={(orig_height, orig_width)}, new={(height, width)}",
        #     )
        wheight, wwidth = math.ceil(height / 2), math.ceil(width / 2)

        if shift == 0:
            shift_size = ShiftSize(0, 0)
        elif shift == 1:
            shift_size = ShiftSize(wheight // 4, wwidth // 4)
        elif shift == 2:
            shift_size = ShiftSize(wheight // 4 * 2, wwidth // 4 * 2)
        else:
            shift_size = ShiftSize(wheight // 4 * 3, wwidth // 4 * 3)
        return (WindowSize(wheight, wwidth), shift_size, height, width)

    @staticmethod
    def get_shift(
        curr_block: tuple,
        state: State,
        *,
        shift_count=4,
    ) -> int:
        """#### Get the shift value for MSW-MSA attention.

        #### Args:
            - `curr_block` (tuple): The current block.
            - `state` (State): The state object.
            - `shift_count` (int, optional): The shift count. Defaults to 4.

        #### Returns:
            - `int`: The shift value.
        """
        mode = state.config.last_shift_mode
        strat = state.config.last_shift_strategy
        shift = int(torch.rand(1, device="cpu").item() * shift_count)
        block_last_shift = state.last_shifts.get(curr_block)
        last_shift = state.last_shift
        if mode == LastShiftMode.BOTH:
            avoid = {block_last_shift, last_shift}
        elif mode == LastShiftMode.BLOCK:
            avoid = {block_last_shift}
        elif mode == LastShiftMode.GLOBAL:
            avoid = {last_shift}
        else:
            avoid = {}
        if shift in avoid:
            if strat == LastShiftStrategy.DECREMENT:
                while shift in avoid:
                    shift -= 1
                    if shift < 0:
                        shift = shift_count - 1
            elif strat == LastShiftStrategy.RETRY:
                while shift in avoid:
                    shift = int(torch.rand(1, device="cpu").item() * shift_count)
            else:
                # Increment
                while shift in avoid:
                    shift = (shift + 1) % shift_count
        return shift

    @classmethod
    def patch(
        cls,
        *,
        model: ModelPatcher.ModelPatcher,
        yaml_parameters: str | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[ModelPatcher.ModelPatcher]:
        """#### Patch the model with MSW-MSA attention.

        #### Args:
            - `model` (ModelPatcher.ModelPatcher): The model patcher.
            - `yaml_parameters` (str | None, optional): The YAML parameters. Defaults to None.
            - `kwargs` (dict[str, Any]): Additional keyword arguments.

        #### Returns:
            - `tuple[ModelPatcher.ModelPatcher]`: The patched model.
        """
        if yaml_parameters:
            import yaml  # noqa: PLC0415

            extra_params = yaml.safe_load(yaml_parameters)
            if extra_params is None:
                pass
            elif not isinstance(extra_params, dict):
                raise ValueError(
                    "MSWMSAAttention: yaml_parameters must either be null or an object",
                )
            else:
                kwargs |= extra_params
        config = Config.build(
            ms=model.get_model_object("model_sampling"),
            **kwargs,
        )
        if not config.use_blocks:
            return (model,)
        if config.verbose:
            logger.info(
                f"** jankhidiffusion: MSW-MSA Attention: Using config: {config}",
            )

        model = model.clone()
        state = State(config)

        def attn_patch(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            extra_options: dict,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """#### Apply attention patch.

            #### Args:
                - `q` (torch.Tensor): The query tensor.
                - `k` (torch.Tensor): The key tensor.
                - `v` (torch.Tensor): The value tensor.
                - `extra_options` (dict): Additional options.

            #### Returns:
                - `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: The patched tensors.
            """
            state.window_args = None
            sigma = get_sigma(extra_options)
            block = extra_options.get("block", ("missing", 0))
            curr_block = block_to_num(*block)
            if state.last_sigma is not None and sigma > state.last_sigma:
                # logging.warning(
                #     f"Doing reset: block={block}, sigma={sigma}, state={state}",
                # )
                state.reset()
            state.last_block = curr_block
            state.last_sigma = sigma
            if block not in config.use_blocks or not check_time(
                sigma,
                config.start_sigma,
                config.end_sigma,
            ):
                return q, k, v
            orig_shape = extra_options["original_shape"]
            # MSW-MSA
            shift = cls.get_shift(curr_block, state)
            state.last_shifts[curr_block] = state.last_shift = shift
            try:
                # get_window_args() can fail with ValueError in rescale_size() for some weird resolutions/aspect ratios
                #  so we catch it here and skip MSW-MSA attention in that case.
                state.window_args = tuple(
                    cls.get_window_args(config, x, orig_shape, shift)
                    if x is not None
                    else None
                    for x in (q, k, v)
                )
                attn_parts = (q,) if q is not None and q is k and q is v else (q, k, v)
                result = tuple(
                    cls.window_partition(tensor, state, idx)
                    if tensor is not None
                    else None
                    for idx, tensor in enumerate(attn_parts)
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    f"** jankhidiffusion: Exception applying MSW-MSA attention: Incompatible model patches or bad resolution. Try using resolutions that are multiples of 64 or set scale/reverse_scale modes to something other than disabled. Original exception: {exc}",
                )
                state.window_args = None
                return q, k, v
            return result * 3 if len(result) == 1 else result

        def attn_output_patch(n: torch.Tensor, extra_options: dict) -> torch.Tensor:
            """#### Apply attention output patch.

            #### Args:
                - `n` (torch.Tensor): The input tensor.
                - `extra_options` (dict): Additional options.

            #### Returns:
                - `torch.Tensor`: The patched tensor.
            """
            if state.window_args is None or state.last_block != block_to_num(
                *extra_options.get("block", ("missing", 0)),
            ):
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
    """Class representing a simplified version of MSW-MSA Attention."""
    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model patched with the MSW-MSA attention effect.",)
    FUNCTION = "go"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "This node applies an attention patch which _may_ slightly improve quality especially when generating at high resolutions. It is a large performance increase on SD1.x, may improve performance on SDXL. This is the simplified version of the node with less parameters. Use ApplyMSWMSAAttention if you require more control. NOTE: Only supports SD1.x, SD2.x and SDXL."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """#### Get input types for the class.

        #### Returns:
            - `dict`: The input types.
        """
        return {
            "required": {
                "model_type": (
                    ("auto", "SD15", "SDXL"),
                    {
                        "tooltip": "Model type being patched. Generally safe to leave on auto. Choose SD15 for SD 1.4, SD 2.x.",
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Model to patch with the MSW-MSA attention effect.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        model_type: str | ModelType,
        model: ModelPatcher.ModelPatcher,
    ) -> tuple[ModelPatcher.ModelPatcher]:
        """#### Apply the MSW-MSA attention patch.

        #### Args:
            - `model_type` (str | ModelType): The model type.
            - `model` (ModelPatcher.ModelPatcher): The model patcher.

        #### Returns:
            - `tuple[ModelPatcher.ModelPatcher]`: The patched model.
        """
        if model_type == "auto":
            guessed_model_type = guess_model_type(model)
            if guessed_model_type not in SIMPLE_PRESETS:
                raise RuntimeError("Unable to guess model type")
            model_type = guessed_model_type
        else:
            model_type = ModelType(model_type)
        preset = SIMPLE_PRESETS.get(model_type)
        if preset is None:
            errstr = f"Unknown model type {model_type!s}"
            raise ValueError(errstr)
        logger.info(
            f"** ApplyMSWMSAAttentionSimple: Using preset {model_type!s}: in/mid/out blocks [{preset.pretty_blocks}], start/end percent {preset.start_time:.2}/{preset.end_time:.2}",
        )
        return ApplyMSWMSAAttention.patch(model=model, **preset.as_dict)


__all__ = ("ApplyMSWMSAAttention", "ApplyMSWMSAAttentionSimple")
