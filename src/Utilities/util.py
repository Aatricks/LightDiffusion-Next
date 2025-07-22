import importlib
from inspect import isfunction
import itertools
import logging
import math
import os
import safetensors.torch
import torch


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """#### Appends dimensions to the end of a tensor until it has target_dims dimensions.

    #### Args:
        - `x` (torch.Tensor): The input tensor.
        - `target_dims` (int): The target number of dimensions.

    #### Returns:
        - `torch.Tensor`: The expanded tensor.
    """
    dims_to_append = target_dims - x.ndim
    expanded = x[(...,) + (None,) * dims_to_append]
    return expanded.detach().clone() if expanded.device.type == "mps" else expanded


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """#### Convert a tensor to a denoised tensor.

    #### Args:
        - `x` (torch.Tensor): The input tensor.
        - `sigma` (torch.Tensor): The noise level.
        - `denoised` (torch.Tensor): The denoised tensor.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    return (x - denoised) / append_dims(sigma, x.ndim)


def load_torch_file(ckpt: str, safe_load: bool = False, device: str = None) -> dict:
    """#### Load a PyTorch checkpoint file.

    #### Args:
        - `ckpt` (str): The path to the checkpoint file.
        - `safe_load` (bool, optional): Whether to use safe loading. Defaults to False.
        - `device` (str, optional): The device to load the checkpoint on. Defaults to None.

    #### Returns:
        - `dict`: The loaded checkpoint.
    """
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if "weights_only" not in torch.load.__code__.co_varnames:
                logging.warning(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
                )
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


def calculate_parameters(sd: dict, prefix: str = "") -> dict:
    """#### Calculate the parameters of a state dictionary.

    #### Args:
        - `sd` (dict): The state dictionary.
        - `prefix` (str, optional): The prefix for the parameters. Defaults to "".

    #### Returns:
        - `dict`: The calculated parameters.
    """
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def state_dict_prefix_replace(
    state_dict: dict, replace_prefix: str, filter_keys: bool = False
) -> dict:
    """#### Replace the prefix of keys in a state dictionary.

    #### Args:
        - `state_dict` (dict): The state dictionary.
        - `replace_prefix` (str): The prefix to replace.
        - `filter_keys` (bool, optional): Whether to filter keys. Defaults to False.

    #### Returns:
        - `dict`: The updated state dictionary.
    """
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


def lcm_of_list(numbers):
    """Calculate LCM of a list of numbers more efficiently."""
    if not numbers:
        return 1

    result = numbers[0]
    for num in numbers[1:]:
        result = torch.lcm(torch.tensor(result), torch.tensor(num)).item()

    return result


def repeat_to_batch_size(
    tensor: torch.Tensor, batch_size: int, dim: int = 0
) -> torch.Tensor:
    """#### Repeat a tensor to match a specific batch size.

    #### Args:
        - `tensor` (torch.Tensor): The input tensor.
        - `batch_size` (int): The target batch size.
        - `dim` (int, optional): The dimension to repeat. Defaults to 0.

    #### Returns:
        - `torch.Tensor`: The repeated tensor.
    """
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(
            dim * [1]
            + [math.ceil(batch_size / tensor.shape[dim])]
            + [1] * (len(tensor.shape) - 1 - dim)
        ).narrow(dim, 0, batch_size)
    return tensor


def set_attr(obj: object, attr: str, value: any) -> any:
    """#### Set an attribute of an object.

    #### Args:
        - `obj` (object): The object.
        - `attr` (str): The attribute name.
        - `value` (any): The value to set.

    #### Returns:
        - `prev`: The previous attribute value.
    """
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj: object, attr: str, value: any) -> any:
    """#### Set an attribute parameter of an object.

    #### Args:
        - `obj` (object): The object.
        - `attr` (str): The attribute name.
        - `value` (any): The value to set.

    #### Returns:
        - `prev`: The previous attribute value.
    """
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj: object, attr: str, value: any) -> None:
    """#### Copy a value to a parameter of an object.

    #### Args:
        - `obj` (object): The object.
        - `attr` (str): The attribute name.
        - `value` (any): The value to set.
    """
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_obj_from_str(string: str, reload: bool = False) -> object:
    """#### Get an object from a string.

    #### Args:
        - `string` (str): The string.
        - `reload` (bool, optional): Whether to reload the module. Defaults to False.

    #### Returns:
        - `object`: The object.
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_attr(obj: object, attr: str) -> any:
    """#### Get an attribute of an object.

    #### Args:
        - `obj` (object): The object.
        - `attr` (str): The attribute name.

    #### Returns:
        - `obj`: The attribute value.
    """
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def lcm(a: int, b: int) -> int:
    """#### Calculate the least common multiple (LCM) of two numbers.

    #### Args:
        - `a` (int): The first number.
        - `b` (int): The second number.

    #### Returns:
        - `int`: The LCM of the two numbers.
    """
    return abs(a * b) // math.gcd(a, b)


def get_full_path(folder_name: str, filename: str) -> str:
    """#### Get the full path of a file in a folder.

    Args:
        folder_name (str): The folder name.
        filename (str): The filename.

    Returns:
        str: The full path of the file.
    """
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def zero_module(module: torch.nn.Module) -> torch.nn.Module:
    """#### Zero out the parameters of a module.

    #### Args:
        - `module` (torch.nn.Module): The module.

    #### Returns:
        - `torch.nn.Module`: The zeroed module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def append_zero(x: torch.Tensor) -> torch.Tensor:
    """#### Append a zero to the end of a tensor.

    #### Args:
        - `x` (torch.Tensor): The input tensor.

    #### Returns:
        - `torch.Tensor`: The tensor with a zero appended.
    """
    return torch.cat([x, x.new_zeros([1])])


def exists(val: any) -> bool:
    """#### Check if a value exists.

    #### Args:
        - `val` (any): The value.

    #### Returns:
        - `bool`: Whether the value exists.
    """
    return val is not None


def default(val: any, d: any) -> any:
    """#### Get the default value of a variable.

    #### Args:
        - `val` (any): The value.
        - `d` (any): The default value.

    #### Returns:
        - `any`: The default value if the value does not exist.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def write_parameters_to_file(
    prompt_entry: str, neg: str, width: int, height: int, cfg: int
) -> None:
    """#### Write parameters to a file.

    #### Args:
        - `prompt_entry` (str): The prompt entry.
        - `neg` (str): The negative prompt entry.
        - `width` (int): The width.
        - `height` (int): The height.
        - `cfg` (int): The CFG.
    """
    with open("./include/prompt.txt", "w") as f:
        f.write(f"prompt: {prompt_entry}")
        f.write(f"neg: {neg}")
        f.write(f"w: {int(width)}\n")
        f.write(f"h: {int(height)}\n")
        f.write(f"cfg: {int(cfg)}\n")


def load_parameters_from_file() -> tuple:
    """#### Load parameters from a file.

    #### Returns:
        - `str`: The prompt entry.
        - `str`: The negative prompt entry.
        - `int`: The width.
        - `int`: The height.
        - `int`: The CFG.
    """
    with open("./include/prompt.txt", "r") as f:
        lines = f.readlines()
        parameters = {}
        for line in lines:
            # Skip empty lines
            if line.strip() == "":
                continue
            key, value = line.split(": ")
            parameters[key] = value.strip()
        prompt = parameters["prompt"]
        neg = parameters["neg"]
        width = int(parameters["w"])
        height = int(parameters["h"])
        cfg = int(parameters["cfg"])
    return prompt, neg, width, height, cfg


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None


class ProgressBar:
    """#### Class representing a progress bar."""

    def __init__(self, total: int):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK


def get_tiled_scale_steps(
    width: int, height: int, tile_x: int, tile_y: int, overlap: int
) -> int:
    """#### Get the number of steps for tiled scaling.

    #### Args:
        - `width` (int): The width.
        - `height` (int): The height.
        - `tile_x` (int): The tile width.
        - `tile_y` (int): The tile height.
        - `overlap` (int): The overlap.

    #### Returns:
        - `int`: The number of steps.
    """
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


@torch.inference_mode()
def tiled_scale_multidim(
    samples: torch.Tensor,
    function,
    tile: tuple = (64, 64),
    overlap: int = 8,
    upscale_amount: int = 4,
    out_channels: int = 3,
    output_device: str = "cpu",
    downscale: bool = False,
    index_formulas: any = None,
    pbar: any = None,
):
    """#### Scale an image using a tiled approach.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `function` (function): The scaling function.
        - `tile` (tuple, optional): The tile size. Defaults to (64, 64).
        - `overlap` (int, optional): The overlap. Defaults to 8.
        - `upscale_amount` (int, optional): The upscale amount. Defaults to 4.
        - `out_channels` (int, optional): The number of output channels. Defaults to 3.
        - `output_device` (str, optional): The output device. Defaults to "cpu".
        - `downscale` (bool, optional): Whether to downscale. Defaults to False.
        - `index_formulas` (any, optional): The index formulas. Defaults to None.
        - `pbar` (any, optional): The progress bar. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The scaled image.
    """
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim: int, val: int) -> int:
        """#### Get the upscale value.

        #### Args:
            - `dim` (int): The dimension.
            - `val` (int): The value.

        #### Returns:
            - `int`: The upscaled value.
        """
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim: int, val: int) -> int:
        """#### Get the downscale value.

        #### Args:
            - `dim` (int): The dimension.
            - `val` (int): The value.

        #### Returns:
            - `int`: The downscaled value.
        """
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim: int, val: int) -> int:
        """#### Get the upscaled position.

        #### Args:
            - `dim` (int): The dimension.
            - `val` (int): The value.

        #### Returns:
            - `int`: The upscaled position.
        """
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim: int, val: int) -> int:
        """#### Get the downscaled position.

        #### Args:
            - `dim` (int): The dimension.
            - `val` (int): The value.

        #### Returns:
            - `int`: The downscaled position.
        """
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a: list) -> list:
        """#### Multiply a list by the upscale amount.

        #### Args:
            - `a` (list): The list.

        #### Returns:
            - `list`: The multiplied list.
        """
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty(
        [samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]),
        device=output_device,
    )

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]

        # handle entire input fitting in a single tile
        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b : b + 1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]),
            device=output_device,
        )
        out_div = torch.zeros(
            [s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]),
            device=output_device,
        )

        positions = [
            range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d])
            if s.shape[d + 2] > tile[d]
            else [0]
            for d in range(dims)
        ]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

            if pbar is not None:
                pbar.update(1)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(
    samples: torch.Tensor,
    function,
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: int = 4,
    out_channels: int = 3,
    output_device: str = "cpu",
    pbar: any = None,
):
    """#### Scale an image using a tiled approach.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `function` (function): The scaling function.
        - `tile_x` (int, optional): The tile width. Defaults to 64.
        - `tile_y` (int, optional): The tile height. Defaults to 64.
        - `overlap` (int, optional): The overlap. Defaults to 8.
        - `upscale_amount` (int, optional): The upscale amount. Defaults to 4.
        - `out_channels` (int, optional): The number of output channels. Defaults to 3.
        - `output_device` (str, optional): The output device. Defaults to "cpu".
        - `pbar` (any, optional): The progress bar. Defaults to None.

    #### Returns:
        - The scaled image.
    """
    return tiled_scale_multidim(
        samples,
        function,
        (tile_y, tile_x),
        overlap=overlap,
        upscale_amount=upscale_amount,
        out_channels=out_channels,
        output_device=output_device,
        pbar=pbar,
    )
