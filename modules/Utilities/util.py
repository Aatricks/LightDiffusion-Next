from inspect import isfunction
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
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
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
    out = {}
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


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int, dim: int = 0) -> torch.Tensor:
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
    with open("./_internal/prompt.txt", "w") as f:
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
    with open("./_internal/prompt.txt", "r") as f:
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
