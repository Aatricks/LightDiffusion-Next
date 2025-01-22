from __future__ import annotations
import importlib
from inspect import isfunction
import itertools
import logging
import math
import os
import pickle
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

load = pickle.load

class Empty:
    pass

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)

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

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



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
        - `cfg` (int): The 
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
        - `int`: The 
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

def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols

@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None, pbar=None):
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
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

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]

        # handle entire input fitting in a single tile
        if all(s.shape[d+2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar is not None:
                pbar.update(1)
            continue

        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) if s.shape[d+2] > tile[d] else [0] for d in range(dims)]

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

        output[b:b+1] = out/out_div
    return output

def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device, pbar=pbar)


import logging
import math
import threading
import torch
import torchsde
from torch import nn



disable_gui = False

logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    """#### Create a beta schedule.

    #### Args:
        - `schedule` (str): The schedule type.
        - `n_timestep` (int): The number of timesteps.
        - `linear_start` (float, optional): The linear start value. Defaults to 1e-4.
        - `linear_end` (float, optional): The linear end value. Defaults to 2e-2.
        - `cosine_s` (float, optional): The cosine s value. Defaults to 8e-3.

    #### Returns:
        - `list`: The beta schedule.
    """
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas


def checkpoint(func, inputs, params, flag):
    """#### Create a checkpoint.

    #### Args:
        - `func` (callable): The function to checkpoint.
        - `inputs` (list): The inputs to the function.
        - `params` (list): The parameters of the function.
        - `flag` (bool): The checkpoint flag.

    #### Returns:
        - `any`: The checkpointed output.
    """
    return func(*inputs)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """#### Create a timestep embedding.

    #### Args:
        - `timesteps` (torch.Tensor): The timesteps.
        - `dim` (int): The embedding dimension.
        - `max_period` (int, optional): The maximum period. Defaults to 10000.
        - `repeat_only` (bool, optional): Whether to repeat only. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The timestep embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

def timestep_embedding_flux(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """#### Create a timestep embedding.

    #### Args:
        - `timesteps` (torch.Tensor): The timesteps.
        - `dim` (int): The embedding dimension.
        - `max_period` (int, optional): The maximum period. Defaults to 10000.
        - `repeat_only` (bool, optional): Whether to repeat only. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The timestep embedding.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """#### Get the sigmas for Karras 

    constructs the noise schedule of Karras et al. (2022).

    #### Args:
        - `n` (int): The number of sigmas.
        - `sigma_min` (float): The minimum sigma value.
        - `sigma_max` (float): The maximum sigma value.
        - `rho` (float, optional): The rho value. Defaults to 7.0.
        - `device` (str, optional): The device to use. Defaults to "cpu".

    #### Returns:
        - `torch.Tensor`: The sigmas.
    """
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """
    #### Calculate the ancestral step in a diffusion process.

    This function computes the values of `sigma_down` and `sigma_up` based on the
    input parameters `sigma_from`, `sigma_to`, and `eta`. These values are used
    in the context of diffusion models to determine the next step in the process.

    #### Parameters:
        - `sigma_from` (float): The starting value of sigma.
        - `sigma_to` (float): The target value of sigma.
        - `eta` (float, optional): A scaling factor for the step size. Default is 1.0.

    #### Returns:
    - `tuple`: A tuple containing `sigma_down` and `sigma_up`:
        - `sigma_down` (float): The computed value of sigma for the downward step.
        - `sigma_up` (float): The computed value of sigma for the upward step.
    """
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    """
    #### Returns a noise sampling function that generates random noise with the same shape as the input tensor `x`.

    #### Args:
        - `x` (torch.Tensor): The input tensor whose shape will be used to generate random noise.

    #### Returns:
        - `function`: A function that takes two arguments, `sigma` and `sigma_next`, and returns a tensor of random noise
                  with the same shape as `x`.
    """
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """#### A class to represent a batched Brownian tree for stochastic differential equations.

    #### Attributes:
        - `cpu_tree` : bool
            Indicates if the tree is on CPU.
        - `sign` : int
            Sign indicating the order of t0 and t1.
        - `batched` : bool
            Indicates if the tree is batched.
        - `trees` : list
            List of BrownianTree instances.

    #### Methods:
        - `__init__(x, t0, t1, seed=None, **kwargs)`:
            Initializes the BatchedBrownianTree with given parameters.
        - `sort(a, b)`:
            Static method to sort two values and return them along with a sign.
        - `__call__(t0, t1)`:
            Calls the Brownian tree with given time points t0 and t1.
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        seed = [seed]
        self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs)
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        """#### Sort two values and return them along with a sign.

        #### Args:
            - `a` (float): The first value.
            - `b` (float): The second value.

        #### Returns:
            - `tuple`: A tuple containing the sorted values and a sign:
        """
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        """#### Call the Brownian tree with given time points t0 and t1.

        #### Args:
            - `t0` (torch.Tensor): The starting time point.
            - `t1` (torch.Tensor): The target time point.

        #### Returns:
            - `torch.Tensor`: The Brownian tree values.
        """
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack(
            [
                tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device)
                for tree in self.trees
            ]
        ) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """#### A class to sample noise using a Brownian tree approach.

    #### Attributes:
        - `transform` (callable): A function to transform the sigma values.
        - `tree` (BatchedBrownianTree): An instance of the BatchedBrownianTree class.

    #### Methods:
        - `__init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False)`:
            Initializes the BrownianTreeNoiseSampler with the given parameters.
        - `__call__(self, sigma, sigma_next)`:
            Samples noise between the given sigma values.
    """

    def __init__(
        self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False
    ):
        """#### Initializes the BrownianTreeNoiseSampler with the given parameters.

        #### Args:
            - `x` (Tensor): The initial tensor.
            - `sigma_min` (float): The minimum sigma value.
            - `sigma_max` (float): The maximum sigma value.
            - `seed` (int, optional): The seed for random number generation. Defaults to None.
            - `transform` (callable, optional): A function to transform the sigma values. Defaults to identity function.
            - `cpu` (bool, optional): Whether to use CPU for computations. Defaults to False.
        """
        self.transform = transform
        t0, t1 = (
            self.transform(torch.as_tensor(sigma_min)),
            self.transform(torch.as_tensor(sigma_max)),
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        """#### Samples noise between the given sigma values.

        #### Args:
            - `sigma` (float): The current sigma value.
            - `sigma_next` (float): The next sigma value.

        #### Returns:
            - `Tensor`: The sampled noise.
        """
        t0, t1 = (
            self.transform(torch.as_tensor(sigma)),
            self.transform(torch.as_tensor(sigma_next)),
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


class PIDStepSizeController:
    """#### A PID (Proportional-Integral-Derivative) Step Size Controller for adaptive step size selection.

    #### Attributes:
        - `h` (float): Initial step size.
        - `b1` (float): Coefficient for the proportional term.
        - `b2` (float): Coefficient for the integral term.
        - `b3` (float): Coefficient for the derivative term.
        - `accept_safety` (float): Safety factor for accepting a proposed step size.
        - `eps` (float): Small value to prevent division by zero.
        - `errs` (list): List to store inverse errors for PID control.

    #### Methods:
        - `__init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8)`:
            Initializes the PIDStepSizeController with given parameters.
        - `limiter(self, x)`:
            Limits the factor to prevent excessive changes in step size.
        - `propose_step(self, error)`:
            Proposes a new step size based on the given error and updates internal state.
    """

    def __init__(
        self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8
    ):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        """#### Limit the factor to prevent excessive changes in step size.

        #### Args:
            - `x` (float): The factor to limit.

        #### Returns:
            - `float`: The limited factor.
        """
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        """#### Propose a new step size based on the given error and update the internal state.

        #### Args:
            - `error` (float): The error value.

        #### Returns:
            - `bool`: True if the proposed step size is accepted, False otherwise.
        """
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = (
            self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        )
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """#### DPMSolver is a class for solving differential equations using the DPM-Solver algorithm.

    #### Args:
        - `model` (nn.Module): The model to be used for solving the differential equations.
        - `extra_args` (dict, optional): Additional arguments to be passed to the model. Defaults to None.
        - `eps_callback` (callable, optional): A callback function to be called after computing epsilon. Defaults to None.
        - `info_callback` (callable, optional): A callback function to be called with information about the solver's progress. Defaults to None.

    #### Methods:
        - `t(sigma)`:
            Converts sigma to time t.
        - `sigma(t)`:
            Converts time t to sigma.
        - `eps(eps_cache, key, x, t, *args, **kwargs)`:
            Computes the epsilon value for the given inputs and caches the result.
        - `dpm_solver_2_step(x, t, t_next, r1=1/2, eps_cache=None)`:
            Performs a 2-step DPM-Solver update.
        - `dpm_solver_3_step(x, t, t_next, r1=1/3, r2=2/3, eps_cache=None)`:
            Performs a 3-step DPM-Solver update.
        - `dpm_solver_adaptive(x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0.0, icoeff=1.0, dcoeff=0.0, accept_safety=0.81, eta=0.0, s_noise=1.0, noise_sampler=None)`:
            Performs an adaptive DPM-Solver update with error control and step size adaptation.
    """

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        """#### Convert sigma to time t.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.

        #### Returns:
            - `torch.Tensor`: The time t.
        """
        return -sigma.log()

    def sigma(self, t):
        """#### Convert time t to sigma.

        #### Args:
            - `t` (torch.Tensor): The time t.

        #### Returns:
            - `torch.Tensor`: The sigma value.
        """
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        """#### Compute the epsilon value for the given inputs and cache the result.

        #### Args:
            - `eps_cache` (dict): The cache for epsilon values.
            - `key` (str): The key for the cache.
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The time t.

        #### Returns:
            - `tuple`: A tuple containing the epsilon value and the updated cache.
        """
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (
            x - self.model(x, sigma, *args, **self.extra_args, **kwargs)
        ) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        """#### Perform a 2-step DPM-Solver update.

        #### Args:
            -`x` (torch.Tensor): The input tensor.
            -`t` (torch.Tensor): The current time t.
            -`t_next` (torch.Tensor): The target time t.
            -`r1` (float, optional): The ratio for the first step. Defaults to 1/2.
            -`eps_cache` (dict, optional): The cache for epsilon values. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and the updated cache.
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        )
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        """#### Perform a 3-step DPM-Solver update.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The current time t.
            - `t_next` (torch.Tensor): The target time t.
            - `r1` (float, optional): The ratio for the first step. Defaults to 1/3.
            - `r2` (float, optional): The ratio for the second step. Defaults to 2/3.
            - `eps_cache` (dict, optional): The cache for epsilon values. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and the updated cache.
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (
            x
            - self.sigma(s2) * (r2 * h).expm1() * eps
            - self.sigma(s2)
            * (r2 / r1)
            * ((r2 * h).expm1() / (r2 * h) - 1)
            * (eps_r1 - eps)
        )
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        )
        return x_3, eps_cache

    def dpm_solver_adaptive(
        self,
        x,
        t_start,
        t_end,
        order=3,
        rtol=0.05,
        atol=0.0078,
        h_init=0.05,
        pcoeff=0.0,
        icoeff=1.0,
        dcoeff=0.0,
        accept_safety=0.81,
        eta=0.0,
        s_noise=1.0,
        noise_sampler=None,
        pipeline=False,
    ):
        """#### Perform an adaptive DPM-Solver update with error control and step size adaptation.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t_start` (torch.Tensor): The starting time t.
            - `t_end` (torch.Tensor): The target time t.
            - `order` (int, optional): The order of the DPM-Solver. Defaults to 3.
            - `rtol` (float, optional): The relative tolerance for error control. Defaults to 0.05.
            - `atol` (float, optional): The absolute tolerance for error control. Defaults to 0.0078.
            - `h_init` (float, optional): The initial step size. Defaults to 0.05.
            - `pcoeff` (float, optional): Coefficient for the proportional term in the PID controller. Defaults to 0.0.
            - `icoeff` (float, optional): Coefficient for the integral term in the PID controller. Defaults to 1.0.
            - `dcoeff` (float, optional): Coefficient for the derivative term in the PID controller. Defaults to 0.0.
            - `accept_safety` (float, optional): Safety factor for accepting a proposed step size. Defaults to 0.81.
            - `eta` (float, optional): The eta parameter for the ancestral step. Defaults to 0.0.
            - `s_noise` (float, optional): The noise scaling factor. Defaults to 1.0.
            - `noise_sampler` (callable, optional): A function to sample noise. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and information about the solver's progress.
        """
        global disable_gui
        disable_gui = True if pipeline is True else False
        noise_sampler = (
            default_noise_sampler(x) if noise_sampler is None else noise_sampler
        )
        forward = t_end > t_start
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(
            h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety
        )
        info = {"steps": 0, "nfe": 0, "n_accept": 0, "n_reject": 0}
        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = (
                torch.minimum(t_end, s + pid.h)
                if forward
                else torch.maximum(t_end, s + pid.h)
            )
            t_, su = t, 0.0
            eps, eps_cache = self.eps(eps_cache, "eps", x, s)
            x - self.sigma(s) * eps
            x_low, eps_cache = self.dpm_solver_2_step(
                x, s, t_, r1=1 / 3, eps_cache=eps_cache
            )
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info["n_accept"] += 1
            else:
                info["n_reject"] += 1
            info["nfe"] += order
            info["steps"] += 1
        return x, info


import logging
import platform
import sys
from enum import Enum
from typing import Tuple, Union

import psutil
import torch

# Only this extra line of code is required to use oneDNN Graph
torch.jit.enable_onednn_fusion(True)
# torch.optimizer.zero_grad(set_to_none=True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.profiler.profile(enabled=False)


# FIXME: This is a workaround for the torch.backends.cuda.matmul.allow_tf32 attribute error

# if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.12.0"):
#     torch.backends.cuda.matmul.allow_tf32 = True


class VRAMState(Enum):
    """#### Enum for VRAM states.
    """
    DISABLED = 0  # No vram present: no need to move _internal to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but _internal still need to be moved between both.


class CPUState(Enum):
    """#### Enum for CPU states.
    """
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

directml_enabled = False
try:
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass


def is_intel_xpu() -> bool:
    """#### Check if Intel XPU is available.

    #### Returns:
        - `bool`: Whether Intel XPU is available.
    """
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device() -> torch.device:
    """#### Get the torch device.
    
    #### Returns:
        - `torch.device`: The torch device.
    """
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev: torch.device = None, torch_total_too: bool = False) -> int:
    """#### Get the total memory.

    #### Args:
        - `dev` (torch.device, optional): The device. Defaults to None.
        - `torch_total_too` (bool, optional): Whether to get the total memory in PyTorch. Defaults to False.

    #### Returns:
        - `int`: The total memory.
    """
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info(
    "Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram)
)
try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except:
        pass
    try:
        XFORMERS_VERSION = xformers.version.__version__
        logging.info("xformers version: {}".format(XFORMERS_VERSION))
        if XFORMERS_VERSION.startswith("0.0.18"):
            logging.warning(
                "\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images."
            )
            logging.warning(
                "Please downgrade or upgrade xformers to a different version.\n"
            )
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False


def is_nvidia() -> bool:
    """#### Checks if user has an Nvidia GPU

    #### Returns
        - `bool`: Whether the GPU is Nvidia
    """
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION is False:
                ENABLE_PYTORCH_ATTENTION = True
            if (
                torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_properties(torch.cuda.current_device()).major
                >= 8
            ):
                VAE_DTYPE = torch.bfloat16
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


FORCE_FP32 = False
FORCE_FP16 = False

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = False

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")


def get_torch_device_name(device: torch.device) -> str:
    """#### Get the name of the torch compatible device

    #### Args:
        - `device` (torch.device): the device

    #### Returns:
        - `str`: the name of the device
    """
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(
                device, torch.cuda.get_device_name(device), allocator_backend
            )
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")

logging.info("VAE dtype: {}".format(VAE_DTYPE))

current_loaded_models = []


def module_size(module: torch.nn.Module) -> int:
    """#### Get the size of a module
    
    #### Args:
        - `module` (torch.nn.Module): The module
    
    #### Returns:
        - `int`: The size of the module
    """
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    """#### Class to load a model
    """
    def __init__(self, model: torch.nn.Module):
        """#### Initialize the class
        
        #### Args:
            - `model`: The model
        """
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def model_memory(self):
        """#### Get the model memory
        
        #### Returns:
            - `int`: The model memory
        """
        return self.model.model_size()

    
    def model_offloaded_memory(self):
        """#### Get the offloaded model memory
        
        #### Returns:
            - `int`: The offloaded model memory
        """
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device: torch.device) -> int:
        """#### Get the required model memory
        
        #### Args:
            - `device`: The device
        
        #### Returns:
            - `int`: The required model memory
        """
        if hasattr(self.model, 'current_loaded_device') and device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory: int = 0, force_patch_weights: bool = False) -> torch.nn.Module:
        """#### Load the model
        
        #### Args:
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch the weights. Defaults to False.
        
        #### Returns:
            - `torch.nn.Module`: The real model
        """
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        try:
            if hasattr(self.model, "patch_model_lowvram") and  lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(
                    device_to=patch_model_to,
                    lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights,
                )
            else:
                self.real_model = self.model.patch_model(
                    device_to=patch_model_to, patch_weights=load_weights
                )
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e
        self.weights_loaded = True
        return self.real_model

    def model_load_flux(self, lowvram_model_memory: int = 0, force_patch_weights: bool = False) -> torch.nn.Module:
        """#### Load the model
        
        #### Args:
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch the weights. Defaults to False.
        
        #### Returns:
            - `torch.nn.Module`: The real model
        """
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        if self.model.loaded_size() > 0:
            use_more_vram = lowvram_model_memory
            if use_more_vram == 0:
                use_more_vram = 1e32
            self.model_use_more_vram(use_more_vram)
        else:
            try:
                self.real_model = self.model.patch_model_flux(
                    device_to=patch_model_to,
                    lowvram_model_memory=lowvram_model_memory,
                    load_weights=load_weights,
                    force_patch_weights=force_patch_weights,
                )
            except Exception as e:
                self.model.unpatch_model(self.model.offload_device)
                self.model_unload()
                raise e

        if (
            is_intel_xpu()
            and "ipex" in globals()
            and self.real_model is not None
        ):
            import ipex
            with torch.no_grad():
                self.real_model = ipex.optimize(
                    self.real_model.eval(),
                    inplace=True,
                    graph_mode=True,
                    concat_linear=True,
                )

        self.weights_loaded = True
        return self.real_model
    
    def should_reload_model(self, force_patch_weights: bool = False) -> bool:
        """#### Checks if the model should be reloaded

        #### Args:
            - `force_patch_weights` (bool, optional): If model reloading should be enforced. Defaults to False.

        #### Returns:
            - `bool`: Whether the model should be reloaded
        """
        if force_patch_weights and self.model.lowvram_patch_counter > 0:
            return True
        return False

    def model_unload(self, unpatch_weights: bool = True) -> None:
        """#### Unloads the patched model

        #### Args:
            - `unpatch_weights` (bool, optional): Whether the weights should be unpatched. Defaults to True.
        """
        self.model.unpatch_model(
            self.model.offload_device, unpatch_weights=unpatch_weights
        )
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def model_use_more_vram(self, extra_memory):
        return self.model.partially_load(self.device, extra_memory)
    
    def __eq__(self, other: torch.nn.Module) -> bool:
        """#### Verify if the model is equal to another

        #### Args:
            - `other` (torch.nn.Module): the other model

        #### Returns:
            - `bool`: Whether the two models are equal
        """
        return self.model is other.model


def minimum_inference_memory() -> int:
    """#### The minimum memory requirement for inference, equals to 1024^3

    #### Returns:
        - `int`: the memory requirement
    """
    return 1024 * 1024 * 1024


def unload_model_clones(model: torch.nn.Module, unload_weights_only:bool = True, force_unload: bool = True) -> bool:
    """#### Unloads the model clones

    #### Args:
        - `model` (torch.nn.Module): The model
        - `unload_weights_only` (bool, optional): Whether to unload only the weights. Defaults to True.
        - `force_unload` (bool, optional): Whether to force the unload. Defaults to True.

    #### Returns:
        - `bool`: Whether the model was unloaded
    """
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight is False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight


def free_memory(memory_required: int, device: torch.device, keep_loaded: list = []) -> None:
    """#### Free memory
    
    #### Args:
        - `memory_required` (int): The required memory
        - `device` (torch.device): The device
        - `keep_loaded` (list, optional): The list of loaded models to keep. Defaults to [].
    """
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append(
                    (sys.getrefcount(shift_model.model), shift_model.model_memory(), i)
                )

    for x in sorted(can_unload):
        i = x[-1]
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(
                device, torch_free_too=True
            )
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()

def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break

WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    EXTRA_RESERVED_VRAM = (
        600 * 1024 * 1024
    )  # Windows is higher because of the shared vram issue

def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM

def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

def load_models_gpu(models: list, memory_required: int = 0, force_patch_weights: bool = False, minimum_memory_required=None, force_full_load=False, flux_enabled: bool = False) -> None:
    """#### Load models on the GPU
    
    #### Args:
        - `models`(list): The models
        - `memory_required` (int, optional): The required memory. Defaults to 0.
        - `force_patch_weights` (bool, optional): Whether to force patch the weights. Defaults to False.
        - `minimum_memory_required` (int, optional): The minimum memory required. Defaults to None.
        - `force_full_load` (bool, optional
        - `flux_enabled` (bool, optional): Whether flux is enabled. Defaults to False.
    """
    global vram_state
    if not flux_enabled:
    
        inference_memory = minimum_inference_memory()
        extra_mem = max(inference_memory, memory_required)

        models = set(models)

        models_to_load = []
        models_already_loaded = []
        for x in models:
            loaded_model = LoadedModel(x)
            loaded = None

            try:
                loaded_model_index = current_loaded_models.index(loaded_model)
            except:
                loaded_model_index = None

            if loaded_model_index is not None:
                loaded = current_loaded_models[loaded_model_index]
                if loaded.should_reload_model(force_patch_weights=force_patch_weights):
                    current_loaded_models.pop(loaded_model_index).model_unload(
                        unpatch_weights=True
                    )
                    loaded = None
                else:
                    models_already_loaded.append(loaded)

            if loaded is None:
                if hasattr(x, "model"):
                    logging.info(f"Requested to load {x.model.__class__.__name__}")
                models_to_load.append(loaded_model)

        if len(models_to_load) == 0:
            devs = set(map(lambda a: a.device, models_already_loaded))
            for d in devs:
                if d != torch.device("cpu"):
                    free_memory(extra_mem, d, models_already_loaded)
            return

        logging.info(
            f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}"
        )

        total_memory_required = {}
        for loaded_model in models_to_load:
            if (
                unload_model_clones(
                    loaded_model.model, unload_weights_only=True, force_unload=False
                )
                is True
            ):  # unload clones where the weights are different
                total_memory_required[loaded_model.device] = total_memory_required.get(
                    loaded_model.device, 0
                ) + loaded_model.model_memory_required(loaded_model.device)

        for device in total_memory_required:
            if device != torch.device("cpu"):
                free_memory(
                    total_memory_required[device] * 1.3 + extra_mem,
                    device,
                    models_already_loaded,
                )

        for loaded_model in models_to_load:
            weights_unloaded = unload_model_clones(
                loaded_model.model, unload_weights_only=False, force_unload=False
            )  # unload the rest of the clones where the weights can stay loaded
            if weights_unloaded is not None:
                loaded_model.weights_loaded = not weights_unloaded

        for loaded_model in models_to_load:
            model = loaded_model.model
            torch_dev = model.load_device
            if is_device_cpu(torch_dev):
                vram_set_state = VRAMState.DISABLED
            else:
                vram_set_state = vram_state
            lowvram_model_memory = 0
            if lowvram_available and (
                vram_set_state == VRAMState.LOW_VRAM
                or vram_set_state == VRAMState.NORMAL_VRAM
            ):
                model_size = loaded_model.model_memory_required(torch_dev)
                current_free_mem = get_free_memory(torch_dev)
                lowvram_model_memory = int(
                    max(64 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3)
                )
                if model_size > (
                    current_free_mem - inference_memory
                ):  # only switch to lowvram if really necessary
                    vram_set_state = VRAMState.LOW_VRAM
                else:
                    lowvram_model_memory = 0

            if vram_set_state == VRAMState.NO_VRAM:
                lowvram_model_memory = 64 * 1024 * 1024

            loaded_model.model_load(
                lowvram_model_memory, force_patch_weights=force_patch_weights
            )
            current_loaded_models.insert(0, loaded_model)
        return
    else:
        inference_memory = minimum_inference_memory()
        extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
        if minimum_memory_required is None:
            minimum_memory_required = extra_mem
        else:
            minimum_memory_required = max(
                inference_memory, minimum_memory_required + extra_reserved_memory()
            )

        models = set(models)

        models_to_load = []
        models_already_loaded = []
        for x in models:
            loaded_model = LoadedModel(x)
            loaded = None

            try:
                loaded_model_index = current_loaded_models.index(loaded_model)
            except:
                loaded_model_index = None

            if loaded_model_index is not None:
                loaded = current_loaded_models[loaded_model_index]
                if loaded.should_reload_model(
                    force_patch_weights=force_patch_weights
                ):  # TODO: cleanup this model reload logic
                    current_loaded_models.pop(loaded_model_index).model_unload(
                        unpatch_weights=True
                    )
                    loaded = None
                else:
                    loaded.currently_used = True
                    models_already_loaded.append(loaded)

            if loaded is None:
                if hasattr(x, "model"):
                    logging.info(f"Requested to load {x.model.__class__.__name__}")
                models_to_load.append(loaded_model)

        if len(models_to_load) == 0:
            devs = set(map(lambda a: a.device, models_already_loaded))
            for d in devs:
                if d != torch.device("cpu"):
                    free_memory(
                        extra_mem + offloaded_memory(models_already_loaded, d),
                        d,
                        models_already_loaded,
                    )
                    free_mem = get_free_memory(d)
                    if free_mem < minimum_memory_required:
                        logging.info(
                            "Unloading models for lowram load."
                        )  # TODO: partial model unloading when this case happens, also handle the opposite case where models can be unlowvramed.
                        models_to_load = free_memory(minimum_memory_required, d)
                        logging.info("{} models unloaded.".format(len(models_to_load)))
                    else:
                        use_more_memory(
                            free_mem - minimum_memory_required, models_already_loaded, d
                        )
            if len(models_to_load) == 0:
                return

        logging.info(
            f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}"
        )

        total_memory_required = {}
        for loaded_model in models_to_load:
            unload_model_clones(
                loaded_model.model, unload_weights_only=True, force_unload=False
            )  # unload clones where the weights are different
            total_memory_required[loaded_model.device] = total_memory_required.get(
                loaded_model.device, 0
            ) + loaded_model.model_memory_required(loaded_model.device)

        for loaded_model in models_already_loaded:
            total_memory_required[loaded_model.device] = total_memory_required.get(
                loaded_model.device, 0
            ) + loaded_model.model_memory_required(loaded_model.device)

        for loaded_model in models_to_load:
            weights_unloaded = unload_model_clones(
                loaded_model.model, unload_weights_only=False, force_unload=False
            )  # unload the rest of the clones where the weights can stay loaded
            if weights_unloaded is not None:
                loaded_model.weights_loaded = not weights_unloaded

        for device in total_memory_required:
            if device != torch.device("cpu"):
                free_memory(
                    total_memory_required[device] * 1.1 + extra_mem,
                    device,
                    models_already_loaded,
                )

        for loaded_model in models_to_load:
            model = loaded_model.model
            torch_dev = model.load_device
            if is_device_cpu(torch_dev):
                vram_set_state = VRAMState.DISABLED
            else:
                vram_set_state = vram_state
            lowvram_model_memory = 0
            if (
                lowvram_available
                and (
                    vram_set_state == VRAMState.LOW_VRAM
                    or vram_set_state == VRAMState.NORMAL_VRAM
                )
                and not force_full_load
            ):
                model_size = loaded_model.model_memory_required(torch_dev)
                current_free_mem = get_free_memory(torch_dev)
                lowvram_model_memory = max(
                    64 * (1024 * 1024),
                    (current_free_mem - minimum_memory_required),
                    min(
                        current_free_mem * 0.4,
                        current_free_mem - minimum_inference_memory(),
                    ),
                )
                if (
                    model_size <= lowvram_model_memory
                ):  # only switch to lowvram if really necessary
                    lowvram_model_memory = 0

            if vram_set_state == VRAMState.NO_VRAM:
                lowvram_model_memory = 64 * 1024 * 1024

            loaded_model.model_load_flux(
                lowvram_model_memory, force_patch_weights=force_patch_weights
            )
            current_loaded_models.insert(0, loaded_model)

        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_mem = get_free_memory(d)
                if free_mem > minimum_memory_required:
                    use_more_memory(
                        free_mem - minimum_memory_required, models_already_loaded, d
                    )
        return

def load_model_gpu(model: torch.nn.Module, flux_enabled:bool = False) -> None:
    """#### Load a model on the GPU
    
    #### Args:
        - `model` (torch.nn.Module): The model
        - `flux_enable` (bool, optional): Whether flux is enabled. Defaults to False.
    """
    return load_models_gpu([model], flux_enabled=flux_enabled)


def cleanup_models(keep_clone_weights_loaded:bool = False):
    """#### Cleanup the models
    
    #### Args:
        - `keep_clone_weights_loaded` (bool, optional): Whether to keep the clone weights loaded. Defaults to False.
    """
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            elif (
                sys.getrefcount(current_loaded_models[i].real_model) <= 3
            ):  # references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype: torch.dtype) -> int:
    """#### Get the size of a dtype

    #### Args:
        - `dtype` (torch.dtype): The dtype

    #### Returns:
        - `int`: The size of the dtype
    """
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device() -> torch.device:
    """#### Get the offload device for UNet
    
    #### Returns:
        - `torch.device`: The offload device
    """
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype) -> torch.device:
    """#### Get the initial load device for UNet
    
    #### Args:
        - `parameters` (int): The parameters
        - `dtype` (torch.dtype): The dtype
    
    #### Returns:
        - `torch.device`: The initial load device
    """
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_dtype(
    device: torch.dtype = None,
    model_params: int = 0,
    supported_dtypes: list = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """#### Get the dtype for UNet

    #### Args:
        - `device` (torch.dtype, optional): The device. Defaults to None.
        - `model_params` (int, optional): The model parameters. Defaults to 0.
        - `supported_dtypes` (list, optional): The supported dtypes. Defaults to [torch.float16, torch.bfloat16, torch.float32].

    #### Returns:
        - `torch.dtype`: The dtype
    """
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32


# None means no manual cast
def unet_manual_cast(
    weight_dtype: torch.dtype,
    inference_device: torch.device,
    supported_dtypes: list = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """#### Manual cast for UNet

    #### Args:
        - `weight_dtype` (torch.dtype): The dtype of the weights
        - `inference_device` (torch.device): The device used for inference
        - `supported_dtypes` (list, optional): The supported dtypes. Defaults to [torch.float16, torch.bfloat16, torch.float32].

    #### Returns:
        - `torch.dtype`: The dtype
    """
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    elif bf16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32


def text_encoder_offload_device() -> torch.device:
    """#### Get the offload device for the text encoder
    
    #### Returns:
        - `torch.device`: The offload device
    """
    return torch.device("cpu")


def text_encoder_device() -> torch.device:
    """#### Get the device for the text encoder
    
    #### Returns:
        - `torch.device`: The device
    """
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")
    
def text_encoder_initial_device(load_device, offload_device, model_size=0):
    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        return offload_device

    if is_device_mps(load_device):
        return offload_device

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device


def text_encoder_dtype(device: torch.device = None) -> torch.dtype:
    """#### Get the dtype for the text encoder

    #### Args:
        - `device` (torch.device, optional): The device used by the text encoder. Defaults to None.

    Returns:
        torch.dtype: The dtype
    """
    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device() -> torch.device:
    """#### Get the intermediate device
    
    #### Returns:
        - `torch.device`: The intermediate device
    """
    return torch.device("cpu")


def vae_device() -> torch.device:
    """#### Get the VAE device
    
    #### Returns:
        - `torch.device`: The VAE device
    """
    return get_torch_device()


def vae_offload_device() -> torch.device:
    """#### Get the offload device for VAE
    
    #### Returns:
        - `torch.device`: The offload device
    """
    return torch.device("cpu")


def vae_dtype():
    """#### Get the dtype for VAE
    
    #### Returns:
        - `torch.dtype`: The dtype
    """
    global VAE_DTYPE
    return VAE_DTYPE


def get_autocast_device(dev: torch.device) -> str:
    """#### Get the autocast device
    
    #### Args:
        - `dev` (torch.device): The device
    
    #### Returns:
        - `str`: The autocast device type
    """
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """#### Check if the device supports the dtype
    
    #### Args:
        - `device` (torch.device): The device to check
        - `dtype`  (torch.dtype): The dtype to check support
        
    #### Returns:
        - `bool`: Whether the dtype is supported by the device
    """
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def device_supports_non_blocking(device: torch.device) -> bool:
    """#### Check if the device supports non-blocking

    #### Args:
        - `device` (torch.device): The device to check

    #### Returns:
        - `bool`: Whether the device supports non-blocking
    """
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    return True

def supports_cast(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False

def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, copy: bool = False) -> torch.Tensor:
    """#### Cast a tensor to a device

    #### Args:
        - `tensor` (torch.Tensor): The tensor to cast
        - `device` (torch.device): The device to cast the tensor to
        - `dtype` (torch.dtype): The dtype precision to cast to
        - `copy` (bool, optional): Whether to copy the tensor. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The tensor cast to the device
    """
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype

def xformers_enabled() -> bool:
    """#### Check if xformers is enabled
    
    #### Returns:
        - `bool`: Whether xformers is enabled
    """
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae() -> bool:
    """#### Check if xformers is enabled for VAE
    
    #### Returns:
        - `bool`: Whether xformers is enabled for VAE
    """
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled() -> bool:
    """#### Check if PyTorch attention is enabled
    
    #### Returns:
        - `bool`: Whether PyTorch attention is enabled
    """
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention() -> bool:
    """#### Check if PyTorch flash attention is enabled and supported.

    #### Returns:
        - `bool`: True if PyTorch flash attention is enabled and supported, False otherwise.
    """
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
    return False


def get_free_memory(dev: torch.device = None, torch_free_too: bool = False) -> Union[int, Tuple[int, int]]:
    """#### Get the free memory available on the device.

    #### Args:
        - `dev` (torch.device, optional): The device to check memory for. Defaults to None.
        - `torch_free_too` (bool, optional): Whether to return both total and torch free memory. Defaults to False.

    #### Returns:
        - `int` or `Tuple[int, int]`: The free memory available. If `torch_free_too` is True, returns a tuple of total and torch free memory.
    """
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = (
                torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            )
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode() -> bool:
    """#### Check if the current mode is CPU.

    #### Returns:
        - `bool`: True if the current mode is CPU, False otherwise.
    """
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode() -> bool:
    """#### Check if the current mode is MPS.

    #### Returns:
        - `bool`: True if the current mode is MPS, False otherwise.
    """
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device: torch.device, type: str) -> bool:
    """#### Check if the device is of a specific type.

    #### Args:
        - `device` (torch.device): The device to check.
        - `type` (str): The type to check for.

    #### Returns:
        - `bool`: True if the device is of the specified type, False otherwise.
    """
    if hasattr(device, "type"):
        if device.type == type:
            return True
    return False


def is_device_cpu(device: torch.device) -> bool:
    """#### Check if the device is a CPU.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is a CPU, False otherwise.
    """
    return is_device_type(device, "cpu")


def is_device_mps(device: torch.device) -> bool:
    """#### Check if the device is an MPS.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is an MPS, False otherwise.
    """
    return is_device_type(device, "mps")


def is_device_cuda(device: torch.device) -> bool:
    """#### Check if the device is a CUDA device.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is a CUDA device, False otherwise.
    """
    return is_device_type(device, "cuda")


def should_use_fp16(
    device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False
) -> bool:
    """#### Determine if FP16 should be used.

    #### Args:
        - `device` (torch.device, optional): The device to check. Defaults to None.
        - `model_params` (int, optional): The number of model parameters. Defaults to 0.
        - `prioritize_performance` (bool, optional): Whether to prioritize performance. Defaults to True.
        - `manual_cast` (bool, optional): Whether to manually  Defaults to False.

    #### Returns:
        - `bool`: True if FP16 should be used, False otherwise.
    """
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    fp16_works = False
    nvidia_10_series = [
        "1080",
        "1070",
        "titan x",
        "p3000",
        "p3200",
        "p4000",
        "p4200",
        "p5000",
        "p5200",
        "p6000",
        "1060",
        "1050",
        "p40",
        "p100",
        "p6",
        "p4",
    ]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    nvidia_16_series = [
        "1660",
        "1650",
        "1630",
        "T500",
        "T550",
        "T600",
        "MX550",
        "MX450",
        "CMP 30HX",
        "T2000",
        "T1000",
        "T1200",
    ]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(
    device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False
) -> bool:
    """#### Determine if BF16 should be used.

    #### Args:
        - `device` (torch.device, optional): The device to check. Defaults to None.
        - `model_params` (int, optional): The number of model parameters. Defaults to 0.
        - `prioritize_performance` (bool, optional): Whether to prioritize performance. Defaults to True.
        - `manual_cast` (bool, optional): Whether to manually  Defaults to False.

    #### Returns:
        - `bool`: True if BF16 should be used, False otherwise.
    """
    if device is not None:
        if is_device_cpu(device):
            return False

    if device is not None:
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def soft_empty_cache(force: bool = False) -> None:
    """#### Softly empty the cache.

    #### Args:
        - `force` (bool, optional): Whether to force emptying the cache. Defaults to False.
    """
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if (
            force or is_nvidia()
        ):  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models() -> None:
    """#### Unload all models."""
    free_memory(1e30, get_torch_device())


def resolve_lowvram_weight(weight: torch.Tensor, model: torch.nn.Module, key: str) -> torch.Tensor:
    """#### Resolve low VRAM weight.

    #### Args:
        - `weight` (torch.Tensor): The weight tensor.
        - `model` (torch.nn.Module): The model.
        - `key` (str): The key.

    #### Returns:
        - `torch.Tensor`: The resolved weight tensor.
    """
    return weight

import torch
from typing import List, Tuple, Any


def get_models_from_cond(cond: dict, model_type: str) -> List[object]:
    """#### Get models from a condition.

    #### Args:
        - `cond` (dict): The condition.
        - `model_type` (str): The model type.

    #### Returns:
        - `List[object]`: The list of models.
    """
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
    return models


def get_additional_models(conds: dict, dtype: torch.dtype) -> Tuple[List[object], int]:
    """#### Load additional models in conditioning.

    #### Args:
        - `conds` (dict): The conditions.
        - `dtype` (torch.dtype): The data type.

    #### Returns:
        - `Tuple[List[object], int]`: The list of models and the inference memory.
    """
    cnets = []
    gligen = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def prepare_sampling(
    model: object, noise_shape: Tuple[int], conds: dict, flux_enabled: bool = False
) -> Tuple[object, dict, List[object]]:
    """#### Prepare the model for 

    #### Args:
        - `model` (object): The model.
        - `noise_shape` (Tuple[int]): The shape of the noise.
        - `conds` (dict): The conditions.
        - `flux_enabled` (bool, optional): Whether flux is enabled. Defaults to False.

    #### Returns:
        - `Tuple[object, dict, List[object]]`: The prepared model, conditions, and additional models.
    """
    real_model = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    memory_required = (
        model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]))
        + inference_memory
    )
    minimum_memory_required = (
        model.memory_required([noise_shape[0]] + list(noise_shape[1:]))
        + inference_memory
    )
    load_models_gpu(
        [model] + models,
        memory_required=memory_required,
        minimum_memory_required=minimum_memory_required,
        flux_enabled=flux_enabled,
    )
    real_model = model.model

    return real_model, conds, models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, "cleanup"):
            m.cleanup()

def cleanup_models(conds: dict, models: List[object]) -> None:
    """#### Clean up the models after 

    #### Args:
        - `conds` (dict): The conditions.
        - `models` (List[object]): The list of models.
    """
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))


def cond_equal_size(c1: Any, c2: Any) -> bool:
    """#### Check if two conditions have equal size.

    #### Args:
        - `c1` (Any): The first condition.
        - `c2` (Any): The second condition.

    #### Returns:
        - `bool`: Whether the conditions have equal size.
    """
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    return True


def can_concat_cond(c1: Any, c2: Any) -> bool:
    """#### Check if two conditions can be concatenated.

    #### Args:
        - `c1` (Any): The first condition.
        - `c2` (Any): The second condition.

    #### Returns:
        - `bool`: Whether the conditions can be concatenated.
    """
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list: List[dict]) -> dict:
    """#### Concatenate a list of conditions.

    #### Args:
        - `c_list` (List[dict]): The list of conditions.

    #### Returns:
        - `dict`: The concatenated conditions.
    """
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def create_cond_with_same_area_if_none(conds: List[dict], c: dict) -> None:
    """#### Create a condition with the same area if none exists.

    #### Args:
        - `conds` (List[dict]): The list of conditions.
        - `c` (dict): The condition.
    """
    if "area" not in c:
        return

    c_area = c["area"]
    smallest = None
    for x in conds:
        if "area" in x:
            a = x["area"]
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif "area" not in smallest:
                            smallest = x
                        else:
                            if smallest["area"][0] * smallest["area"][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if "area" in smallest:
        if smallest["area"] == c_area:
            return

    out = c.copy()
    out["model_conds"] = smallest[
        "model_conds"
    ].copy()
    conds += [out]


import torch


class CONDRegular:
    """#### Class representing a regular condition."""

    def __init__(self, cond: torch.Tensor):
        """#### Initialize the CONDRegular class.

        #### Args:
            - `cond` (torch.Tensor): The condition tensor.
        """
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDRegular":
        """#### Copy the condition with a new condition.

        #### Args:
            - `cond` (torch.Tensor): The new condition.

        #### Returns:
            - `CONDRegular`: The copied condition.
        """
        return self.__class__(cond)

    def process_cond(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> "CONDRegular":
        """#### Process the condition.

        #### Args:
            - `batch_size` (int): The batch size.
            - `device` (torch.device): The device.

        #### Returns:
            - `CONDRegular`: The processed condition.
        """
        return self._copy_with(
            repeat_to_batch_size(self.cond, batch_size).to(device)
        )
        
    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.
        
        #### Args:
            - `other` (CONDRegular): The other condition.
            
        #### Returns:
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others: list) -> torch.Tensor:
        """#### Concatenate conditions.
        
        #### Args:
            - `others` (list): The list of other conditions.
            
        #### Returns:
            - `torch.Tensor`: The concatenated conditions.
        """
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class CONDCrossAttn(CONDRegular):
    """#### Class representing a cross-attention condition."""

    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.
        
        #### Args:
            - `other` (CONDRegular): The other condition.
            
        #### Returns:   
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                return False

            mult_min = torch.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if (
                diff > 4
            ):  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        return True
    
    def concat(self, others: list) -> torch.Tensor:
        """#### Concatenate cross-attention conditions.

        #### Args:
            - `others` (list): The list of other conditions.

        #### Returns:
            - `torch.Tensor`: The concatenated conditions.
        """
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(
                    1, crossattn_max_len // c.shape[1], 1
                )  # padding with repeat doesn't change result, but avoids an error on tensor shape
            out.append(c)
        return torch.cat(out)


def convert_cond(cond: list) -> list:
    """#### Convert conditions to cross-attention conditions.

    #### Args:
        - `cond` (list): The list of conditions.

    #### Returns:
        - `list`: The converted conditions.
    """
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def calc_cond_batch(
    model: object,
    conds: list,
    x_in: torch.Tensor,
    timestep: torch.Tensor,
    model_options: dict,
) -> list:
    """#### Calculate the condition batch.

    #### Args:
        - `model` (object): The model.
        - `conds` (list): The list of conditions.
        - `x_in` (torch.Tensor): The input tensor.
        - `timestep` (torch.Tensor): The timestep tensor.
        - `model_options` (dict): The model options.

    #### Returns:
        - `list`: The calculated condition batch.
    """
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) * 1.5 < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c["control"] = control.get_control(
                input_x, timestep_, c, len(cond_or_uncond)
            )

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c["transformer_options"] = transformer_options

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            if a is None:
                out_conds[cond_index] += output[o] * mult[o]
                out_counts[cond_index] += mult[o]
            else:
                out_c = out_conds[cond_index]
                out_cts = out_counts[cond_index]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * mult[o]
                out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds


def encode_model_conds(
    model_function: callable,
    conds: list,
    noise: torch.Tensor,
    device: torch.device,
    prompt_type: str,
    **kwargs,
) -> list:
    """#### Encode model conditions.

    #### Args:
        - `model_function` (callable): The model function.
        - `conds` (list): The list of conditions.
        - `noise` (torch.Tensor): The noise tensor.
        - `device` (torch.device): The device.
        - `prompt_type` (str): The prompt type.
        - `**kwargs`: Additional keyword arguments.

    #### Returns:
        - `list`: The encoded model conditions.
    """
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4:  # TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        for k in out:
            model_conds[k] = out[k]
        x["model_conds"] = model_conds
        conds[t] = x
    return conds

def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if "area" in c:
            area = c["area"]
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified["area"] = area
                c = modified
                conditions[i] = c

        if "mask" in c:
            mask = c["mask"]
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=dims, mode="bilinear", align_corners=False
                ).squeeze(1)

            modified["mask"] = mask
            conditions[i] = modified

def process_conds(
    model: object,
    noise: torch.Tensor,
    conds: dict,
    device: torch.device,
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    seed: int = None,
) -> dict:
    """#### Process conditions.

    #### Args:
        - `model` (object): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `conds` (dict): The conditions.
        - `device` (torch.device): The device.
        - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
        - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
        - `seed` (int, optional): The seed. Defaults to None.

    #### Returns:
        - `dict`: The processed conditions.
    """
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(
                model.extra_conds,
                conds[k],
                noise,
                device,
                k,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed,
            )

    # make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(
                    list(
                        filter(
                            lambda c: c.get("control_apply_to_uncond", False) is True,
                            positive,
                        )
                    ),
                    conds[k],
                    "control",
                    lambda cond_cnets, x: cond_cnets[x],
                )
                apply_empty_x_to_equal_area(
                    positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x]
                )

    return conds

import collections
import logging
import numpy as np
import torch


def calculate_start_end_timesteps(model: torch.nn.Module, conds: list) -> None:
    """#### Calculate the start and end timesteps for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `conds` (list): The list of conditions.
    """
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if "start_percent" in x:
            timestep_start = s.percent_to_sigma(x["start_percent"])
        if "end_percent" in x:
            timestep_end = s.percent_to_sigma(x["end_percent"])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if timestep_start is not None:
                n["timestep_start"] = timestep_start
            if timestep_end is not None:
                n["timestep_end"] = timestep_end
            conds[t] = n


def pre_run_control(model: torch.nn.Module, conds: list) -> None:
    """#### Pre-run control for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `conds` (list): The list of conditions.
    """
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        def percent_to_timestep_function(a):
            return s.percent_to_sigma(a)
        if "control" in x:
            x["control"].pre_run(model, percent_to_timestep_function)


def apply_empty_x_to_equal_area(
    conds: list, uncond: list, name: str, uncond_fill_func: callable
) -> None:
    """#### Apply empty x to equal area.

    #### Args:
        - `conds` (list): The list of conditions.
        - `uncond` (list): The list of unconditional conditions.
        - `name` (str): The name.
        - `uncond_fill_func` (callable): The unconditional fill function.
    """
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if "area" not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if "area" not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n


def get_area_and_mult(
    conds: dict, x_in: torch.Tensor, timestep_in: int
) -> collections.namedtuple:
    """#### Get the area and multiplier.

    #### Args:
        - `conds` (dict): The conditions.
        - `x_in` (torch.Tensor): The input tensor.
        - `timestep_in` (int): The timestep.

    #### Returns:
        - `collections.namedtuple`: The area and multiplier.
    """
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
    mask = torch.ones_like(input_x)
    mult = mask * strength

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(
            batch_size=x_in.shape[0], device=x_in.device, area=area
        )

    control = conds.get("control", None)
    patches = None
    cond_obj = collections.namedtuple(
        "cond_obj", ["input_x", "mult", "conditioning", "area", "control", "patches"]
    )
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def normal_scheduler(
    model_sampling: torch.nn.Module, steps: int, sgm: bool = False, floor: bool = False
) -> torch.FloatTensor:
    """#### Create a normal scheduler.

    #### Args:
        - `model_sampling` (torch.nn.Module): The model sampling module.
        - `steps` (int): The number of steps.
        - `sgm` (bool, optional): Whether to use SGM. Defaults to False.
        - `floor` (bool, optional): Whether to floor the values. Defaults to False.

    #### Returns:
        - `torch.FloatTensor`: The scheduler.
    """
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def calculate_sigmas(
    model_sampling: torch.nn.Module, scheduler_name: str, steps: int
) -> torch.Tensor:
    """#### Calculate the sigmas for a model.

    #### Args:
        - `model_sampling` (torch.nn.Module): The model sampling module.
        - `scheduler_name` (str): The scheduler name.
        - `steps` (int): The number of steps.

    #### Returns:
        - `torch.Tensor`: The calculated sigmas.
    """
    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(
            n=steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model_sampling, steps)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model_sampling, steps)
    else:
        logging.error("error invalid scheduler {}".format(scheduler_name))
    return sigmas


def prepare_noise(
    latent_image: torch.Tensor, seed: int, noise_inds: list = None
) -> torch.Tensor:
    """#### Prepare noise for a latent image.

    #### Args:
        - `latent_image` (torch.Tensor): The latent image tensor.
        - `seed` (int): The seed for random noise.
        - `noise_inds` (list, optional): The noise indices. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The prepared noise tensor.
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn(
            [1] + list(latent_image.size())[1:],
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


import torch

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return cast_to(
        weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy
    )

def cast_bias_weight(s: torch.nn.Module, input: torch.Tensor= None, dtype:torch.dtype = None, device:torch.device = None, bias_dtype:torch.dtype = None) -> tuple:
    """#### Cast the bias and weight of a module to match the input tensor.

    #### Args:
        - `s` (torch.nn.Module): The module.
        - `input` (torch.Tensor): The input tensor.

    #### Returns:
        - `tuple`: The cast weight and bias.
    """
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = s.bias_function is not None
        bias = cast_to(
            s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function
        )
        if has_function:
            bias = s.bias_function(bias)

    has_function = s.weight_function is not None
    weight = cast_to(
        s.weight, dtype, device, non_blocking=non_blocking, copy=has_function
    )
    if has_function:
        weight = s.weight_function(weight)
    return weight, bias

class CastWeightBiasOp:
    """#### Class representing a cast weight and bias operation."""

    comfy_cast_weights: bool = False
    weight_function: callable = None
    bias_function: callable = None


class disable_weight_init:
    """#### Class representing a module with disabled weight initialization."""

    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)
    
    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        """#### Conv2d layer with disabled weight initialization."""

        def reset_parameters(self) -> None:
            """#### Reset the parameters of the Conv2d layer."""
            return None

        def forward_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """#### Forward pass with comfy cast weights.

            #### Args:
                - `input` (torch.Tensor): The input tensor.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the Conv2d layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.comfy_cast_weights:
                return self.forward_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)
        
    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None
        
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        
        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        """#### GroupNorm layer with disabled weight initialization."""

        def reset_parameters(self) -> None:
            """#### Reset the parameters of the GroupNorm layer."""
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        """#### LayerNorm layer with disabled weight initialization."""

        def reset_parameters(self) -> None:
            """#### Reset the parameters of the LayerNorm layer."""
            return None

        def forward_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """#### Forward pass with cast weights.

            #### Args:
                - `input` (torch.Tensor): The input tensor.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the LayerNorm layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.comfy_cast_weights:
                return self.forward_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)
    
    @classmethod
    def conv_nd(s, dims: int, *args, **kwargs) -> torch.nn.Conv2d:
        """#### Create a Conv2d layer with the specified dimensions.

        #### Args:
            - `dims` (int): The number of dimensions.
            - `*args`: Variable length argument list.
            - `**kwargs`: Arbitrary keyword arguments.

        #### Returns:
            - `torch.nn.Conv2d`: The Conv2d layer.
        """
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    """#### Class representing a module with manual casting."""

    class Linear(disable_weight_init.Linear):
        """#### Linear layer with manual casting."""

        comfy_cast_weights: bool = True
    
    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        """#### Conv2d layer with manual casting."""

        comfy_cast_weights: bool = True
        
    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        """#### GroupNorm layer with manual casting."""

        comfy_cast_weights: bool = True

    class LayerNorm(disable_weight_init.LayerNorm):
        """#### LayerNorm layer with manual casting."""

        comfy_cast_weights: bool = True
    
    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


try :
    import xformers
except ImportError:
    pass
import torch

BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
except:
    pass

def attention_xformers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None, skip_reshape=False
) -> torch.Tensor:
    """#### Make an attention call using xformers. Fastest attention implementation.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    disabled_xformers = False

    if BROKEN_XFORMERS:
        if b * heads > 65535:
            disabled_xformers = True

    if not disabled_xformers:
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            disabled_xformers = True

    if disabled_xformers:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty(
            [q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device
        )
        mask_out[:, :, : mask.shape[-1]] = mask
        mask = mask_out[:, :, : mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    else:
        out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_pytorch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None, skip_reshape=False
) -> torch.Tensor:
    """#### Make an attention call using PyTorch.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def xformers_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using xformers.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.

    Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


def pytorch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using PyTorch.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out


"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""

# TODO: Check if multiprocessing is possible for this module
from PIL import Image
import numpy as np
from sympy import im
import torch
import torch.nn as nn



def conv(n_in: int, n_out: int, **kwargs) -> disable_weight_init.Conv2d:
    """#### Create a convolutional layer.

    #### Args:
        - `n_in` (int): The number of input channels.
        - `n_out` (int): The number of output channels.

    #### Returns:
        - `torch.nn.Module`: The convolutional layer.
    """
    return disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """#### Class representing a clamping layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the clamping layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The clamped tensor.
        """
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    """#### Class representing a block layer."""

    def __init__(self, n_in: int, n_out: int):
        """#### Initialize the block layer.

        #### Args:
            - `n_in` (int): The number of input channels.
            - `n_out` (int): The number of output channels.

        #### Returns:
            - `Block`: The block layer.
        """
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            disable_weight_init.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder2(latent_channels: int = 4) -> nn.Sequential:
    """#### Create an encoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The encoder.
    """
    return nn.Sequential(
        conv(3, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, latent_channels),
    )


def Decoder2(latent_channels: int = 4) -> nn.Sequential:
    """#### Create a decoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The decoder.
    """
    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 64),
        nn.ReLU(),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, 3),
    )


class TAESD(nn.Module):
    """#### Class representing a Tiny AutoEncoder for Stable Diffusion.

    #### Attributes:
        - `latent_magnitude` (float): Magnitude of the latent space.
        - `latent_shift` (float): Shift value for the latent space.
        - `vae_shift` (torch.nn.Parameter): Shift parameter for the VAE.
        - `vae_scale` (torch.nn.Parameter): Scale parameter for the VAE.
        - `taesd_encoder` (Encoder2): Encoder network for the TAESD.
        - `taesd_decoder` (Decoder2): Decoder network for the TAESD.

    #### Args:
        - `encoder_path` (str, optional): Path to the encoder model file. Defaults to None.
        - `decoder_path` (str, optional): Path to the decoder model file. Defaults to "./_internal/vae_approx/taesd_decoder.safetensors".
        - `latent_channels` (int, optional): Number of channels in the latent space. Defaults to 4.

    #### Methods:
        - `scale_latents(x)`:
            Scales raw latents to the range [0, 1].
        - `unscale_latents(x)`:
            Unscales latents from the range [0, 1] to raw latents.
        - `decode(x)`:
            Decodes the given latent representation to the original space.
        - `encode(x)`:
            Encodes the given input to the latent space.
    """

    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(
        self,
        encoder_path: str = None,
        decoder_path: str = None,
        latent_channels: int = 4,
    ):
        """#### Initialize the TAESD model.

        #### Args:
            - `encoder_path` (str, optional): Path to the encoder model file. Defaults to None.
            - `decoder_path` (str, optional): Path to the decoder model file. Defaults to "./_internal/vae_approx/taesd_decoder.safetensors".
            - `latent_channels` (int, optional): Number of channels in the latent space. Defaults to 4.
        """
        super().__init__()
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.taesd_encoder = Encoder2(latent_channels)
        self.taesd_decoder = Decoder2(latent_channels)
        decoder_path = (
            "./_internal/vae_approx/taesd_decoder.safetensors"
            if decoder_path is None
            else decoder_path
        )
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(
                load_torch_file(encoder_path, safe_load=True)
            )
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(
                load_torch_file(decoder_path, safe_load=True)
            )

    @staticmethod
    def scale_latents(x: torch.Tensor) -> torch.Tensor:
        """#### Scales raw latents to the range [0, 1].

        #### Args:
            - `x` (torch.Tensor): The raw latents.

        #### Returns:
            - `torch.Tensor`: The scaled latents.
        """
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x: torch.Tensor) -> torch.Tensor:
        """#### Unscales latents from the range [0, 1] to raw latents.

        #### Args:
            - `x` (torch.Tensor): The scaled latents.

        #### Returns:
            - `torch.Tensor`: The raw latents.
        """
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """#### Decodes the given latent representation to the original space.

        #### Args:
            - `x` (torch.Tensor): The latent representation.

        #### Returns:
            - `torch.Tensor`: The decoded representation.
        """
        device = next(self.taesd_decoder.parameters()).device
        x = x.to(device)
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """#### Encodes the given input to the latent space.

        #### Args:
            - `x` (torch.Tensor): The input.

        #### Returns:
            - `torch.Tensor`: The latent representation.
        """
        device = next(self.taesd_encoder.parameters()).device
        x = x.to(device)
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift


def taesd_preview(x: torch.Tensor, flux: bool = False):
    """Preview the batched latent tensors as images.
    
    Args:
        x (torch.Tensor): Input latent tensor with shape [B,C,H,W] 
        flux (bool): Whether using flux model (for channel ordering)
    """
    if app_instance.app.previewer_var.get() is True:
        taesd_instance = TAESD()
        
        # Handle channel dimension
        if x.shape[1] != 4:
            desired_channels = 4
            current_channels = x.shape[1]
            
            if current_channels > desired_channels:
                x = x[:, :desired_channels, :, :]
            else:
                padding = torch.zeros(x.shape[0], desired_channels - current_channels, 
                                   x.shape[2], x.shape[3], device=x.device)
                x = torch.cat([x, padding], dim=1)

        # Process entire batch at once
        decoded_batch = taesd_instance.decode(x)
        
        images = []
        
        # Convert each image in batch 
        for decoded in decoded_batch:
            # Handle channel dimension
            if decoded.shape[0] == 1:
                decoded = decoded.repeat(3, 1, 1)
                
            # Apply different normalization for flux vs standard mode
            if flux:
                # For flux: Assume BGR ordering and different normalization
                decoded = decoded[[2,1,0], :, :] # BGR -> RGB
                # Adjust normalization for flux model range
                decoded = decoded.clamp(-1, 1)
                decoded = (decoded + 1.0) * 0.5 # Scale from [-1,1] to [0,1]
            else:
                # Standard normalization
                decoded = (decoded + 1.0) / 2.0
            
            # Convert to numpy and uint8
            image_np = (decoded.cpu().detach().numpy() * 255.0)
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
            # Create PIL Image
            img = Image.fromarray(image_np, mode='RGB')
            images.append(img)
            
        # Update display with all images
        app_instance.app.update_image(images)
    else:
        pass


import torch


class CONDRegular:
    """#### Class representing a regular condition."""

    def __init__(self, cond: torch.Tensor):
        """#### Initialize the CONDRegular class.

        #### Args:
            - `cond` (torch.Tensor): The condition tensor.
        """
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDRegular":
        """#### Copy the condition with a new condition.

        #### Args:
            - `cond` (torch.Tensor): The new condition.

        #### Returns:
            - `CONDRegular`: The copied condition.
        """
        return self.__class__(cond)

    def process_cond(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> "CONDRegular":
        """#### Process the condition.

        #### Args:
            - `batch_size` (int): The batch size.
            - `device` (torch.device): The device.

        #### Returns:
            - `CONDRegular`: The processed condition.
        """
        return self._copy_with(
            repeat_to_batch_size(self.cond, batch_size).to(device)
        )
        
    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.
        
        #### Args:
            - `other` (CONDRegular): The other condition.
            
        #### Returns:
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others: list) -> torch.Tensor:
        """#### Concatenate conditions.
        
        #### Args:
            - `others` (list): The list of other conditions.
            
        #### Returns:
            - `torch.Tensor`: The concatenated conditions.
        """
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class CONDCrossAttn(CONDRegular):
    """#### Class representing a cross-attention condition."""

    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.
        
        #### Args:
            - `other` (CONDRegular): The other condition.
            
        #### Returns:   
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                return False

            mult_min = torch.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if (
                diff > 4
            ):  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        return True
    
    def concat(self, others: list) -> torch.Tensor:
        """#### Concatenate cross-attention conditions.

        #### Args:
            - `others` (list): The list of other conditions.

        #### Returns:
            - `torch.Tensor`: The concatenated conditions.
        """
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(
                    1, crossattn_max_len // c.shape[1], 1
                )  # padding with repeat doesn't change result, but avoids an error on tensor shape
            out.append(c)
        return torch.cat(out)


def convert_cond(cond: list) -> list:
    """#### Convert conditions to cross-attention conditions.

    #### Args:
        - `cond` (list): The list of conditions.

    #### Returns:
        - `list`: The converted conditions.
    """
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def calc_cond_batch(
    model: object,
    conds: list,
    x_in: torch.Tensor,
    timestep: torch.Tensor,
    model_options: dict,
) -> list:
    """#### Calculate the condition batch.

    #### Args:
        - `model` (object): The model.
        - `conds` (list): The list of conditions.
        - `x_in` (torch.Tensor): The input tensor.
        - `timestep` (torch.Tensor): The timestep tensor.
        - `model_options` (dict): The model options.

    #### Returns:
        - `list`: The calculated condition batch.
    """
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) * 1.5 < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c["control"] = control.get_control(
                input_x, timestep_, c, len(cond_or_uncond)
            )

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c["transformer_options"] = transformer_options

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            if a is None:
                out_conds[cond_index] += output[o] * mult[o]
                out_counts[cond_index] += mult[o]
            else:
                out_c = out_conds[cond_index]
                out_cts = out_counts[cond_index]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * mult[o]
                out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds


def encode_model_conds(
    model_function: callable,
    conds: list,
    noise: torch.Tensor,
    device: torch.device,
    prompt_type: str,
    **kwargs,
) -> list:
    """#### Encode model conditions.

    #### Args:
        - `model_function` (callable): The model function.
        - `conds` (list): The list of conditions.
        - `noise` (torch.Tensor): The noise tensor.
        - `device` (torch.device): The device.
        - `prompt_type` (str): The prompt type.
        - `**kwargs`: Additional keyword arguments.

    #### Returns:
        - `list`: The encoded model conditions.
    """
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4:  # TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        for k in out:
            model_conds[k] = out[k]
        x["model_conds"] = model_conds
        conds[t] = x
    return conds

def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if "area" in c:
            area = c["area"]
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified["area"] = area
                c = modified
                conditions[i] = c

        if "mask" in c:
            mask = c["mask"]
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1), size=dims, mode="bilinear", align_corners=False
                ).squeeze(1)

            modified["mask"] = mask
            conditions[i] = modified

def process_conds(
    model: object,
    noise: torch.Tensor,
    conds: dict,
    device: torch.device,
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    seed: int = None,
) -> dict:
    """#### Process conditions.

    #### Args:
        - `model` (object): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `conds` (dict): The conditions.
        - `device` (torch.device): The device.
        - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
        - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
        - `seed` (int, optional): The seed. Defaults to None.

    #### Returns:
        - `dict`: The processed conditions.
    """
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(
                model.extra_conds,
                conds[k],
                noise,
                device,
                k,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed,
            )

    # make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(
                    list(
                        filter(
                            lambda c: c.get("control_apply_to_uncond", False) is True,
                            positive,
                        )
                    ),
                    conds[k],
                    "control",
                    lambda cond_cnets, x: cond_cnets[x],
                )
                apply_empty_x_to_equal_area(
                    positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x]
                )

    return conds

import torch
import torch.nn as nn


class GEGLU(nn.Module):
    """#### Class representing the GEGLU activation function.

    GEGLU is a gated activation function that is a combination of GELU and ReLU,
    used to fire the neurons in the network.

    #### Args:
        - `dim_in` (int): The input dimension.
        - `dim_out` (int): The output dimension.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = manual_cast.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the GEGLU activation function.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


import torch
import torch.nn as nn
import logging



def Normalize(
    in_channels: int, dtype: torch.dtype = None, device: torch.device = None
) -> torch.nn.GroupNorm:
    """#### Normalize the input channels.

    #### Args:
        - `in_channels` (int): The input channels.
        - `dtype` (torch.dtype, optional): The data type. Defaults to `None`.
        - `device` (torch.device, optional): The device. Defaults to `None`.

    #### Returns:
        - `torch.nn.GroupNorm`: The normalized input channels
    """
    return torch.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


if xformers_enabled():
    logging.info("Using xformers cross attention")
    optimized_attention = attention_xformers
else:
    logging.info("Using pytorch cross attention")
    optimized_attention = attention_pytorch

optimized_attention_masked = optimized_attention


def optimized_attention_for_device() -> attention_pytorch:
    """#### Get the optimized attention for a device.

    #### Returns:
        - `function`: The optimized attention function.
    """
    return attention_pytorch


class CrossAttention(nn.Module):
    """#### Cross attention module, which applies attention across the query and context.

    #### Args:
        - `query_dim` (int): The query dimension.
        - `context_dim` (int, optional): The context dimension. Defaults to `None`.
        - `heads` (int, optional): The number of heads. Defaults to `8`.
        - `dim_head` (int, optional): The head dimension. Defaults to `64`.
        - `dropout` (float, optional): The dropout rate. Defaults to `0.0`.
        - `dtype` (torch.dtype, optional): The data type. Defaults to `None`.
        - `device` (torch.device, optional): The device. Defaults to `None`.
        - `operations` (disable_weight_init, optional): The operations. Defaults to `disable_weight_init`.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: disable_weight_init = disable_weight_init,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """#### Forward pass of the cross attention module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `context` (torch.Tensor, optional): The context tensor. Defaults to `None`.
            - `value` (torch.Tensor, optional): The value tensor. Defaults to `None`.
            - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = optimized_attention(q, k, v, self.heads)
        return self.to_out(out)


class AttnBlock(nn.Module):
    """#### Attention block, which applies attention to the input tensor.

    #### Args:
        - `in_channels` (int): The input channels.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        if xformers_enabled_vae():
            logging.info("Using xformers attention in VAE")
            self.optimized_attention = xformers_attention
        else:
            logging.info("Using pytorch attention in VAE")
            self.optimized_attention = pytorch_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the attention block.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels: int, attn_type: str = "vanilla") -> AttnBlock:
    """#### Make an attention block.

    #### Args:
        - `in_channels` (int): The input channels.
        - `attn_type` (str, optional): The attention type. Defaults to "vanilla".

    #### Returns:
        - `AttnBlock`: A class instance of the attention block.
    """
    return AttnBlock(in_channels)


import threading
import torch
from tqdm.auto import trange, tqdm



disable_gui = False


@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pipeline=False,
):
    """#### Perform ancestral sampling using the Euler method.

    #### Args:
        - `model` (torch.nn.Module): The model to use for denoising.
        - `x` (torch.Tensor): The input tensor to be denoised.
        - `sigmas` (list or torch.Tensor): A list or tensor of sigma values for the noise schedule.
        - `extra_args` (dict, optional): Additional arguments to pass to the model. Defaults to None.
        - `callback` (callable, optional): A callback function to be called at each iteration. Defaults to None.
        - `disable` (bool, optional): If True, disables the progress bar. Defaults to None.
        - `eta` (float, optional): The eta parameter for the ancestral step. Defaults to 1.0.
        - `s_noise` (float, optional): The noise scaling factor. Defaults to 1.0.
        - `noise_sampler` (callable, optional): A function to sample noise. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The denoised tensor after ancestral 
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
                
        # Rest of sampling code remains the same
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            
    return x


@torch.no_grad()
def sample_dpm_adaptive(
    model,
    x,
    sigma_min,
    sigma_max,
    extra_args=None,
    callback=None,
    disable=None,
    order=3,
    rtol=0.05,
    atol=0.0078,
    h_init=0.05,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    accept_safety=0.81,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
    return_info=False,
    pipeline=False,
):
    """
    #### Samples from a diffusion probabilistic model using an adaptive step size solver.

    This function implements the DPM-Solver-12 and DPM-Solver-23 methods with adaptive step size as described in the paper
    https://arxiv.org/abs/2206.00927.

    #### Args:
        - `model` (torch.nn.Module): The diffusion model to sample from.
        - `x` (torch.Tensor): The initial tensor to start sampling from.
        - `sigma_min` (float): The minimum sigma value for the sampling process.
        - `sigma_max` (float): The maximum sigma value for the sampling process.
        - `extra_args` (dict, optional): Additional arguments to pass to the model. Default is None.
        - `callback` (callable, optional): A callback function to be called with progress information. Default is None.
        - `disable` (bool, optional): If True, disables the progress bar. Default is None.
        - `order` (int, optional): The order of the solver. Default is 3.
        - `rtol` (float, optional): Relative tolerance for adaptive step size. Default is 0.05.
        - `atol` (float, optional): Absolute tolerance for adaptive step size. Default is 0.0078.
        - `h_init` (float, optional): Initial step size. Default is 0.05.
        - `pcoeff` (float, optional): Coefficient for the predictor step. Default is 0.0.
        - `icoeff` (float, optional): Coefficient for the corrector step. Default is 1.0.
        - `dcoeff` (float, optional): Coefficient for the diffusion step. Default is 0.0.
        - `accept_safety` (float, optional): Safety factor for step acceptance. Default is 0.81.
        - `eta` (float, optional): Noise scale for the sampling process. Default is 0.0.
        - `s_noise` (float, optional): Scale of the noise to be added. Default is 1.0.
        - `noise_sampler` (callable, optional): A function to sample noise. Default is None.
        - `return_info` (bool, optional): If True, returns additional information about the sampling process. Default is False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
        - `dict` (optional): Additional information about the sampling process if `return_info` is True.

    #### Raises:
        - `ValueError`: If sigma_min or sigma_max is less than or equal to 0.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(
            model, extra_args, eps_callback=pbar.update
        )
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        with torch.amp.autocast(device_type="cuda"):
            x, info = dpm_solver.dpm_solver_adaptive(
                x,
                dpm_solver.t(sigma_max.clone().detach()),
                dpm_solver.t(sigma_min.clone().detach()),
                order,
                rtol,
                atol,
                h_init,
                pcoeff,
                icoeff,
                dcoeff,
                accept_safety,
                eta,
                s_noise,
                noise_sampler,
                pipeline,
            )
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    pipeline=False,
):
    """
    #### Samples from a model using the DPM-Solver++(2M) SDE method.

    #### Args:
        - `model` (torch.nn.Module): The model to sample from.
        - `x` (torch.Tensor): The initial input tensor.
        - `sigmas` (torch.Tensor): A tensor of sigma values for the SDE.
        - `extra_args` (dict, optional): Additional arguments for the model. Default is None.
        - `callback` (callable, optional): A callback function to be called at each step. Default is None.
        - `disable` (bool, optional): If True, disables the progress bar. Default is None.
        - `eta` (float, optional): The eta parameter for the SDE. Default is 1.0.
        - `s_noise` (float, optional): The noise scale parameter. Default is 1.0.
        - `noise_sampler` (callable, optional): A noise sampler function. Default is None.
        - `solver_type` (str, optional): The type of solver to use ('midpoint' or 'heun'). Default is "midpoint".
        - `pipeline` (bool, optional): If True, disables the progress bar. Default is False.

    #### Returns:
        - `torch.Tensor`: The final sampled tensor.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=True
        )
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None
    h = None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            x = (
                sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x
                + (-h - eta_h).expm1().neg() * denoised
            )
            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                        1 / r
                    ) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                        denoised - old_denoised
                    )
            if eta:
                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * s_noise
                )
        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    pipeline=False,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    global disable_gui
    disable_gui = True if pipeline is True else False
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        if s_churn > 0:
            gamma = (
                min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                if s_tmin <= sigmas[i] <= s_tmax
                else 0.0
            )
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x

import math
import torch


def cfg_function(
    model: torch.nn.Module,
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    cond_scale: float,
    x: torch.Tensor,
    timestep: int,
    model_options: dict = {},
    cond: torch.Tensor = None,
    uncond: torch.Tensor = None,
) -> torch.Tensor:
    """#### Apply classifier-free guidance (CFG) to the model predictions.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `cond_pred` (torch.Tensor): The conditioned prediction.
        - `uncond_pred` (torch.Tensor): The unconditioned prediction.
        - `cond_scale` (float): The CFG scale.
        - `x` (torch.Tensor): The input tensor.
        - `timestep` (int): The current timestep.
        - `model_options` (dict, optional): Additional model options. Defaults to {}.
        - `cond` (torch.Tensor, optional): The conditioned tensor. Defaults to None.
        - `uncond` (torch.Tensor, optional): The unconditioned tensor. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The CFG result.
    """
    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options,
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "model": model,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x,
        }
        cfg_result = fn(args)

    return cfg_result


def sampling_function(
    model: torch.nn.Module,
    x: torch.Tensor,
    timestep: int,
    uncond: torch.Tensor,
    condo: torch.Tensor,
    cond_scale: float,
    model_options: dict = {},
    seed: int = None,
) -> torch.Tensor:
    """#### Perform sampling with 

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `x` (torch.Tensor): The input tensor.
        - `timestep` (int): The current timestep.
        - `uncond` (torch.Tensor): The unconditioned tensor.
        - `condo` (torch.Tensor): The conditioned tensor.
        - `cond_scale` (float): The CFG scale.
        - `model_options` (dict, optional): Additional model options. Defaults to {}.
        - `seed` (int, optional): The random seed. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    if (
        math.isclose(cond_scale, 1.0)
        and model_options.get("disable_cfg1_optimization", False) is False
    ):
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [condo, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {
            "conds": conds,
            "conds_out": out,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "model": model,
            "model_options": model_options,
        }
        out = fn(args)

    return cfg_function(
        model,
        out[0],
        out[1],
        cond_scale,
        x,
        timestep,
        model_options=model_options,
        cond=condo,
        uncond=uncond_,
    )


class CFGGuider:
    """#### Class for guiding the sampling process with """
    def __init__(self, model_patcher, flux=False):
        """#### Initialize the CFGGuider.

        #### Args:
            - `model_patcher` (object): The model patcher.
        """
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0
        self.flux = flux

    def set_conds(self, positive, negative):
        """#### Set the conditions for 

        #### Args:
            - `positive` (torch.Tensor): The positive condition.
            - `negative` (torch.Tensor): The negative condition.
        """
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        """#### Set the CFG scale.

        #### Args:
            - `cfg` (float): The CFG scale.
        """
        self.cfg = cfg

    def inner_set_conds(self, conds):
        """#### Set the internal conditions.

        #### Args:
            - `conds` (dict): The conditions.
        """
        for k in conds:
            self.original_conds[k] = convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        """#### Call the CFGGuider to predict noise.

        #### Returns:
            - `torch.Tensor`: The predicted noise.
        """
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """#### Predict noise using 

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `timestep` (int): The current timestep.
            - `model_options` (dict, optional): Additional model options. Defaults to {}.
            - `seed` (int, optional): The random seed. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The predicted noise.
        """
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        pipeline=False,
    ):
        """#### Perform the inner sampling process.

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `device` (torch.device): The device to use.
            - `sampler` (object): The sampler object.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `denoise_mask` (torch.Tensor): The denoise mask tensor.
            - `callback` (callable): The callback function.
            - `disable_pbar` (bool): Whether to disable the progress bar.
            - `seed` (int): The random seed.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(
            self.inner_model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
            pipeline=pipeline,
        )
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        pipeline=False,
    ):
        """#### Perform the sampling process with 

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `sampler` (object): The sampler object.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            - `callback` (callable, optional): The callback function. Defaults to None.
            - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
            - `seed` (int, optional): The random seed. Defaults to None.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = prepare_sampling(
            self.model_patcher, noise.shape, self.conds, flux_enabled=self.flux
        )
        device = self.model_patcher.load_device

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        output = self.inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            pipeline=pipeline,
        )

        cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output

from einops import rearrange
import torch
import torch.nn as nn

if xformers_enabled():
    pass

ops = disable_weight_init

_ATTN_PRECISION = "fp32"


class FeedForward(nn.Module):
    """#### FeedForward neural network module.

    #### Args:
        - `dim` (int): The input dimension.
        - `dim_out` (int, optional): The output dimension. Defaults to None.
        - `mult` (int, optional): The multiplier for the inner dimension. Defaults to 4.
        - `glu` (bool, optional): Whether to use Gated Linear Units. Defaults to False.
        - `dropout` (float, optional): The dropout rate. Defaults to 0.0.
        - `dtype` (torch.dtype, optional): The data type. Defaults to None.
        - `device` (torch.device, optional): The device. Defaults to None.
        - `operations` (object, optional): The operations module. Defaults to `ops`.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: object = ops,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU()
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the FeedForward network.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """#### Basic Transformer block.

    #### Args:
        - `dim` (int): The input dimension.
        - `n_heads` (int): The number of attention heads.
        - `d_head` (int): The dimension of each attention head.
        - `dropout` (float, optional): The dropout rate. Defaults to 0.0.
        - `context_dim` (int, optional): The context dimension. Defaults to None.
        - `gated_ff` (bool, optional): Whether to use Gated FeedForward. Defaults to True.
        - `checkpoint` (bool, optional): Whether to use checkpointing. Defaults to True.
        - `ff_in` (bool, optional): Whether to use FeedForward input. Defaults to False.
        - `inner_dim` (int, optional): The inner dimension. Defaults to None.
        - `disable_self_attn` (bool, optional): Whether to disable self-attention. Defaults to False.
        - `disable_temporal_crossattention` (bool, optional): Whether to disable temporal cross-attention. Defaults to False.
        - `switch_temporal_ca_to_sa` (bool, optional): Whether to switch temporal cross-attention to self-attention. Defaults to False.
        - `dtype` (torch.dtype, optional): The data type. Defaults to None.
        - `device` (torch.device, optional): The device. Defaults to None.
        - `operations` (object, optional): The operations module. Defaults to `ops`.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        ff_in: bool = False,
        inner_dim: int = None,
        disable_self_attn: bool = False,
        disable_temporal_crossattention: bool = False,
        switch_temporal_ca_to_sa: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: object = ops,
    ):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(
            inner_dim,
            dim_out=dim,
            dropout=dropout,
            glu=gated_ff,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        context_dim_attn2 = None
        if not switch_temporal_ca_to_sa:
            context_dim_attn2 = context_dim

        self.attn2 = CrossAttention(
            query_dim=inner_dim,
            context_dim=context_dim_attn2,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is self-attn if context is none
        self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        """#### Forward pass of the Basic Transformer block.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `context` (torch.Tensor, optional): The context tensor. Defaults to None.
            - `transformer_options` (dict, optional): Additional transformer options. Defaults to {}.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return checkpoint(
            self._forward,
            (x, context, transformer_options),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        """#### Internal forward pass of the Basic Transformer block.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `context` (torch.Tensor, optional): The context tensor. Defaults to None.
            - `transformer_options` (dict, optional): Additional transformer options. Defaults to {}.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches_replace = {}

        for k in transformer_options:
            extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        n = self.norm1(x)
        context_attn1 = None
        value_attn1 = None

        transformer_block = (block[0], block[1], block_index)
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        n = self.attn1(n, context=context_attn1, value=value_attn1)

        x += n

        if self.attn2 is not None:
            n = self.norm2(x)
            context_attn2 = context
            value_attn2 = None

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block
            n = self.attn2(n, context=context_attn2, value=value_attn2)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    """#### Spatial Transformer module.

    #### Args:
        - `in_channels` (int): The number of input channels.
        - `n_heads` (int): The number of attention heads.
        - `d_head` (int): The dimension of each attention head.
        - `depth` (int, optional): The depth of the  Defaults to 1.
        - `dropout` (float, optional): The dropout rate. Defaults to 0.0.
        - `context_dim` (int, optional): The context dimension. Defaults to None.
        - `disable_self_attn` (bool, optional): Whether to disable self-attention. Defaults to False.
        - `use_linear` (bool, optional): Whether to use linear projections. Defaults to False.
        - `use_checkpoint` (bool, optional): Whether to use checkpointing. Defaults to True.
        - `dtype` (torch.dtype, optional): The data type. Defaults to None.
        - `device` (torch.device, optional): The device. Defaults to None.
        - `operations` (object, optional): The operations module. Defaults to `ops`.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int = None,
        disable_self_attn: bool = False,
        use_linear: bool = False,
        use_checkpoint: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: object = ops,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        if not use_linear:
            self.proj_in = operations.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )
        self.use_linear = use_linear

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        """#### Forward pass of the Spatial Transformer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `context` (torch.Tensor, optional): The context tensor. Defaults to None.
            - `transformer_options` (dict, optional): Additional transformer options. Defaults to {}.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def count_blocks(state_dict_keys: list, prefix_string: str) -> int:
    """#### Count the number of blocks in a state dictionary.

    #### Args:
        - `state_dict_keys` (list): The list of state dictionary keys.
        - `prefix_string` (str): The prefix string to match.

    #### Returns:
        - `int`: The number of blocks.
    """
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c is False:
            break
        count += 1
    return count


def calculate_transformer_depth(
    prefix: str, state_dict_keys: list, state_dict: dict
) -> tuple:
    """#### Calculate the depth of a 

    #### Args:
        - `prefix` (str): The prefix string.
        - `state_dict_keys` (list): The list of state dictionary keys.
        - `state_dict` (dict): The state dictionary.

    #### Returns:
        - `tuple`: The transformer depth, context dimension, use of linear in transformer, and time stack.
    """
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(
        list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
    )
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(
            state_dict_keys, transformer_prefix + "{}"
        )
        context_dim = state_dict[
            "{}0.attn2.to_k.weight".format(transformer_prefix)
        ].shape[1]
        use_linear_in_transformer = (
            len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
        )
        time_stack = (
            "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict
            or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
        )
        return (
            last_transformer_depth,
            context_dim,
            use_linear_in_transformer,
            time_stack,
        )
    return None


from enum import Enum
import threading
import torch.nn as nn

import math
import torch



class TimestepBlock1(nn.Module):
    """#### A block for timestep embedding."""
    pass


class TimestepEmbedSequential1(nn.Sequential, TimestepBlock1):
    """#### A sequential block for timestep embedding."""
    pass


class EPS:
    """#### Class for EPS calculations."""

    def calculate_input(self, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """#### Calculate the input for EPS.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `noise` (torch.Tensor): The noise tensor.

        #### Returns:
            - `torch.Tensor`: The calculated input tensor.
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        """#### Calculate the denoised tensor.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `model_output` (torch.Tensor): The model output tensor.
            - `model_input` (torch.Tensor): The model input tensor.

        #### Returns:
            - `torch.Tensor`: The denoised tensor.
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor, max_denoise: bool = False) -> torch.Tensor:
        """#### Scale the noise.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `max_denoise` (bool, optional): Whether to apply maximum denoising. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The scaled noise tensor.
        """
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """#### Inverse the noise scaling.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The inversely scaled noise tensor.
        """
        return latent


class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)


class ModelSamplingDiscrete(torch.nn.Module):
    """#### Class for discrete model """

    def __init__(self, model_config: dict = None):
        """#### Initialize the ModelSamplingDiscrete class.

        #### Args:
            - `model_config` (dict, optional): The model configuration. Defaults to None.
        """
        super().__init__()
        sampling_settings = model_config.sampling_settings
        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        self._register_schedule(
            given_betas=None,
            beta_schedule=beta_schedule,
            timesteps=1000,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
        )
        self.sigma_data = 1.0

    def _register_schedule(
        self,
        given_betas: torch.Tensor = None,
        beta_schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ):
        """#### Register the schedule for the model.

        #### Args:
            - `given_betas` (torch.Tensor, optional): The given betas. Defaults to None.
            - `beta_schedule` (str, optional): The beta schedule. Defaults to "linear".
            - `timesteps` (int, optional): The number of timesteps. Defaults to 1000.
            - `linear_start` (float, optional): The linear start value. Defaults to 1e-4.
            - `linear_end` (float, optional): The linear end value. Defaults to 2e-2.
            - `cosine_s` (float, optional): The cosine s value. Defaults to 8e-3.
        """
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas: torch.Tensor):
        """#### Set the sigmas for the model.

        #### Args:
            - `sigmas` (torch.Tensor): The sigmas tensor.
        """
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self) -> torch.Tensor:
        """#### Get the minimum sigma value.

        #### Returns:
            - `torch.Tensor`: The minimum sigma value.
        """
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        """#### Get the maximum sigma value.

        #### Returns:
            - `torch.Tensor`: The maximum sigma value.
        """
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """#### Convert sigma to timestep.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.

        #### Returns:
            - `torch.Tensor`: The timestep value.
        """
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """#### Convert timestep to sigma.

        #### Args:
            - `timestep` (torch.Tensor): The timestep value.

        #### Returns:
            - `torch.Tensor`: The sigma value.
        """
        t = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)
    
    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class InterruptProcessingException(Exception):
    """#### Exception class for interrupting processing."""
    pass


interrupt_processing_mutex = threading.RLock()

interrupt_processing = False


class KSamplerX0Inpaint:
    """#### Class for KSampler X0 Inpainting."""

    def __init__(self, model: torch.nn.Module, sigmas: torch.Tensor):
        """#### Initialize the KSamplerX0Inpaint class.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `sigmas` (torch.Tensor): The sigmas tensor.
        """
        self.inner_model = model
        self.sigmas = sigmas

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor, denoise_mask: torch.Tensor, model_options: dict = {}, seed: int = None) -> torch.Tensor:
        """#### Call the KSamplerX0Inpaint class.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `sigma` (torch.Tensor): The sigma value.
            - `denoise_mask` (torch.Tensor): The denoise mask tensor.
            - `model_options` (dict, optional): The model options. Defaults to {}.
            - `seed` (int, optional): The seed value. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        return out


class Sampler:
    """#### Class for """

    def max_denoise(self, model_wrap: torch.nn.Module, sigmas: torch.Tensor) -> bool:
        """#### Check if maximum denoising is required.

        #### Args:
            - `model_wrap` (torch.nn.Module): The model wrapper.
            - `sigmas` (torch.Tensor): The sigmas tensor.

        #### Returns:
            - `bool`: Whether maximum denoising is required.
        """
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


class KSAMPLER(Sampler):
    """#### Class for KSAMPLER."""

    def __init__(self, sampler_function: callable, extra_options: dict = {}, inpaint_options: dict = {}):
        """#### Initialize the KSAMPLER class.

        #### Args:
            - `sampler_function` (callable): The sampler function.
            - `extra_options` (dict, optional): The extra options. Defaults to {}.
            - `inpaint_options` (dict, optional): The inpaint options. Defaults to {}.
        """
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(
        self,
        model_wrap: torch.nn.Module,
        sigmas: torch.Tensor,
        extra_args: dict,
        callback: callable,
        noise: torch.Tensor,
        latent_image: torch.Tensor = None,
        denoise_mask: torch.Tensor = None,
        disable_pbar: bool = False,
        pipeline: bool = False,
    ) -> torch.Tensor:
        """#### Sample using the KSAMPLER.

        #### Args:
            - `model_wrap` (torch.nn.Module): The model wrapper.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `extra_args` (dict): The extra arguments.
            - `callback` (callable): The callback function.
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
            - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            pipeline=pipeline,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )
        return samples


def ksampler(sampler_name: str, pipeline: bool = False, extra_options: dict = {}, inpaint_options: dict = {}) -> KSAMPLER:
    """#### Get a KSAMPLER.

    #### Args:
        - `sampler_name` (str): The sampler name.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.
        - `extra_options` (dict, optional): The extra options. Defaults to {}.
        - `inpaint_options` (dict, optional): The inpaint options. Defaults to {}.

    #### Returns:
        - `KSAMPLER`: The KSAMPLER object.
    """
    if sampler_name == "dpm_adaptive":

        def dpm_adaptive_function(
            model: torch.nn.Module,
            noise: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: dict,
            callback: callable,
            disable: bool,
            pipeline: bool,
            **extra_options,
        ) -> torch.Tensor:
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpm_adaptive(
                model,
                noise,
                sigma_min,
                sigmas[0],
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                pipeline=pipeline,
                **extra_options,
            )

        sampler_function = dpm_adaptive_function
    elif sampler_name == "dpmpp_2m_sde":

        def dpmpp_sde_function(
            model: torch.nn.Module,
            noise: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: dict,
            callback: callable,
            disable: bool,
            pipeline: bool,
            **extra_options,
        ) -> torch.Tensor:
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpmpp_2m_sde(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                pipeline=pipeline,
                **extra_options,
            )

        sampler_function = dpmpp_sde_function
    elif sampler_name == "euler_ancestral":

        def euler_ancestral_function(
            model: torch.nn.Module,
            noise: torch.Tensor,
            sigmas: torch.Tensor,
            extra_args: dict,
            callback: callable,
            disable: bool,
            pipeline: bool,
        ) -> torch.Tensor:
            return sample_euler_ancestral(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                pipeline=pipeline,
                **extra_options,
            )

        sampler_function = euler_ancestral_function
    
    elif sampler_name == "euler":

        def euler_function(model, noise, sigmas, extra_args, callback, disable, pipeline=False):
            return sample_euler(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                pipeline=pipeline,
                **extra_options,
            )

        sampler_function = euler_function

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def sample(
    model: torch.nn.Module,
    noise: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    cfg: float,
    device: torch.device,
    sampler: KSAMPLER,
    sigmas: torch.Tensor,
    model_options: dict = {},
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    callback: callable = None,
    disable_pbar: bool = False,
    seed: int = None,
    pipeline: bool = False,
    flux: bool = False,
) -> torch.Tensor:
    """#### Sample using the given parameters.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `positive` (torch.Tensor): The positive tensor.
        - `negative` (torch.Tensor): The negative tensor.
        - `cfg` (float): The CFG value.
        - `device` (torch.device): The device.
        - `sampler` (KSAMPLER): The KSAMPLER object.
        - `sigmas` (torch.Tensor): The sigmas tensor.
        - `model_options` (dict, optional): The model options. Defaults to {}.
        - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
        - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
        - `callback` (callable, optional): The callback function. Defaults to None.
        - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
        - `seed` (int, optional): The seed value. Defaults to None.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    cfg_guider = CFGGuider(model, flux=flux)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        pipeline=pipeline,
    )


def sampler_object(name: str, pipeline: bool = False) -> KSAMPLER:
    """#### Get a sampler object.

    #### Args:
        - `name` (str): The sampler name.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.

    #### Returns:
        - `KSAMPLER`: The KSAMPLER object.
    """
    sampler = ksampler(name, pipeline=pipeline)
    return sampler


class KSampler1:
    """#### Class for KSampler1."""

    def __init__(
        self,
        model: torch.nn.Module,
        steps: int,
        device,
        sampler: str = None,
        scheduler: str = None,
        denoise: float = None,
        model_options: dict = {},
        pipeline: bool = False,
    ):
        """#### Initialize the KSampler1 class.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `steps` (int): The number of steps.
            - `device` (torch.device): The device.
            - `sampler` (str, optional): The sampler name. Defaults to None.
            - `scheduler` (str, optional): The scheduler name. Defaults to None.
            - `denoise` (float, optional): The denoise factor. Defaults to None.
            - `model_options` (dict, optional): The model options. Defaults to {}.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.
        """
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options
        self.pipeline = pipeline

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        """#### Calculate the sigmas for the given steps.

        #### Args:
            - `steps` (int): The number of steps.

        #### Returns:
            - `torch.Tensor`: The calculated sigmas.
        """
        sigmas = calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )
        return sigmas

    def set_steps(self, steps: int, denoise: float = None):
        """#### Set the steps and calculate the sigmas.

        #### Args:
            - `steps` (int): The number of steps.
            - `denoise` (float, optional): The denoise factor. Defaults to None.
        """
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps / denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1) :]

    def sample(
        self,
        noise: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        cfg: float,
        latent_image: torch.Tensor = None,
        start_step: int = None,
        last_step: int = None,
        force_full_denoise: bool = False,
        denoise_mask: torch.Tensor = None,
        sigmas: torch.Tensor = None,
        callback: callable = None,
        disable_pbar: bool = False,
        seed: int = None,
        pipeline: bool = False,
        flux: bool = False,
    ) -> torch.Tensor:
        """#### Sample using the KSampler1.

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `positive` (torch.Tensor): The positive tensor.
            - `negative` (torch.Tensor): The negative tensor.
            - `cfg` (float): The CFG value.
            - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
            - `start_step` (int, optional): The start step. Defaults to None.
            - `last_step` (int, optional): The last step. Defaults to None.
            - `force_full_denoise` (bool, optional): Whether to force full denoise. Defaults to False.
            - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            - `sigmas` (torch.Tensor, optional): The sigmas tensor. Defaults to None.
            - `callback` (callable, optional): The callback function. Defaults to None.
            - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
            - `seed` (int, optional): The seed value. Defaults to None.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[: last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler, pipeline=pipeline)

        return sample(
            self.model,
            noise,
            positive,
            negative,
            cfg,
            self.device,
            sampler,
            sigmas,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
            pipeline=pipeline,
            flux=flux
        )


def sample1(
    model: torch.nn.Module,
    noise: torch.Tensor,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: torch.Tensor,
    negative: torch.Tensor,
    latent_image: torch.Tensor,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step: int = None,
    last_step: int = None,
    force_full_denoise: bool = False,
    noise_mask: torch.Tensor = None,
    sigmas: torch.Tensor = None,
    callback: callable = None,
    disable_pbar: bool = False,
    seed: int = None,
    pipeline: bool = False,
    flux: bool = False,
) -> torch.Tensor:
    """#### Sample using the given parameters.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `steps` (int): The number of steps.
        - `cfg` (float): The CFG value.
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (torch.Tensor): The positive tensor.
        - `negative` (torch.Tensor): The negative tensor.
        - `latent_image` (torch.Tensor): The latent image tensor.
        - `denoise` (float, optional): The denoise factor. Defaults to 1.0.
        - `disable_noise` (bool, optional): Whether to disable noise. Defaults to False.
        - `start_step` (int, optional): The start step. Defaults to None.
        - `last_step` (int, optional): The last step. Defaults to None.
        - `force_full_denoise` (bool, optional): Whether to force full denoise. Defaults to False.
        - `noise_mask` (torch.Tensor, optional): The noise mask tensor. Defaults to None.
        - `sigmas` (torch.Tensor, optional): The sigmas tensor. Defaults to None.
        - `callback` (callable, optional): The callback function. Defaults to None.
        - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
        - `seed` (int, optional): The seed value. Defaults to None.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    sampler = KSampler1(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
        pipeline=pipeline,
    )

    samples = sampler.sample(
        noise,
        positive,
        negative,
        cfg=cfg,
        latent_image=latent_image,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        denoise_mask=noise_mask,
        sigmas=sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        pipeline=pipeline,
        flux=flux
    )
    samples = samples.to(intermediate_device())
    return samples


def common_ksampler(
    model: torch.nn.Module,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: torch.Tensor,
    negative: torch.Tensor,
    latent: dict,
    denoise: float = 1.0,
    disable_noise: bool = False,
    start_step: int = None,
    last_step: int = None,
    force_full_denoise: bool = False,
    pipeline: bool = False,
    flux: bool = False,
) -> tuple:
    """#### Common ksampler function.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `seed` (int): The seed value.
        - `steps` (int): The number of steps.
        - `cfg` (float): The CFG value.
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (torch.Tensor): The positive tensor.
        - `negative` (torch.Tensor): The negative tensor.
        - `latent` (dict): The latent dictionary.
        - `denoise` (float, optional): The denoise factor. Defaults to 1.0.
        - `disable_noise` (bool, optional): Whether to disable noise. Defaults to False.
        - `start_step` (int, optional): The start step. Defaults to None.
        - `last_step` (int, optional): The last step. Defaults to None.
        - `force_full_denoise` (bool, optional): Whether to force full denoise. Defaults to False.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.

    #### Returns:
        - `tuple`: The output tuple containing the latent dictionary and samples.
    """
    latent_image = latent["samples"]
    latent_image = fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]
    samples = sample1(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        seed=seed,
        pipeline=pipeline,
        flux=flux
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler2:
    """#### Class for KSampler2."""

    def sample(
        self,
        model: torch.nn.Module,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: torch.Tensor,
        negative: torch.Tensor,
        latent_image: torch.Tensor,
        denoise: float = 1.0,
        pipeline: bool = False,
        flux: bool = False,
    ) -> tuple:
        """#### Sample using the KSampler2.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `seed` (int): The seed value.
            - `steps` (int): The number of steps.
            - `cfg` (float): The CFG value.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler name.
            - `positive` (torch.Tensor): The positive tensor.
            - `negative` (torch.Tensor): The negative tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `denoise` (float, optional): The denoise factor. Defaults to 1.0.
            - `pipeline` (bool, optional): Whether to use the  Defaults to False.

        #### Returns:
            - `tuple`: The output tuple containing the latent dictionary and samples.
        """
        return common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            pipeline=pipeline,
            flux=flux
        )


class ModelType(Enum):
    """#### Enum for Model Types."""
    EPS = 1
    FLUX = 8


def model_sampling(model_config: dict, model_type: ModelType, flux: bool = False) -> torch.nn.Module:
    """#### Create a model sampling instance.

    #### Args:
        - `model_config` (dict): The model configuration.
        - `model_type` (ModelType): The model type.

    #### Returns:
        - `torch.nn.Module`: The model sampling instance.
    """
    if not flux:
        s = ModelSamplingDiscrete
        if model_type == ModelType.EPS:
            c = EPS

        class ModelSampling(s, c):
            pass

        return ModelSampling(model_config)
    else:
        c = CONST
        s = ModelSamplingFlux

        class ModelSampling(s, c):
            pass

        return ModelSampling(model_config)


def sample_custom(
    model: torch.nn.Module,
    noise: torch.Tensor,
    cfg: float,
    sampler: KSAMPLER,
    sigmas: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    latent_image: torch.Tensor,
    noise_mask: torch.Tensor = None,
    callback: callable = None,
    disable_pbar: bool = False,
    seed: int = None,
    pipeline: bool = False,
) -> torch.Tensor:
    """#### Custom sampling function.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `cfg` (float): The CFG value.
        - `sampler` (KSAMPLER): The KSAMPLER object.
        - `sigmas` (torch.Tensor): The sigmas tensor.
        - `positive` (torch.Tensor): The positive tensor.
        - `negative` (torch.Tensor): The negative tensor.
        - `latent_image` (torch.Tensor): The latent image tensor.
        - `noise_mask` (torch.Tensor, optional): The noise mask tensor. Defaults to None.
        - `callback` (callable, optional): The callback function. Defaults to None.
        - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
        - `seed` (int, optional): The seed value. Defaults to None.
        - `pipeline` (bool, optional): Whether to use the  Defaults to False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    samples = sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        model.load_device,
        sampler,
        sigmas,
        model_options=model.model_options,
        latent_image=latent_image,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        pipeline=pipeline,
    )
    samples = samples.to(intermediate_device())
    return samples

import torch

class CLIPTextModel_(torch.nn.Module):
    def __init__(
        self,
        config_dict: dict,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPTextModel_ module.

        #### Args:
            - `config_dict` (dict): The configuration dictionary.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(
            embed_dim,
            num_positions=num_positions,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )
        self.final_layer_norm = operations.LayerNorm(
            embed_dim, dtype=dtype, device=device
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor = None,
        intermediate_output: int = None,
        final_layer_norm_intermediate: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> tuple:
        """#### Forward pass for the CLIPTextModel_ module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.
            - `attention_mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `intermediate_output` (int, optional): The intermediate output layer. Defaults to None.
            - `final_layer_norm_intermediate` (bool, optional): Whether to apply final layer normalization to the intermediate output. Defaults to True.

        #### Returns:
            - `tuple`: The output tensor, the intermediate output tensor, and the pooled output tensor.
        """
        x = self.embeddings(input_tokens, dtype=dtype)
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            (
                torch.round(input_tokens).to(dtype=torch.int, device=x.device)
                == self.eos_token_id
            )
            .int()
            .argmax(dim=-1),
        ]
        return x, i, pooled_output

class CLIPTextModel(torch.nn.Module):
    def __init__(
        self,
        config_dict: dict,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPTextModel module.

        #### Args:
            - `config_dict` (dict): The configuration dictionary.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.dtype = dtype

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """#### Get the input embeddings.

        #### Returns:
            - `torch.nn.Embedding`: The input embeddings.
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings: torch.nn.Embedding) -> None:
        """#### Set the input embeddings.

        #### Args:
            - `embeddings` (torch.nn.Embedding): The input embeddings.
        """
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs) -> tuple:
        """#### Forward pass for the CLIPTextModel module.

        #### Args:
            - `*args`: Variable length argument list.
            - `**kwargs`: Arbitrary keyword arguments.

        #### Returns:
            - `tuple`: The output tensors.
        """
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])

from abc import abstractmethod
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F



oai_ops = disable_weight_init


class TimestepBlock1(nn.Module):
    """#### Abstract class representing a timestep block."""

    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the timestep block.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        pass


def forward_timestep_embed1(
    ts: nn.ModuleList,
    x: torch.Tensor,
    emb: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    transformer_options: Optional[Dict[str, Any]] = {},
    output_shape: Optional[torch.Size] = None,
    time_context: Optional[torch.Tensor] = None,
    num_video_frames: Optional[int] = None,
    image_only_indicator: Optional[bool] = None,
) -> torch.Tensor:
    """#### Forward pass for timestep embedding.

    #### Args:
        - `ts` (nn.ModuleList): The list of timestep blocks.
        - `x` (torch.Tensor): The input tensor.
        - `emb` (torch.Tensor): The embedding tensor.
        - `context` (torch.Tensor, optional): The context tensor. Defaults to None.
        - `transformer_options` (dict, optional): The transformer options. Defaults to {}.
        - `output_shape` (torch.Size, optional): The output shape. Defaults to None.
        - `time_context` (torch.Tensor, optional): The time context tensor. Defaults to None.
        - `num_video_frames` (int, optional): The number of video frames. Defaults to None.
        - `image_only_indicator` (bool, optional): The image only indicator. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    for layer in ts:
        if isinstance(layer, TimestepBlock1):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample1):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class Upsample1(nn.Module):
    """#### Class representing an upsample layer."""

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the upsample layer.

        #### Args:
            - `channels` (int): The number of input channels.
            - `use_conv` (bool): Whether to use convolution.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `padding` (int, optional): The padding size. Defaults to 1.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = operations.conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
                dtype=dtype,
                device=device,
            )

    def forward(
        self, x: torch.Tensor, output_shape: Optional[torch.Size] = None
    ) -> torch.Tensor:
        """#### Forward pass for the upsample layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `output_shape` (torch.Size, optional): The output shape. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        assert x.shape[1] == self.channels
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample1(nn.Module):
    """#### Class representing a downsample layer."""

    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the downsample layer.

        #### Args:
            - `channels` (int): The number of input channels.
            - `use_conv` (bool): Whether to use convolution.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `padding` (int, optional): The padding size. Defaults to 1.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = operations.conv_nd(
            dims,
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the downsample layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock1(TimestepBlock1):
    """#### Class representing a residual block layer."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the residual block layer.

        #### Args:
            - `channels` (int): The number of input channels.
            - `emb_channels` (int): The number of embedding channels.
            - `dropout` (float): The dropout rate.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `use_conv` (bool, optional): Whether to use convolution. Defaults to False.
            - `use_scale_shift_norm` (bool, optional): Whether to use scale shift normalization. Defaults to False.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `use_checkpoint` (bool, optional): Whether to use checkpointing. Defaults to False.
            - `up` (bool, optional): Whether to use upsampling. Defaults to False.
            - `down` (bool, optional): Whether to use downsampling. Defaults to False.
            - `kernel_size` (int, optional): The kernel size. Defaults to 3.
            - `exchange_temb_dims` (bool, optional): Whether to exchange embedding dimensions. Defaults to False.
            - `skip_t_emb` (bool, optional): Whether to skip embedding. Defaults to False.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                emb_channels,
                (2 * self.out_channels if use_scale_shift_norm else self.out_channels),
                dtype=dtype,
                device=device,
            ),
        )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(
                dims,
                self.out_channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 1, dtype=dtype, device=device
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the residual block layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Internal forward pass for the residual block layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if emb_out is not None:
            h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


ops = disable_weight_init


class ResnetBlock(nn.Module):
    """#### Class representing a ResNet block layer."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        """#### Initialize the ResNet block layer.

        #### Args:
            - `in_channels` (int): The number of input channels.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `conv_shortcut` (bool, optional): Whether to use convolution shortcut. Defaults to False.
            - `dropout` (float): The dropout rate.
            - `temb_channels` (int, optional): The number of embedding channels. Defaults to 512.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = ops.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = ops.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ops.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the ResNet block layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `temb` (torch.Tensor): The embedding tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


import numpy as np
import torch


def center_of_bbox(bbox: list) -> tuple[float, float]:
    """#### Calculate the center of a bounding box.

    #### Args:
        - `bbox` (list): The bounding box coordinates [x1, y1, x2, y2].

    #### Returns:
        - `tuple[float, float]`: The center coordinates (x, y).
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def make_2d_mask(mask: torch.Tensor) -> torch.Tensor:
    """#### Convert a mask to 2D.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.

    #### Returns:
        - `torch.Tensor`: The 2D mask tensor.
    """
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask


def combine_masks2(masks: list) -> torch.Tensor | None:
    """#### Combine multiple masks into one.

    #### Args:
        - `masks` (list): A list of mask tensors.

    #### Returns:
        - `torch.Tensor | None`: The combined mask tensor or None if no masks are provided.
    """
    try:
        mask = torch.from_numpy(np.array(masks[0]).astype(np.uint8))
    except:
        print("No Human Detected")
        return None
    return mask


def dilate_mask(
    mask: torch.Tensor, dilation_factor: int, iter: int = 1
) -> torch.Tensor:
    """#### Dilate a mask.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.
        - `dilation_factor` (int): The dilation factor.
        - `iter` (int, optional): The number of iterations. Defaults to 1.

    #### Returns:
        - `torch.Tensor`: The dilated mask tensor.
    """
    return make_2d_mask(mask)


def make_3d_mask(mask: torch.Tensor) -> torch.Tensor:
    """#### Convert a mask to 3D.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.

    #### Returns:
        - `torch.Tensor`: The 3D mask tensor.
    """
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


import logging
import math
from typing import Any, Dict, List, Optional
import torch.nn as nn
import torch as th
import torch


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
}

# taken from https://github.com/TencentARC/T2I-Adapter


def unet_to_diffusers(unet_config: dict) -> dict:
    """#### Convert a UNet configuration to a diffusers configuration.

    #### Args:
        - `unet_config` (dict): The UNet configuration.

    #### Returns:
        - `dict`: The diffusers configuration.
    """
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = (
                "input_blocks.{}.0.op.{}".format(n, k)
            )

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = (
            "middle_block.1.{}".format(b)
        )
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map[
                "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)
            ] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map[
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])
            ] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        length = num_res_blocks[x] + 1
        for i in range(length):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(
                            n, t, b
                        )
            if i == length - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map[
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k)
                    ] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


def apply_control1(h: th.Tensor, control: any, name: str) -> th.Tensor:
    """#### Apply control to a tensor.

    #### Args:
        - `h` (torch.Tensor): The input tensor.
        - `control` (any): The control to apply.
        - `name` (str): The name of the control.

    #### Returns:
        - `torch.Tensor`: The controlled tensor.
    """
    return h


oai_ops = disable_weight_init


class UNetModel1(nn.Module):
    """#### UNet Model class."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: list,
        dropout: float = 0,
        channel_mult: tuple = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: int = None,
        use_checkpoint: bool = False,
        dtype: th.dtype = th.float32,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_spatial_transformer: bool = False,  # custom transformer support
        transformer_depth: int = 1,  # custom transformer support
        context_dim: int = None,  # custom transformer support
        n_embed: int = None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy: bool = True,
        disable_self_attentions: list = None,
        num_attention_blocks: list = None,
        disable_middle_self_attn: bool = False,
        use_linear_in_transformer: bool = False,
        adm_in_channels: int = None,
        transformer_depth_middle: int = None,
        transformer_depth_output: list = None,
        use_temporal_resblock: bool = False,
        use_temporal_attention: bool = False,
        time_context_dim: int = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: any = None,
        merge_factor: float = 0.0,
        video_kernel_size: int = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        device: th.device = None,
        operations: any = oai_ops,
    ):
        """#### Initialize the UNetModel1 class.

        #### Args:
            - `image_size` (int): The size of the input image.
            - `in_channels` (int): The number of input channels.
            - `model_channels` (int): The number of model channels.
            - `out_channels` (int): The number of output channels.
            - `num_res_blocks` (list): The number of residual blocks.
            - `dropout` (float, optional): The dropout rate. Defaults to 0.
            - `channel_mult` (tuple, optional): The channel multiplier. Defaults to (1, 2, 4, 8).
            - `conv_resample` (bool, optional): Whether to use convolutional resampling. Defaults to True.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `num_classes` (int, optional): The number of classes. Defaults to None.
            - `use_checkpoint` (bool, optional): Whether to use checkpointing. Defaults to False.
            - `dtype` (torch.dtype, optional): The data type. Defaults to torch.float32.
            - `num_heads` (int, optional): The number of heads. Defaults to -1.
            - `num_head_channels` (int, optional): The number of head channels. Defaults to -1.
            - `num_heads_upsample` (int, optional): The number of heads for upsampling. Defaults to -1.
            - `use_scale_shift_norm` (bool, optional): Whether to use scale-shift normalization. Defaults to False.
            - `resblock_updown` (bool, optional): Whether to use residual blocks for up/down  Defaults to False.
            - `use_new_attention_order` (bool, optional): Whether to use a new attention order. Defaults to False.
            - `use_spatial_transformer` (bool, optional): Whether to use a spatial  Defaults to False.
            - `transformer_depth` (int, optional): The depth of the  Defaults to 1.
            - `context_dim` (int, optional): The context dimension. Defaults to None.
            - `n_embed` (int, optional): The number of embeddings. Defaults to None.
            - `legacy` (bool, optional): Whether to use legacy mode. Defaults to True.
            - `disable_self_attentions` (list, optional): The list of self-attentions to disable. Defaults to None.
            - `num_attention_blocks` (list, optional): The number of attention blocks. Defaults to None.
            - `disable_middle_self_attn` (bool, optional): Whether to disable middle self-attention. Defaults to False.
            - `use_linear_in_transformer` (bool, optional): Whether to use linear in  Defaults to False.
            - `adm_in_channels` (int, optional): The number of ADM input channels. Defaults to None.
            - `transformer_depth_middle` (int, optional): The depth of the middle  Defaults to None.
            - `transformer_depth_output` (list, optional): The depth of the output  Defaults to None.
            - `use_temporal_resblock` (bool, optional): Whether to use temporal residual blocks. Defaults to False.
            - `use_temporal_attention` (bool, optional): Whether to use temporal attention. Defaults to False.
            - `time_context_dim` (int, optional): The time context dimension. Defaults to None.
            - `extra_ff_mix_layer` (bool, optional): Whether to use an extra feed-forward mix layer. Defaults to False.
            - `use_spatial_context` (bool, optional): Whether to use spatial context. Defaults to False.
            - `merge_strategy` (any, optional): The merge strategy. Defaults to None.
            - `merge_factor` (float, optional): The merge factor. Defaults to 0.0.
            - `video_kernel_size` (int, optional): The video kernel size. Defaults to None.
            - `disable_temporal_crossattention` (bool, optional): Whether to disable temporal cross-attention. Defaults to False.
            - `max_ddpm_temb_period` (int, optional): The maximum DDPM temporal embedding period. Defaults to 10000.
            - `device` (torch.device, optional): The device to use. Defaults to None.
            - `operations` (any, optional): The operations to use. Defaults to oai_ops.
        """
        super().__init__()

        if context_dim is not None:
            assert use_spatial_transformer, "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None

        self.default_num_video_frames = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(
                model_channels, time_embed_dim, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, time_embed_dim, dtype=self.dtype, device=device
            ),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential1(
                    operations.conv_nd(
                        dims,
                        in_channels,
                        model_channels,
                        3,
                        padding=1,
                        dtype=self.dtype,
                        device=device,
                    )
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch: int,
            num_heads: int,
            dim_head: int,
            depth: int = 1,
            context_dim: int = None,
            use_checkpoint: bool = False,
            disable_self_attn: bool = False,
        ) -> SpatialTransformer:
            """#### Get an attention layer.

            #### Args:
                - `ch` (int): The number of channels.
                - `num_heads` (int): The number of heads.
                - `dim_head` (int): The dimension of each head.
                - `depth` (int, optional): The depth of the  Defaults to 1.
                - `context_dim` (int, optional): The context dimension. Defaults to None.
                - `use_checkpoint` (bool, optional): Whether to use checkpointing. Defaults to False.
                - `disable_self_attn` (bool, optional): Whether to disable self-attention. Defaults to False.

            #### Returns:
                - `SpatialTransformer`: The attention layer.
            """
            return SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )

        def get_resblock(
            merge_factor: float,
            merge_strategy: any,
            video_kernel_size: int,
            ch: int,
            time_embed_dim: int,
            dropout: float,
            out_channels: int,
            dims: int,
            use_checkpoint: bool,
            use_scale_shift_norm: bool,
            down: bool = False,
            up: bool = False,
            dtype: th.dtype = None,
            device: th.device = None,
            operations: any = oai_ops,
        ) -> ResBlock1:
            """#### Get a residual block.

            #### Args:
                - `merge_factor` (float): The merge factor.
                - `merge_strategy` (any): The merge strategy.
                - `video_kernel_size` (int): The video kernel size.
                - `ch` (int): The number of channels.
                - `time_embed_dim` (int): The time embedding dimension.
                - `dropout` (float): The dropout rate.
                - `out_channels` (int): The number of output channels.
                - `dims` (int): The number of dimensions.
                - `use_checkpoint` (bool): Whether to use checkpointing.
                - `use_scale_shift_norm` (bool): Whether to use scale-shift normalization.
                - `down` (bool, optional): Whether to use downsampling. Defaults to False.
                - `up` (bool, optional): Whether to use upsampling. Defaults to False.
                - `dtype` (torch.dtype, optional): The data type. Defaults to None.
                - `device` (torch.device, optional): The device. Defaults to None.
                - `operations` (any, optional): The operations to use. Defaults to oai_ops.

            #### Returns:
                - `ResBlock1`: The residual block.
            """
            return ResBlock1(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_channels,
                use_checkpoint=use_checkpoint,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                dtype=dtype,
                device=device,
                operations=operations,
            )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    dim_head = ch // num_heads
                    disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential1(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Downsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        dim_head = ch // num_heads
        mid_block = [
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )
        ]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [
                    get_attention_layer(  # always uses a self-attn
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth_middle,
                        context_dim=context_dim,
                        disable_self_attn=disable_middle_self_attn,
                        use_checkpoint=use_checkpoint,
                    ),
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=None,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    ),
                ]
            self.middle_block = TimestepEmbedSequential1(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    dim_head = ch // num_heads
                    disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Upsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            zero_module(
                operations.conv_nd(
                    dims,
                    model_channels,
                    out_channels,
                    3,
                    padding=1,
                    dtype=self.dtype,
                    device=device,
                )
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
        transformer_options: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> torch.Tensor:
        """#### Forward pass of the UNet model.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `timesteps` (Optional[torch.Tensor], optional): The timesteps tensor. Defaults to None.
            - `context` (Optional[torch.Tensor], optional): The context tensor. Defaults to None.
            - `y` (Optional[torch.Tensor], optional): The class labels tensor. Defaults to None.
            - `control` (Optional[torch.Tensor], optional): The control tensor. Defaults to None.
            - `transformer_options` (Dict[str, Any], optional): Options for the  Defaults to {}.
            - `**kwargs` (Any): Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels
        ).to(x.dtype)
        emb = self.time_embed(t_emb)
        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control1(h, control, "input")
            hs.append(h)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed1(
                self.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control1(h, control, "middle")

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control1(hsp, control, "output")

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        return self.out(h)


def detect_unet_config(state_dict: Dict[str, torch.Tensor], key_prefix: str) -> Dict[str, Any]:
    """#### Detect the UNet configuration from a state dictionary.

    #### Args:
        - `state_dict` (Dict[str, torch.Tensor]): The state dictionary.
        - `key_prefix` (str): The key prefix.

    #### Returns:
        - `Dict[str, Any]`: The detected UNet configuration.
    """
    state_dict_keys = list(state_dict.keys())

    if (
        "{}joint_blocks.0.context_block.attn.qkv.weight".format(key_prefix)
        in state_dict_keys
    ):  # mmdit model
        unet_config = {}
        unet_config["in_channels"] = state_dict[
            "{}x_embedder.proj.weight".format(key_prefix)
        ].shape[1]
        patch_size = state_dict["{}x_embedder.proj.weight".format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size
        final_layer = "{}final_layer.linear.weight".format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (
                patch_size * patch_size
            )

        unet_config["depth"] = (
            state_dict["{}x_embedder.proj.weight".format(key_prefix)].shape[0] // 64
        )
        unet_config["input_size"] = None
        y_key = "{}y_embedder.mlp.0.weight".format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        context_key = "{}context_embedder.weight".format(key_prefix)
        if context_key in state_dict_keys:
            in_features = state_dict[context_key].shape[1]
            out_features = state_dict[context_key].shape[0]
            unet_config["context_embedder_config"] = {
                "target": "torch.nn.Linear",
                "params": {"in_features": in_features, "out_features": out_features},
            }
        num_patches_key = "{}pos_embed".format(key_prefix)
        if num_patches_key in state_dict_keys:
            num_patches = state_dict[num_patches_key].shape[1]
            unet_config["num_patches"] = num_patches
            unet_config["pos_embed_max_size"] = round(math.sqrt(num_patches))

        rms_qk = "{}joint_blocks.0.context_block.attn.ln_q.weight".format(key_prefix)
        if rms_qk in state_dict_keys:
            unet_config["qk_norm"] = "rms"

        unet_config["pos_embed_scaling_factor"] = None  # unused for inference
        context_processor = "{}context_processor.layers.0.attn.qkv.weight".format(
            key_prefix
        )
        if context_processor in state_dict_keys:
            unet_config["context_processor_layers"] = count_blocks(
                state_dict_keys,
                "{}context_processor.layers.".format(key_prefix) + "{}.",
            )
        return unet_config

    if "{}clf.1.weight".format(key_prefix) in state_dict_keys:  # stable cascade
        unet_config = {}
        text_mapper_name = "{}clip_txt_mapper.weight".format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config["stable_cascade_stage"] = "c"
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536:  # stage c lite
                unet_config["c_cond"] = 1536
                unet_config["c_hidden"] = [1536, 1536]
                unet_config["nhead"] = [24, 24]
                unet_config["blocks"] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048:  # stage c full
                unet_config["c_cond"] = 2048
        elif "{}clip_mapper.weight".format(key_prefix) in state_dict_keys:
            unet_config["stable_cascade_stage"] = "b"
            w = state_dict["{}down_blocks.1.0.channelwise.0.weight".format(key_prefix)]
            if w.shape[-1] == 640:
                unet_config["c_hidden"] = [320, 640, 1280, 1280]
                unet_config["nhead"] = [-1, -1, 20, 20]
                unet_config["blocks"] = [[2, 6, 28, 6], [6, 28, 6, 2]]
                unet_config["block_repeat"] = [[1, 1, 1, 1], [3, 3, 2, 2]]
            elif w.shape[-1] == 576:  # stage b lite
                unet_config["c_hidden"] = [320, 576, 1152, 1152]
                unet_config["nhead"] = [-1, 9, 18, 18]
                unet_config["blocks"] = [[2, 4, 14, 4], [4, 14, 4, 2]]
                unet_config["block_repeat"] = [[1, 1, 1, 1], [2, 2, 2, 2]]
        return unet_config

    if (
        "{}rotary_pos_emb.inv_freq".format(key_prefix) in state_dict_keys
    ):  # stable audio dit
        unet_config = {}
        unet_config["audio_model"] = "dit1.0"
        return unet_config

    if (
        "{}double_layers.0.attn.w1q.weight".format(key_prefix) in state_dict_keys
    ):  # aura flow dit
        unet_config = {}
        unet_config["max_seq"] = state_dict[
            "{}positional_encoding".format(key_prefix)
        ].shape[1]
        unet_config["cond_seq_dim"] = state_dict[
            "{}cond_seq_linear.weight".format(key_prefix)
        ].shape[1]
        double_layers = count_blocks(
            state_dict_keys, "{}double_layers.".format(key_prefix) + "{}."
        )
        single_layers = count_blocks(
            state_dict_keys, "{}single_layers.".format(key_prefix) + "{}."
        )
        unet_config["n_double_layers"] = double_layers
        unet_config["n_layers"] = double_layers + single_layers
        return unet_config

    if "{}mlp_t5.0.weight".format(key_prefix) in state_dict_keys:  # Hunyuan DiT
        unet_config = {}
        unet_config["image_model"] = "hydit"
        unet_config["depth"] = count_blocks(
            state_dict_keys, "{}blocks.".format(key_prefix) + "{}."
        )
        unet_config["hidden_size"] = state_dict[
            "{}x_embedder.proj.weight".format(key_prefix)
        ].shape[0]
        if unet_config["hidden_size"] == 1408 and unet_config["depth"] == 40:  # DiT-g/2
            unet_config["mlp_ratio"] = 4.3637
        if state_dict["{}extra_embedder.0.weight".format(key_prefix)].shape[1] == 3968:
            unet_config["size_cond"] = True
            unet_config["use_style_cond"] = True
            unet_config["image_model"] = "hydit1"
        return unet_config

    if (
        "{}double_blocks.0.img_attn.norm.key_norm.scale".format(key_prefix)
        in state_dict_keys
    ):  # Flux
        dit_config = {}
        dit_config["image_model"] = "flux"
        dit_config["in_channels"] = 16
        dit_config["vec_in_dim"] = 768
        dit_config["context_in_dim"] = 4096
        dit_config["hidden_size"] = 3072
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 24
        dit_config["depth"] = count_blocks(
            state_dict_keys, "{}double_blocks.".format(key_prefix) + "{}."
        )
        dit_config["depth_single_blocks"] = count_blocks(
            state_dict_keys, "{}single_blocks.".format(key_prefix) + "{}."
        )
        dit_config["axes_dim"] = [16, 56, 56]
        dit_config["theta"] = 10000
        dit_config["qkv_bias"] = True
        dit_config["guidance_embed"] = (
            "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
        )
        return dit_config

    if "{}input_blocks.0.0.weight".format(key_prefix) not in state_dict_keys:
        return None

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False,
    }

    y_input = "{}label_emb.0.0.weight".format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    model_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[0]
    in_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[1]

    out_key = "{}out.2.weight".format(key_prefix)
    if out_key in state_dict:
        out_channels = state_dict[out_key].shape[0]
    else:
        out_channels = 4

    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False
    video_model_cross = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(
        state_dict_keys, "{}input_blocks".format(key_prefix) + ".{}."
    )
    for count in range(input_block_count):
        prefix = "{}input_blocks.{}.".format(key_prefix, count)
        prefix_output = "{}output_blocks.{}.".format(
            key_prefix, input_block_count - count - 1
        )

        block_keys = sorted(
            list(filter(lambda a: a.startswith(prefix), state_dict_keys))
        )
        if len(block_keys) == 0:
            break

        block_keys_output = sorted(
            list(filter(lambda a: a.startswith(prefix_output), state_dict_keys))
        )

        if "{}0.op.weight".format(prefix) in block_keys:  # new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(
                prefix_output, state_dict_keys, state_dict
            )
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = (
                    state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0]
                    // model_channels
                )

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        out[3]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(
                    prefix_output, state_dict_keys, state_dict
                )
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(
            state_dict_keys,
            "{}middle_block.1.transformer_blocks.".format(key_prefix) + "{}",
        )
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config["use_linear_in_transformer"] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    if video_model:
        unet_config["extra_ff_mix_layer"] = True
        unet_config["use_spatial_context"] = True
        unet_config["merge_strategy"] = "learned_with_images"
        unet_config["merge_factor"] = 0.0
        unet_config["video_kernel_size"] = [3, 1, 1]
        unet_config["use_temporal_resblock"] = True
        unet_config["use_temporal_attention"] = True
        unet_config["disable_temporal_crossattention"] = not video_model_cross
    else:
        unet_config["use_temporal_resblock"] = False
        unet_config["use_temporal_attention"] = False

    return unet_config


def model_config_from_unet_config(unet_config: Dict[str, Any], state_dict: Optional[Dict[str, torch.Tensor]] = None) -> Any:
    """#### Get the model configuration from a UNet configuration.

    #### Args:
        - `unet_config` (Dict[str, Any]): The UNet configuration.
        - `state_dict` (Optional[Dict[str, torch.Tensor]], optional): The state dictionary. Defaults to None.

    #### Returns:
        - `Any`: The model configuration.
    """

    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None


def model_config_from_unet(state_dict: Dict[str, torch.Tensor], unet_key_prefix: str, use_base_if_no_match: bool = False) -> Any:
    """#### Get the model configuration from a UNet state dictionary.

    #### Args:
        - `state_dict` (Dict[str, torch.Tensor]): The state dictionary.
        - `unet_key_prefix` (str): The UNet key prefix.
        - `use_base_if_no_match` (bool, optional): Whether to use the base configuration if no match is found. Defaults to False.

    #### Returns:
        - `Any`: The model configuration.
    """
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config(unet_config, state_dict)
    return model_config


def unet_dtype1(
    device: Optional[torch.device] = None,
    model_params: int = 0,
    supported_dtypes: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """#### Get the dtype for the UNet model.

    #### Args:
        - `device` (Optional[torch.device], optional): The device. Defaults to None.
        - `model_params` (int, optional): The model parameters. Defaults to 0.
        - `supported_dtypes` (List[torch.dtype], optional): The supported dtypes. Defaults to [torch.float16, torch.bfloat16, torch.float32].

    #### Returns:
        - `torch.dtype`: The dtype for the UNet model.
    """
    return torch.float16

import json
import logging
import numbers
import torch



def gen_empty_tokens(special_tokens: dict, length: int) -> list:
    """#### Generate a list of empty tokens.

    #### Args:
        - `special_tokens` (dict): The special tokens.
        - `length` (int): The length of the token list.

    #### Returns:
        - `list`: The list of empty tokens.
    """
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    """#### Class representing a CLIP token weight encoder."""

    def encode_token_weights(self, token_weight_pairs: list) -> tuple:
        """#### Encode token weights.

        #### Args:
            - `token_weight_pairs` (list): The token weight pairs.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if len(output) == 0:
            r = (out[-1:].to(intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = (
                        v[:sections]
                        .flatten()
                        .unsqueeze(dim=0)
                        .to(intermediate_device())
                    )
                extra[k] = v

            r = r + (extra,)
        return r

class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """#### Uses the CLIP transformer encoder for text (from huggingface)."""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version: str = "openai/clip-vit-large-patch14",
        device: str = "cpu",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        layer_idx: int = None,
        textmodel_json_config: str = None,
        dtype: torch.dtype = None,
        model_class: type = CLIPTextModel,
        special_tokens: dict = {"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state: bool = True,
        enable_attention_masks: bool = False,
        zero_out_masked:bool = False,
        return_projected_pooled: bool = True,
        return_attention_masks: bool = False,
        model_options={},
    ):
        """#### Initialize the SDClipModel.

        #### Args:
            - `version` (str, optional): The version of the model. Defaults to "openai/clip-vit-large-patch14".
            - `device` (str, optional): The device to use. Defaults to "cpu".
            - `max_length` (int, optional): The maximum length of the input. Defaults to 77.
            - `freeze` (bool, optional): Whether to freeze the model parameters. Defaults to True.
            - `layer` (str, optional): The layer to use. Defaults to "last".
            - `layer_idx` (int, optional): The index of the layer. Defaults to None.
            - `textmodel_json_config` (str, optional): The path to the JSON config file. Defaults to None.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `model_class` (type, optional): The model class. Defaults to 
            - `special_tokens` (dict, optional): The special tokens. Defaults to {"start": 49406, "end": 49407, "pad": 49407}.
            - `layer_norm_hidden_state` (bool, optional): Whether to normalize the hidden state. Defaults to True.
            - `enable_attention_masks` (bool, optional): Whether to enable attention masks. Defaults to False.
            - `zero_out_masked` (bool, optional): Whether to zero out masked tokens. Defaults to False.
            - `return_projected_pooled` (bool, optional): Whether to return the projected pooled output. Defaults to True.
            - `return_attention_masks` (bool, optional): Whether to return the attention masks. Defaults to False.
            - `model_options` (dict, optional): Additional model options. Defaults to {}.
        """
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = "./_internal/clip/sd1_clip_config.json"

        with open(textmodel_json_config) as f:
            config = json.load(f)

        operations = model_options.get("custom_operations", None)
        if operations is None:
            operations = manual_cast

        self.operations = operations
        self.transformer = model_class(config, dtype, device, self.operations)
        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def freeze(self) -> None:
        """#### Freeze the model parameters."""
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options: dict) -> None:
        """#### Set the CLIP options.

        #### Args:
            - `options` (dict): The options to set.
        """
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self) -> None:
        """#### Reset the CLIP options to default."""
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens: list, current_embeds: torch.nn.Embedding) -> list:
        """#### Set up the textual embeddings.

        #### Args:
            - `tokens` (list): The input tokens.
            - `current_embeds` (torch.nn.Embedding): The current embeddings.

        #### Returns:
            - `list`: The processed tokens.
        """
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, numbers.Integral):
                    tokens_temp += [int(y)]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        logging.warning(
                            "WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
                                y.shape[0], current_embeds.weight.shape[1]
                            )
                        )
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = self.operations.Embedding(
                next_new_token + 1,
                current_embeds.weight.shape[1],
                device=current_embeds.weight.device,
                dtype=current_embeds.weight.dtype,
            )
            new_embedding.weight[:token_dict_size] = current_embeds.weight
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))
            ]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens: list) -> tuple:
        """#### Forward pass of the model.

        #### Args:
            - `tokens` (list): The input tokens.

        #### Returns:
            - `tuple`: The output and the pooled output.
        """
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if (
            self.enable_attention_masks
            or self.zero_out_masked
            or self.return_attention_masks
        ):
            attention_mask = torch.zeros_like(tokens)
            end_token = self.special_tokens.get("end", -1)
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == end_token:
                        break

        attention_mask_model = None
        if self.enable_attention_masks:
            attention_mask_model = attention_mask

        outputs = self.transformer(
            tokens,
            attention_mask_model,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=torch.float32,
        )
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0:
            return z, pooled_output, extra

        return z, pooled_output

    def encode(self, tokens: list) -> tuple:
        """#### Encode the input tokens.

        #### Args:
            - `tokens` (list): The input tokens.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        return self(tokens)

    def load_sd(self, sd: dict) -> None:
        """#### Load the state dictionary.

        #### Args:
            - `sd` (dict): The state dictionary.
        """
        return self.transformer.load_state_dict(sd, strict=False)


class SD1ClipModel(torch.nn.Module):
    """#### Class representing the SD1ClipModel."""

    def __init__(
        self, device: str = "cpu", dtype: torch.dtype = None, clip_name: str = "l", clip_model: type = SDClipModel, **kwargs
    ):
        """#### Initialize the SD1ClipModel.

        #### Args:
            - `device` (str, optional): The device to use. Defaults to "cpu".
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `clip_name` (str, optional): The name of the CLIP model. Defaults to "l".
            - `clip_model` (type, optional): The CLIP model class. Defaults to SDClipModel.
            - `**kwargs`: Additional keyword arguments.
        """
        super().__init__()
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        self.lowvram_patch_counter = 0
        self.model_loaded_weight_memory = 0
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def set_clip_options(self, options: dict) -> None:
        """#### Set the CLIP options.

        #### Args:
            - `options` (dict): The options to set.
        """
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self) -> None:
        """#### Reset the CLIP options to default."""
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
        """#### Encode token weights.

        #### Args:
            - `token_weight_pairs` (dict): The token weight pairs.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out, pooled = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out, pooled

import logging
import os
import traceback
import torch
from transformers import CLIPTokenizer

def model_options_long_clip(sd, tokenizer_data, model_options):
    w = sd.get("clip_l.text_model.embeddings.position_embedding.weight", None)
    if w is None:
        w = sd.get("text_model.embeddings.position_embedding.weight", None)
    return tokenizer_data, model_options

def parse_parentheses(string: str) -> list:
    """#### Parse a string with nested parentheses.

    #### Args:
        - `string` (str): The input string.

    #### Returns:
        - `list`: The parsed list of strings.
    """
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string: str, current_weight: float) -> list:
    """#### Parse a string into tokens with weights.

    #### Args:
        - `string` (str): The input string.
        - `current_weight` (float): The current weight.

    #### Returns:
        - `list`: The list of token-weight pairs.
    """
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text: str) -> str:
    """#### Escape important characters in a string.

    #### Args:
        - `text` (str): The input text.

    #### Returns:
        - `str`: The escaped text.
    """
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text: str) -> str:
    """#### Unescape important characters in a string.

    #### Args:
        - `text` (str): The input text.

    #### Returns:
        - `str`: The unescaped text.
    """
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def expand_directory_list(directories: list) -> list:
    """#### Expand a list of directories to include all subdirectories.

    #### Args:
        - `directories` (list): The list of directories.

    #### Returns:
        - `list`: The expanded list of directories.
    """
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def load_embed(embedding_name: str, embedding_directory: list, embedding_size: int, embed_key: str = None) -> torch.Tensor:
    """#### Load an embedding from a directory.

    #### Args:
        - `embedding_name` (str): The name of the embedding.
        - `embedding_directory` (list): The list of directories to search.
        - `embedding_size` (int): The size of the embedding.
        - `embed_key` (str, optional): The key for the embedding. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The loaded embedding.
    """
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = [".safetensors", ".pt", ".bin"]
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch

            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            if "weights_only" in torch.load.__code__.co_varnames:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            else:
                embed = torch.load(embed_path, map_location="cpu")
    except Exception:
        logging.warning(
            "{}\n\nerror loading embedding, skipping loading: {}".format(
                traceback.format_exc(), embedding_name
            )
        )
        return None

    if embed_out is None:
        if "string_to_param" in embed:
            values = embed["string_to_param"].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            values = embed.values()
            embed_out = next(iter(values))
    return embed_out


class SDTokenizer:
    """#### Class representing a Stable Diffusion tokenizer."""

    def __init__(
        self,
        tokenizer_path: str = None,
        max_length: int = 77,
        pad_with_end: bool = True,
        embedding_directory: str = None,
        embedding_size: int = 768,
        embedding_key: str = "clip_l",
        tokenizer_class: type = CLIPTokenizer,
        has_start_token: bool = True,
        pad_to_max_length: bool = True,
        min_length: int = None,
    ):
        """#### Initialize the SDTokenizer.

        #### Args:
            - `tokenizer_path` (str, optional): The path to the tokenizer. Defaults to None.
            - `max_length` (int, optional): The maximum length of the input. Defaults to 77.
            - `pad_with_end` (bool, optional): Whether to pad with the end token. Defaults to True.
            - `embedding_directory` (str, optional): The directory for embeddings. Defaults to None.
            - `embedding_size` (int, optional): The size of the embeddings. Defaults to 768.
            - `embedding_key` (str, optional): The key for the embeddings. Defaults to "clip_l".
            - `tokenizer_class` (type, optional): The tokenizer class. Defaults to CLIPTokenizer.
            - `has_start_token` (bool, optional): Whether the tokenizer has a start token. Defaults to True.
            - `pad_to_max_length` (bool, optional): Whether to pad to the maximum length. Defaults to True.
            - `min_length` (int, optional): The minimum length of the input. Defaults to None.
        """
        if tokenizer_path is None:
            tokenizer_path = "_internal/sd1_tokenizer/"
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name: str) -> tuple:
        """#### Try to get an embedding.

        #### Args:
            - `embedding_name` (str): The name of the embedding.

        #### Returns:
            - `tuple`: The embedding and any leftover text.
        """
        embed = load_embed(
            embedding_name,
            self.embedding_directory,
            self.embedding_size,
            self.embedding_key,
        )
        if embed is None:
            stripped = embedding_name.strip(",")
            if len(stripped) < len(embedding_name):
                embed = load_embed(
                    stripped,
                    self.embedding_directory,
                    self.embedding_size,
                    self.embedding_key,
                )
                return (embed, embedding_name[len(stripped) :])
        return (embed, "")

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> list:
        """#### Tokenize text with weights.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `list`: The tokenized text with weights.
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = (
                unescape_important(weighted_segment).replace("\n", " ").split(" ")
            )
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if (
                    word.startswith(self.embedding_identifier)
                    and self.embedding_directory is not None
                ):
                    embedding_name = word[len(self.embedding_identifier) :].strip("\n")
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(
                            f"warning, embedding:{embedding_name} does not exist, ignoring"
                        )
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append(
                                [(embed[x], weight) for x in range(embed.shape[0])]
                            )
                        print("loading ", embedding_name)
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                # parse word
                tokens.append(
                    [
                        (t, weight)
                        for t in self.tokenizer(word)["input_ids"][
                            self.tokens_start : -1
                        ]
                    ]
                )

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend(
                            [(t, w, i + 1) for t, w in t_group[:remaining_length]]
                        )
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair: list) -> list:
        """#### Untokenize a list of token-weight pairs.

        #### Args:
            - `token_weight_pair` (list): The list of token-weight pairs.

        #### Returns:
            - `list`: The untokenized list.
        """
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SD1Tokenizer:
    """#### Class representing the SD1Tokenizer."""

    def __init__(self, embedding_directory: str = None, clip_name: str = "l", tokenizer: type = SDTokenizer):
        """#### Initialize the SD1Tokenizer.

        #### Args:
            - `embedding_directory` (str, optional): The directory for embeddings. Defaults to None.
            - `clip_name` (str, optional): The name of the CLIP model. Defaults to "l".
            - `tokenizer` (type, optional): The tokenizer class. Defaults to SDTokenizer.
        """
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict:
        """#### Tokenize text with weights.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `dict`: The tokenized text with weights.
        """
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(
            text, return_word_ids
        )
        return out

    def untokenize(self, token_weight_pair: list) -> list:
        """#### Untokenize a list of token-weight pairs.

        #### Args:
            - `token_weight_pair` (list): The list of token-weight pairs.

        #### Returns:
            - `list`: The untokenized list.
        """
        return getattr(self, self.clip).untokenize(token_weight_pair)

from typing import Literal
import torch
import torch.nn as nn

ConvMode = Literal["CNA", "NAC", "CNAC"]

def act(act_type: str, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    """#### Get the activation layer.

    #### Args:
        - `act_type` (str): The type of activation.
        - `inplace` (bool, optional): Whether to perform the operation in-place. Defaults to True.
        - `neg_slope` (float, optional): The negative slope for LeakyReLU. Defaults to 0.2.
        - `n_prelu` (int, optional): The number of PReLU parameters. Defaults to 1.

    #### Returns:
        - `nn.Module`: The activation layer.
    """
    act_type = act_type.lower()
    layer = nn.LeakyReLU(neg_slope, inplace)
    return layer

def get_valid_padding(kernel_size: int, dilation: int) -> int:
    """#### Get the valid padding for a convolutional layer.

    #### Args:
        - `kernel_size` (int): The size of the kernel.
        - `dilation` (int): The dilation rate.

    #### Returns:
        - `int`: The valid padding.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def sequential(*args: nn.Module) -> nn.Sequential:
    """#### Create a sequential container.

    #### Args:
        - `*args` (nn.Module): The modules to include in the sequential container.

    #### Returns:
        - `nn.Sequential`: The sequential container.
    """
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: str = "zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
    mode: ConvMode = "CNA",
    c2x2: bool = False,
) -> nn.Sequential:
    """#### Create a convolutional block.

    #### Args:
        - `in_nc` (int): The number of input channels.
        - `out_nc` (int): The number of output channels.
        - `kernel_size` (int): The size of the kernel.
        - `stride` (int, optional): The stride of the convolution. Defaults to 1.
        - `dilation` (int, optional): The dilation rate. Defaults to 1.
        - `groups` (int, optional): The number of groups. Defaults to 1.
        - `bias` (bool, optional): Whether to include a bias term. Defaults to True.
        - `pad_type` (str, optional): The type of padding. Defaults to "zero".
        - `norm_type` (str | None, optional): The type of normalization. Defaults to None.
        - `act_type` (str | None, optional): The type of activation. Defaults to "relu".
        - `mode` (ConvMode, optional): The mode of the convolution. Defaults to "CNA".
        - `c2x2` (bool, optional): Whether to use 2x2 convolutions. Defaults to False.

    #### Returns:
        - `nn.Sequential`: The convolutional block.
    """
    assert mode in ("CNA", "NAC", "CNAC"), "Wrong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if mode in ("CNA", "CNAC"):
        return sequential(None, c, None, a)

def upconv_block(
    in_nc: int,
    out_nc: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    pad_type: str = "zero",
    norm_type: str | None = None,
    act_type: str = "relu",
    mode: str = "nearest",
    c2x2: bool = False,
) -> nn.Sequential:
    """#### Create an upsampling convolutional block.

    #### Args:
        - `in_nc` (int): The number of input channels.
        - `out_nc` (int): The number of output channels.
        - `upscale_factor` (int, optional): The upscale factor. Defaults to 2.
        - `kernel_size` (int, optional): The size of the kernel. Defaults to 3.
        - `stride` (int, optional): The stride of the convolution. Defaults to 1.
        - `bias` (bool, optional): Whether to include a bias term. Defaults to True.
        - `pad_type` (str, optional): The type of padding. Defaults to "zero".
        - `norm_type` (str | None, optional): The type of normalization. Defaults to None.
        - `act_type` (str, optional): The type of activation. Defaults to "relu".
        - `mode` (str, optional): The mode of upsampling. Defaults to "nearest".
        - `c2x2` (bool, optional): Whether to use 2x2 convolutions. Defaults to False.

    #### Returns:
        - `nn.Sequential`: The upsampling convolutional block.
    """
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        c2x2=c2x2,
    )
    return sequential(upsample, conv)

class ShortcutBlock(nn.Module):
    """#### Elementwise sum the output of a submodule to its input."""

    def __init__(self, submodule: nn.Module):
        """#### Initialize the ShortcutBlock.

        #### Args:
            - `submodule` (nn.Module): The submodule to apply.
        """
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        output = x + self.sub(x)
        return output

def hash_arg(arg: any) -> any:
    """#### Hash an argument.

    #### Args:
        - `arg` (any): The argument to hash.

    #### Returns:
        - `any`: The hashed argument.
    """
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)

from typing import Dict, Tuple
import torch

class LatentFormat:
    """#### Base class for latent formats.

    #### Attributes:
        - `scale_factor` (float): The scale factor for the latent format.

    #### Returns:
        - `LatentFormat`: A latent format object.
    """

    scale_factor: float = 1.0
    latent_channels: int = 4
    
    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent input, by multiplying it by the scale factor.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent output, by dividing it by the scale factor.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return latent / self.scale_factor

class SD15(LatentFormat):
    """#### SD15 latent format.

    #### Args:
        - `LatentFormat` (LatentFormat): The base latent format class.
    """
    latent_channels: int = 4
    def __init__(self, scale_factor: float = 0.18215):
        """#### Initialize the SD15 latent format.

        #### Args:
            - `scale_factor` (float, optional): The scale factor. Defaults to 0.18215.
        """
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"
        
class SD3(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645, 0.0177, 0.1052],
            [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360],
            [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364],
            [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283],
            [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700],
            [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220],
            [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259],
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Flux1(SD3):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609],
            [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503],
            [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530],
            [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001],
            [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013],
            [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680],
        ]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class EmptyLatentImage:
    """#### A class to generate an empty latent image.

    #### Args:
        - `Device` (Device): The device to use for the latent image.
    """

    def __init__(self):
        """#### Initialize the EmptyLatentImage class."""
        self.device = intermediate_device()

    def generate(
        self, width: int, height: int, batch_size: int = 1
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """#### Generate an empty latent image

        #### Args:
            - `width` (int): The width of the latent image.
            - `height` (int): The height of the latent image.
            - `batch_size` (int, optional): The batch size. Defaults to 1.

        #### Returns:
            - `Tuple[Dict[str, torch.Tensor]]`: The generated latent image.
        """
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)

def fix_empty_latent_channels(model, latent_image):
    latent_channels = model.get_model_object(
        "latent_format"
    ).latent_channels  # Resize the empty latent image so it has the right number of channels
    if (
        latent_channels != latent_image.shape[1]
        and torch.count_nonzero(latent_image) == 0
    ):
        latent_image = repeat_to_batch_size(latent_image, latent_channels, dim=1)
    return latent_image

from collections import namedtuple
import numpy as np
import torch

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


def segs_bitwise_and_mask(segs: tuple, mask: torch.Tensor) -> tuple:
    """#### Apply bitwise AND operation between segmentation masks and a given mask.

    #### Args:
        - `segs` (tuple): A tuple containing segmentation information.
        - `mask` (torch.Tensor): The mask tensor.

    #### Returns:
        - `tuple`: A tuple containing the original segmentation and the updated items.
    """
    mask = make_2d_mask(mask)
    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[
            crop_region[1] : crop_region[3], crop_region[0] : crop_region[2]
        ]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(
            seg.cropped_image,
            new_mask,
            seg.confidence,
            seg.crop_region,
            seg.bbox,
            seg.label,
            None,
        )
        items.append(item)

    return segs[0], items


class SegsBitwiseAndMask:
    """#### Class to apply bitwise AND operation between segmentation masks and a given mask."""

    def doit(self, segs: tuple, mask: torch.Tensor) -> tuple:
        """#### Apply bitwise AND operation between segmentation masks and a given mask.

        #### Args:
            - `segs` (tuple): A tuple containing segmentation information.
            - `mask` (torch.Tensor): The mask tensor.

        #### Returns:
            - `tuple`: A tuple containing the original segmentation and the updated items.
        """
        return (segs_bitwise_and_mask(segs, mask),)


class SEGSLabelFilter:
    """#### Class to filter segmentation labels."""

    @staticmethod
    def filter(segs: tuple, labels: list) -> tuple:
        """#### Filter segmentation labels.

        #### Args:
            - `segs` (tuple): A tuple containing segmentation information.
            - `labels` (list): A list of labels to filter.

        #### Returns:
            - `tuple`: A tuple containing the original segmentation and an empty list.
        """
        labels = set([label.strip() for label in labels])
        return (
            segs,
            (segs[0], []),
        )


import numpy as np
import torch
from PIL import Image
import torchvision



def _tensor_check_image(image: torch.Tensor) -> None:
    """#### Check if the input is a valid tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
    """
    return


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """#### Convert a tensor to a PIL image.

    #### Args:
        - `image` (torch.Tensor): The input tensor.

    #### Returns:
        - `Image.Image`: The converted PIL image.
    """
    _tensor_check_image(image)
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def general_tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """#### Resize a tensor image using bilinear interpolation.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `w` (int): The target width.
        - `h` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The resized tensor image.
    """
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """#### Convert a PIL image to a tensor.

    #### Args:
        - `image` (Image.Image): The input PIL image.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class TensorBatchBuilder:
    """#### Class for building a batch of tensors."""

    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def concat(self, new_tensor: torch.Tensor) -> None:
        """#### Concatenate a new tensor to the batch.

        #### Args:
            - `new_tensor` (torch.Tensor): The new tensor to concatenate.
        """
        self.tensor = new_tensor


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """#### Resize a tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `w` (int): The target width.
        - `h` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The resized tensor image.
    """
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)


def tensor_paste(
    image1: torch.Tensor,
    image2: torch.Tensor,
    left_top: tuple[int, int],
    mask: torch.Tensor,
) -> None:
    """#### Paste one tensor image onto another using a mask.

    #### Args:
        - `image1` (torch.Tensor): The base tensor image.
        - `image2` (torch.Tensor): The tensor image to paste.
        - `left_top` (tuple[int, int]): The top-left corner where the image2 will be pasted.
        - `mask` (torch.Tensor): The mask tensor.
    """
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    mask = mask[:, :h, :w, :]
    image1[:, y : y + h, x : x + w, :] = (1 - mask) * image1[
        :, y : y + h, x : x + w, :
    ] + mask * image2[:, :h, :w, :]
    return


def tensor_convert_rgba(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """#### Convert a tensor image to RGBA format.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `prefer_copy` (bool, optional): Whether to prefer copying the tensor. Defaults to True.

    #### Returns:
        - `torch.Tensor`: The converted RGBA tensor image.
    """
    _tensor_check_image(image)
    alpha = torch.ones((*image.shape[:-1], 1))
    return torch.cat((image, alpha), axis=-1)


def tensor_convert_rgb(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """#### Convert a tensor image to RGB format.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `prefer_copy` (bool, optional): Whether to prefer copying the tensor. Defaults to True.

    #### Returns:
        - `torch.Tensor`: The converted RGB tensor image.
    """
    _tensor_check_image(image)
    return image


def tensor_get_size(image: torch.Tensor) -> tuple[int, int]:
    """#### Get the size of a tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.

    #### Returns:
        - `tuple[int, int]`: The width and height of the tensor image.
    """
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor_putalpha(image: torch.Tensor, mask: torch.Tensor) -> None:
    """#### Add an alpha channel to a tensor image using a mask.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `mask` (torch.Tensor): The mask tensor.
    """
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_mask(mask: torch.Tensor) -> None:
    """#### Check if the input is a valid tensor mask.

    #### Args:
        - `mask` (torch.Tensor): The input tensor mask.
    """
    return


def tensor_gaussian_blur_mask(
    mask: torch.Tensor | np.ndarray, kernel_size: int, sigma: float = 10.0
) -> torch.Tensor:
    """#### Apply Gaussian blur to a tensor mask.

    #### Args:
        - `mask` (torch.Tensor | np.ndarray): The input tensor mask.
        - `kernel_size` (int): The size of the Gaussian kernel.
        - `sigma` (float, optional): The standard deviation of the Gaussian kernel. Defaults to 10.0.

    #### Returns:
        - `torch.Tensor`: The blurred tensor mask.
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]

    _tensor_check_mask(mask)

    kernel_size = kernel_size * 2 + 1

    prev_device = mask.device
    device = get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    )(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """#### Convert a numpy array to a tensor.

    #### Args:
        - `image` (np.ndarray): The input numpy array.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    return torch.from_numpy(image)


from typing import List
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

orig_torch_load = torch.load

# importing YOLO breaking original torch.load capabilities
torch.load = orig_torch_load


def load_yolo(model_path: str) -> YOLO:
    """#### Load YOLO model.

    #### Args:
        - `model_path` (str): The path to the YOLO model.

    #### Returns:
        - `YOLO`: The YOLO model initialized with the specified model path.
    """
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        print("please download yolo model")


def inference_bbox(
    model: YOLO,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
) -> List:
    """#### Perform inference on an image and return bounding boxes.

    #### Args:
        - `model` (YOLO): The YOLO model.
        - `image` (Image.Image): The image to perform inference on.
        - `confidence` (float): The confidence threshold for the bounding boxes.
        - `device` (str): The device to run the model on.

    #### Returns:
        - `List[List[str, List[int], np.ndarray, float]]`: The list of bounding boxes.
    """
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def create_segmasks(results: List) -> List:
    """#### Create segmentation masks from the results of the inference.

    #### Args:
        - `results` (List[List[str, List[int], np.ndarray, float]]): The results of the inference.

    #### Returns:
        - `List[List[int], np.ndarray, float]`: The list of segmentation masks.
    """
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def dilate_masks(segmasks: List, dilation_factor: int, iter: int = 1) -> List:
    """#### Dilate the segmentation masks.

    #### Args:
        - `segmasks` (List[List[int], np.ndarray, float]): The segmentation masks.
        - `dilation_factor` (int): The dilation factor.
        - `iter` (int): The number of iterations.

    #### Returns:
        - `List[List[int], np.ndarray, float]`: The dilated segmentation masks.
    """
    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def normalize_region(limit: int, startp: int, size: int) -> List:
    """#### Normalize the region.

    #### Args:
        - `limit` (int): The limit.
        - `startp` (int): The start point.
        - `size` (int): The size.

    #### Returns:
        - `List[int]`: The normalized start and end points.
    """
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)

    return int(new_startp), int(new_endp)


def make_crop_region(w: int, h: int, bbox: List, crop_factor: float) -> List:
    """#### Make the crop region.

    #### Args:
        - `w` (int): The width.
        - `h` (int): The height.
        - `bbox` (List[int]): The bounding box.
        - `crop_factor` (float): The crop factor.

    #### Returns:
        - `List[x1: int, y1: int, x2: int, y2: int]`: The crop region.
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """#### Crop the ndarray in 2 dimensions.

    #### Args:
        - `npimg` (np.ndarray): The ndarray to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `np.ndarray`: The cropped ndarray.
    """
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_ndarray4(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """#### Crop the ndarray in 4 dimensions.

    #### Args:
        - `npimg` (np.ndarray): The ndarray to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `np.ndarray`: The cropped ndarray.
    """
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


def crop_image(image: Image.Image, crop_region: List) -> Image.Image:
    """#### Crop the image.

    #### Args:
        - `image` (Image.Image): The image to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `Image.Image`: The cropped image.
    """
    return crop_ndarray4(image, crop_region)


def segs_scale_match(segs: List[np.ndarray], target_shape: List) -> List:
    """#### Match the scale of the segmentation masks.

    #### Args:
        - `segs` (List[np.ndarray]): The segmentation masks.
        - `target_shape` (List[int]): The target shape.

    #### Returns:
        - `List[np.ndarray]`: The matched segmentation masks.
    """
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs


import math
import os
import torch
from transformers import T5TokenizerFast

activations = {
    "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}

class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, ff_activation, dtype, device, operations):
        super().__init__()
        self.wi_0 = operations.Linear(
            model_dim, ff_dim, bias=False, dtype=dtype, device=device
        )
        self.wi_1 = operations.Linear(
            model_dim, ff_dim, bias=False, dtype=dtype, device=device
        )
        self.wo = operations.Linear(
            ff_dim, model_dim, bias=False, dtype=dtype, device=device
        )
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.act = activations[ff_activation]

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        # x = self.dropout(x)
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(
        self, model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations
    ):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                model_dim, ff_dim, ff_activation, dtype, device, operations
            )

        self.layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # x = x + self.dropout(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.k = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.v = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.o = operations.Linear(
            inner_dim, model_dim, bias=False, dtype=dtype, device=device
        )
        self.num_heads = num_heads

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(
                self.relative_attention_num_buckets,
                self.num_heads,
                device=device,
                dtype=dtype,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket, out_dtype=dtype
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)

        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias

        out = optimized_attention(
            q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask
        )
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            model_dim,
            inner_dim,
            num_heads,
            relative_attention_bias,
            dtype,
            device,
            operations,
        )
        self.layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        self.layer_norm(x)
        output, past_bias = self.SelfAttention(
            self.layer_norm(x),
            mask=mask,
            past_bias=past_bias,
            optimized_attention=optimized_attention,
        )
        # x = x + self.dropout(attention_output)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                model_dim,
                inner_dim,
                ff_dim,
                num_heads,
                relative_attention_bias,
                dtype,
                device,
                operations,
            )
        )
        self.layer.append(
            T5LayerFF(
                model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations
            )
        )

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        x, past_bias = self.layer[0](x, mask, past_bias, optimized_attention)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention,
        dtype,
        device,
        operations,
    ):
        super().__init__()

        self.block = torch.nn.ModuleList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    ff_activation,
                    gated_act,
                    num_heads,
                    relative_attention_bias=((not relative_attention) or (i == 0)),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
    ):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        intermediate = None
        optimized_attention = optimized_attention_for_device()
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        model_dim = config_dict["d_model"]

        self.encoder = T5Stack(
            self.num_layers,
            model_dim,
            model_dim,
            config_dict["d_ff"],
            config_dict["dense_act_fn"],
            config_dict["is_gated_act"],
            config_dict["num_heads"],
            config_dict["model_type"] != "umt5",
            dtype,
            device,
            operations,
        )
        self.dtype = dtype
        self.shared = operations.Embedding(
            config_dict["vocab_size"], model_dim, device=device, dtype=dtype
        )

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

    def forward(self, input_ids, *args, **kwargs):
        x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x)  # Fix for fp8 T5 base
        return self.encoder(x, *args, **kwargs)

class T5XXLModel(SDClipModel):
    def __init__(
        self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}
    ):
        textmodel_json_config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "./clip/t5_config_xxl.json",
        )
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=T5,
            model_options=model_options,
        )


class T5XXLTokenizer(SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "./clip/t5_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="t5xxl",
            tokenizer_class=T5TokenizerFast,
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=256,
        )


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return cast_to_input(self.weight, x) * x


class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        clip_l_tokenizer_class = tokenizer_data.get(
            "clip_l_tokenizer_class", SDTokenizer
        )
        self.clip_l = clip_l_tokenizer_class(embedding_directory=embedding_directory)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = pick_weight_dtype(dtype_t5, dtype, device)
        clip_l_class = model_options.get("clip_l_class", SDClipModel)
        self.clip_l = clip_l_class(
            device=device,
            dtype=dtype,
            return_projected_pooled=False,
            model_options=model_options,
        )
        self.t5xxl = T5XXLModel(
            device=device, dtype=dtype_t5, model_options=model_options
        )
        self.dtypes = set([dtype, dtype_t5])

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)


def flux_clip(dtype_t5=None):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(
                dtype_t5=dtype_t5,
                device=device,
                dtype=dtype,
                model_options=model_options,
            )

    return FluxClipModel_

import copy
import logging
import uuid

import torch


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights
    m.weight_function = None
    m.bias_function = None

class ModelPatcher:
    def __init__(
        self,
        model: torch.nn.Module,
        load_device: torch.device,
        offload_device: torch.device,
        size: int = 0,
        current_device: torch.device = None,
        weight_inplace_update: bool = False,
    ):
        """#### Initialize the ModelPatcher class.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `load_device` (torch.device): The device to load the model on.
            - `offload_device` (torch.device): The device to offload the model to.
            - `size` (int, optional): The size of the model. Defaults to 0.
            - `current_device` (torch.device, optional): The current device. Defaults to None.
            - `weight_inplace_update` (bool, optional): Whether to update weights in place. Defaults to False.
        """
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()

        if not hasattr(self.model, "model_loaded_weight_memory"):
            self.model.model_loaded_weight_memory = 0
            
        if not hasattr(self.model, "model_lowvram"):
            self.model.model_lowvram = False
        
        if not hasattr(self.model, "lowvram_patch_counter"):
            self.model.lowvram_patch_counter = 0
    
    def loaded_size(self) -> int:
        """#### Get the loaded size
        
        #### Returns:
            - `int`: The loaded size
        """
        return self.model.model_loaded_weight_memory
    
    def model_size(self) -> int:
        """#### Get the size of the model.

        #### Returns:
            - `int`: The size of the model.
        """
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self) -> "ModelPatcher":
        """#### Clone the ModelPatcher object.

        #### Returns:
            - `ModelPatcher`: The cloned ModelPatcher object.
        """
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other: object) -> bool:
        """#### Check if the object is a clone.

        #### Args:
            - `other` (object): The other object.

        #### Returns:
            - `bool`: Whether the object is a clone.
        """
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape: tuple) -> float:
        """#### Calculate the memory required for the model.

        #### Args:
            - `input_shape` (tuple): The input shape.

        #### Returns:
            - `float`: The memory required.
        """
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: callable) -> None:
        """#### Set the UNet function wrapper for the model.

        #### Args:
            - `unet_wrapper_function` (callable): The UNet function wrapper.
        """
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function: callable) -> None:
        """#### Set the denoise mask function for the model.

        #### Args:
            - `denoise_mask_function` (callable): The denoise mask function.
        """
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name: str) -> object:
        """#### Get an object from the model.

        #### Args:
            - `name` (str): The name of the object.

        #### Returns:
            - `object`: The object.
        """
        return get_attr(self.model, name)

    def model_patches_to(self, device: torch.device) -> None:
        """#### Move model patches to a device.

        #### Args:
            - `device` (torch.device): The device.
        """
        self.model_options["transformer_options"]
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self) -> torch.dtype:
        """#### Get the data type of the model.

        #### Returns:
            - `torch.dtype`: The data type.
        """
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(
        self, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> list:
        """#### Add patches to the model.

        #### Args:
            - `patches` (dict): The patches to add.
            - `strength_patch` (float, optional): The strength of the patches. Defaults to 1.0.
            - `strength_model` (float, optional): The strength of the model. Defaults to 1.0.

        #### Returns:
            - `list`: The list of patched keys.
        """
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)
    
    def set_model_patch(self, patch: list, name: str):
        """#### Set a patch for the model.

        #### Args:
            - `patch` (list): The patch.
            - `name` (str): The name of the patch.
        """
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch: list):
        """#### Set the attention 1 patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch: list):
        """#### Set the attention 2 patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn2_patch")
        
    def set_model_attn1_output_patch(self, patch: list):
        """#### Set the attention 1 output patch for the model.

        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch: list):
        """#### Set the attention 2 output patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn2_output_patch")
    
    def model_state_dict(self, filter_prefix: str = None) -> dict:
        """#### Get the state dictionary of the model.

        #### Args:
            - `filter_prefix` (str, optional): The prefix to filter. Defaults to None.

        #### Returns:
            - `dict`: The state dictionary.
        """
        sd = self.model.state_dict()
        list(sd.keys())
        return sd

    def patch_weight_to_device(self, key: str, device_to: torch.device = None) -> None:
        """#### Patch the weight of a key to a device.

        #### Args:
            - `key` (str): The key.
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
        """
        if key not in self.patches:
            return

        weight = get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(
                device=self.offload_device, copy=inplace_update
            )

        if device_to is not None:
            temp_weight = cast_to_device(
                weight, device_to, torch.float32, copy=True
            )
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(
            weight.dtype
        )
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)
    
    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        loading = []
        for n, m in self.model.named_modules():
            if hasattr(m, "comfy_cast_weights") or hasattr(m, "weight"):
                loading.append((module_size(m), n, m))

        load_completely = []
        loading.sort(reverse=True)
        for x in loading:
            n = x[1]
            m = x[2]
            module_mem = x[0]

            lowvram_weight = False

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                        continue

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "comfy_cast_weights"):
                    if m.comfy_cast_weights:
                        wipe_lowvram_weight(m)

                if hasattr(m, "weight"):
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m))

        load_completely.sort(reverse=True)
        for x in load_completely:
            n = x[1]
            m = x[2]
            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights is True:
                    continue

            self.patch_weight_to_device(weight_key, device_to=device_to)
            self.patch_weight_to_device(bias_key, device_to=device_to)
            logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
            m.comfy_patched_weights = True

        for x in load_completely:
            x[2].to(device_to)

        if lowvram_counter > 0:
            logging.info(
                "loaded partially {} {} {}".format(
                    lowvram_model_memory / (1024 * 1024),
                    mem_counter / (1024 * 1024),
                    patch_counter,
                )
            )
            self.model.model_lowvram = True
        else:
            logging.info(
                "loaded completely {} {} {}".format(
                    lowvram_model_memory / (1024 * 1024),
                    mem_counter / (1024 * 1024),
                    full_load,
                )
            )
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        
        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter
    
    def patch_model_flux(
        self,
        device_to: torch.device = None,
        lowvram_model_memory: int =0,
        load_weights: bool = True,
        force_patch_weights: bool = False,
    ):
        """#### Patch the model.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `load_weights` (bool, optional): Whether to load weights. Defaults to True.
            - `force_patch_weights` (bool, optional): Whether to force patch weights. Defaults to False.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        for k in self.object_patches:
            old = set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if lowvram_model_memory == 0:
            full_load = True
        else:
            full_load = False

        if load_weights:
            self.load(
                device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                full_load=full_load,
            )
        return self.model

    def patch_model_lowvram_flux(
        self,
        device_to: torch.device = None,
        lowvram_model_memory: int = 0,
        force_patch_weights: bool = False,
    ) -> torch.nn.Module:
        """#### Patch the model for low VRAM.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch weights. Defaults to False.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        self.patch_model(device_to)

        logging.info(
            "loading in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024))
        )

        class LowVramPatch:
            def __init__(self, key: str, model_patcher: "ModelPatcher"):
                self.key = key
                self.model_patcher = model_patcher

            def __call__(self, weight: torch.Tensor) -> torch.Tensor:
                return self.model_patcher.calculate_weight(
                    self.model_patcher.patches[self.key], weight, self.key
                )

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model
    
    def patch_model(
        self, device_to: torch.device = None, patch_weights: bool = True
    ) -> torch.nn.Module:
        """#### Patch the model.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `patch_weights` (bool, optional): Whether to patch weights. Defaults to True.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        for k in self.object_patches:
            old = set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning(
                        "could not patch. key doesn't exist in model: {}".format(key)
                    )
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(
        self,
        device_to: torch.device = None,
        lowvram_model_memory: int = 0,
        force_patch_weights: bool = False,
    ) -> torch.nn.Module:
        """#### Patch the model for low VRAM.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch weights. Defaults to False.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        self.patch_model(device_to, patch_weights=False)

        logging.info(
            "loading in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024))
        )

        class LowVramPatch:
            def __init__(self, key: str, model_patcher: "ModelPatcher"):
                self.key = key
                self.model_patcher = model_patcher

            def __call__(self, weight: torch.Tensor) -> torch.Tensor:
                return self.model_patcher.calculate_weight(
                    self.model_patcher.patches[self.key], weight, self.key
                )

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(
        self, patches: list, weight: torch.Tensor, key: str
    ) -> torch.Tensor:
        """#### Calculate the weight of a key.

        #### Args:
            - `patches` (list): The list of patches.
            - `weight` (torch.Tensor): The weight tensor.
            - `key` (str): The key.

        #### Returns:
            - `torch.Tensor`: The calculated weight.
        """
        for p in patches:
            alpha = p[0]
            v = p[1]
            p[2]
            v[0]
            v = v[1]
            mat1 = cast_to_device(v[0], weight.device, torch.float32)
            mat2 = cast_to_device(v[1], weight.device, torch.float32)
            v[4]
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            weight += (
                (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
                .reshape(weight.shape)
                .type(weight.dtype)
            )
        return weight

    def unpatch_model(
        self, device_to: torch.device = None, unpatch_weights: bool = True
    ) -> None:
        """#### Unpatch the model.

        #### Args:
            - `device_to` (torch.device, optional): The device to unpatch to. Defaults to None.
            - `unpatch_weights` (bool, optional): Whether to unpatch weights. Defaults to True.
        """
        if unpatch_weights:
            keys = list(self.backup.keys())
            for k in keys:
                set_attr_param(self.model, k, self.backup[k])
            self.backup.clear()
            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        self.object_patches_backup.clear()
    
    def partially_load(self, device_to, extra_memory=0):
        self.unpatch_model(unpatch_weights=False)
        self.patch_model(patch_weights=False)
        full_load = False
        if self.model.model_lowvram is False:
            return 0
        if self.model.model_loaded_weight_memory + extra_memory > self.model_size():
            full_load = True
        current_used = self.model.model_loaded_weight_memory
        self.load(
            device_to,
            lowvram_model_memory=current_used + extra_memory,
            full_load=full_load,
        )
        return self.model.model_loaded_weight_memory - current_used

def unet_prefix_from_state_dict(state_dict):
    candidates = [
        "model.diffusion_model.",  # ldm/sgm models
        "model.model.",  # audio models
    ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model."  # aura flow and others

def load_diffusion_model_state_dict(
    sd, model_options={}
):  # load unet in diffusers or regular format
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = unet_prefix_from_state_dict(sd)
    temp_sd = state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = calculate_parameters(sd)
    load_device = get_torch_device()
    model_config = model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd

    offload_device = unet_offload_device()
    if dtype is None:
        unet_dtype2 = unet_dtype(
            model_params=parameters,
            supported_dtypes=model_config.supported_inference_dtypes,
        )
    else:
        unet_dtype2 = dtype

    manual_cast_dtype = unet_manual_cast(
        unet_dtype2, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype2, manual_cast_dtype)
    model_config.custom_operations = model_options.get(
        "custom_operations", model_config.custom_operations
    )
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device)


import logging
import math
import torch



class BaseModel(torch.nn.Module):
    """#### Base class for models."""

    def __init__(
        self,
        model_config: object,
        model_type: ModelType = ModelType.EPS,
        device: torch.device = None,
        unet_model: object = UNetModel1,
        flux: bool = False,
    ):
        """#### Initialize the BaseModel class.

        #### Args:
            - `model_config` (object): The model configuration.
            - `model_type` (ModelType, optional): The model type. Defaults to ModelType.EPS.
            - `device` (torch.device, optional): The device to use. Defaults to None.
            - `unet_model` (object, optional): The UNet model. Defaults to UNetModel1.
        """
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        if flux:
            if not unet_config.get("disable_unet_model_creation", False):
                operations = model_config.custom_operations
                self.diffusion_model = unet_model(
                    **unet_config, device=device, operations=operations
                )
                logging.info(
                    "model weight dtype {}, manual cast: {}".format(
                        self.get_dtype(), self.manual_cast_dtype
                    )
                )
        else:
            if not unet_config.get("disable_unet_model_creation", False):
                if self.manual_cast_dtype is not None:
                    operations = manual_cast
                else:
                    operations = disable_weight_init
                self.diffusion_model = unet_model(
                    **unet_config, device=device, operations=operations
                )
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type, flux=flux)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor if flux else 2.0

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: torch.Tensor = None,
        c_crossattn: torch.Tensor = None,
        control: torch.Tensor = None,
        transformer_options: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        """#### Apply the model to the input tensor.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The timestep tensor.
            - `c_concat` (torch.Tensor, optional): The concatenated condition tensor. Defaults to None.
            - `c_crossattn` (torch.Tensor, optional): The cross-attention condition tensor. Defaults to None.
            - `control` (torch.Tensor, optional): The control tensor. Defaults to None.
            - `transformer_options` (dict, optional): The transformer options. Defaults to {}.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(
            xc,
            t,
            context=context,
            control=control,
            transformer_options=transformer_options,
            **extra_conds,
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self) -> torch.dtype:
        """#### Get the data type of the model.

        #### Returns:
            - `torch.dtype`: The data type.
        """
        return self.diffusion_model.dtype

    def encode_adm(self, **kwargs) -> None:
        """#### Encode the ADM.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `None`: The encoded ADM.
        """
        return None

    def extra_conds(self, **kwargs) -> dict:
        """#### Get the extra conditions.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `dict`: The extra conditions.
        """
        out = {}
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out["crossattn_controlnet"] = CONDCrossAttn(cross_attn_cnet)

        return out

    def load_model_weights(self, sd: dict, unet_prefix: str = "") -> "BaseModel":
        """#### Load the model weights.

        #### Args:
            - `sd` (dict): The state dictionary.
            - `unet_prefix` (str, optional): The UNet prefix. Defaults to "".

        #### Returns:
            - `BaseModel`: The model with loaded weights.
        """
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent input.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent output.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return self.latent_format.process_out(latent)

    def memory_required(self, input_shape: tuple) -> float:
        """#### Calculate the memory required for the model.

        #### Args:
            - `input_shape` (tuple): The input shape.

        #### Returns:
            - `float`: The memory required.
        """
        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        # TODO: this needs to be tweaked
        area = input_shape[0] * math.prod(input_shape[2:])
        return (area * dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (
            1024 * 1024
        )


class BASE:
    """#### Base class for model configurations."""

    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    custom_operations = None

    @classmethod
    def matches(cls, unet_config: dict, state_dict: dict = None) -> bool:
        """#### Check if the UNet configuration matches.

        #### Args:
            - `unet_config` (dict): The UNet configuration.
            - `state_dict` (dict, optional): The state dictionary. Defaults to None.

        #### Returns:
            - `bool`: Whether the configuration matches.
        """
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict: dict, prefix: str = "") -> ModelType:
        """#### Get the model type.

        #### Args:
            - `state_dict` (dict): The state dictionary.
            - `prefix` (str, optional): The prefix. Defaults to "".

        #### Returns:
            - `ModelType`: The model type.
        """
        return ModelType.EPS

    def inpaint_model(self) -> bool:
        """#### Check if the model is an inpaint model.

        #### Returns:
            - `bool`: Whether the model is an inpaint model.
        """
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config: dict):
        """#### Initialize the BASE class.

        #### Args:
            - `unet_config` (dict): The UNet configuration.
        """
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(
        self, state_dict: dict, prefix: str = "", device: torch.device = None
    ) -> BaseModel:
        """#### Get the model.

        #### Args:
            - `state_dict` (dict): The state dictionary.
            - `prefix` (str, optional): The prefix. Defaults to "".
            - `device` (torch.device, optional): The device to use. Defaults to None.

        #### Returns:
            - `BaseModel`: The model.
        """
        out = BaseModel(
            self, model_type=self.model_type(state_dict, prefix), device=device
        )
        return out

    def process_unet_state_dict(self, state_dict: dict) -> dict:
        """#### Process the UNet state dictionary.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        return state_dict

    def process_vae_state_dict(self, state_dict: dict) -> dict:
        """#### Process the VAE state dictionary.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        return state_dict

    def set_inference_dtype(
        self, dtype: torch.dtype, manual_cast_dtype: torch.dtype
    ) -> None:
        """#### Set the inference data type.

        #### Args:
            - `dtype` (torch.dtype): The data type.
            - `manual_cast_dtype` (torch.dtype): The manual cast data type.
        """
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


import math
import numpy as np
import torch
from PIL import Image


def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int) -> int:
    """#### Calculate the number of steps required for tiled scaling.

    #### Args:
        - `width` (int): The width of the image.
        - `height` (int): The height of the image.
        - `tile_x` (int): The width of each tile.
        - `tile_y` (int): The height of each tile.
        - `overlap` (int): The overlap between tiles.

    #### Returns:
        - `int`: The number of steps required for tiled scaling.
    """
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples: torch.Tensor,
    function: callable,
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4,
    out_channels: int = 3,
    pbar: any = None,
) -> torch.Tensor:
    """#### Perform tiled scaling on a batch of samples.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `function` (callable): The function to apply to each tile.
        - `tile_x` (int, optional): The width of each tile. Defaults to 64.
        - `tile_y` (int, optional): The height of each tile. Defaults to 64.
        - `overlap` (int, optional): The overlap between tiles. Defaults to 8.
        - `upscale_amount` (float, optional): The upscale amount. Defaults to 4.
        - `out_channels` (int, optional): The number of output channels. Defaults to 3.
        - `pbar` (any, optional): The progress bar. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The scaled output tensor.
    """
    output = torch.empty(
        (
            samples.shape[0],
            out_channels,
            round(samples.shape[2] * upscale_amount),
            round(samples.shape[3] * upscale_amount),
        ),
        device="cpu",
    )
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        out_div = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y : y + tile_y, x : x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                        1.0 / feather
                    ) * (t + 1)
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                        1.0 / feather
                    ) * (t + 1)
                out[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += ps * mask
                out_div[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += mask

        output[b : b + 1] = out / out_div
    return output


def flatten(img: Image.Image, bgcolor: str) -> Image.Image:
    """#### Replace transparency with a background color.

    #### Args:
        - `img` (Image.Image): The input image.
        - `bgcolor` (str): The background color.

    #### Returns:
        - `Image.Image`: The image with transparency replaced by the background color.
    """
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert(
        "RGB"
    )


BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
    """#### Convert a tensor to a PIL image.

    #### Args:
        - `img_tensor` (torch.Tensor): The input tensor.
        - `batch_index` (int, optional): The batch index. Defaults to 0.

    #### Returns:
        - `Image.Image`: The converted PIL image.
    """
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """#### Convert a PIL image to a tensor.

    #### Args:
        - `image` (Image.Image): The input PIL image.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_crop_region(mask: Image.Image, pad: int = 0) -> tuple:
    """#### Get the coordinates of the white rectangular mask region.

    #### Args:
        - `mask` (Image.Image): The input mask image in 'L' mode.
        - `pad` (int, optional): The padding to apply. Defaults to 0.

    #### Returns:
        - `tuple`: The coordinates of the crop region.
    """
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region: tuple, image_size: tuple) -> tuple:
    """#### Remove the extra pixel added by the get_crop_region function.

    #### Args:
        - `region` (tuple): The crop region coordinates.
        - `image_size` (tuple): The size of the image.

    #### Returns:
        - `tuple`: The fixed crop region coordinates.
    """
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region: tuple, width: int, height: int, target_width: int, target_height: int) -> tuple:
    """#### Expand a crop region to a specified target size.

    #### Args:
        - `region` (tuple): The crop region coordinates.
        - `width` (int): The width of the image.
        - `height` (int): The height of the image.
        - `target_width` (int): The desired width of the crop region.
        - `target_height` (int): The desired height of the crop region.

    #### Returns:
        - `tuple`: The expanded crop region coordinates and the target size.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def crop_cond(cond: list, region: tuple, init_size: tuple, canvas_size: tuple, tile_size: tuple, w_pad: int = 0, h_pad: int = 0) -> list:
    """#### Crop conditioning data to match a specific region.

    #### Args:
        - `cond` (list): The conditioning data.
        - `region` (tuple): The crop region coordinates.
        - `init_size` (tuple): The initial size of the image.
        - `canvas_size` (tuple): The size of the canvas.
        - `tile_size` (tuple): The size of the tile.
        - `w_pad` (int, optional): The width padding. Defaults to 0.
        - `h_pad` (int, optional): The height padding. Defaults to 0.

    #### Returns:
        - `list`: The cropped conditioning data.
    """
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        cropped.append(n)
    return cropped

from collections import OrderedDict
import functools
import math
import re
from typing import Union, Dict
import torch
import torch.nn as nn


class RRDB(nn.Module):
    """#### Residual in Residual Dense Block (RRDB) class.

    #### Args:
        - `nf` (int): Number of filters.
        - `kernel_size` (int, optional): Kernel size. Defaults to 3.
        - `gc` (int, optional): Growth channel. Defaults to 32.
        - `stride` (int, optional): Stride. Defaults to 1.
        - `bias` (bool, optional): Whether to use bias. Defaults to True.
        - `pad_type` (str, optional): Padding type. Defaults to "zero".
        - `norm_type` (str, optional): Normalization type. Defaults to None.
        - `act_type` (str, optional): Activation type. Defaults to "leakyrelu".
        - `mode` (ConvMode, optional): Convolution mode. Defaults to "CNA".
        - `_convtype` (str, optional): Convolution type. Defaults to "Conv2D".
        - `_spectral_norm` (bool, optional): Whether to use spectral normalization. Defaults to False.
        - `plus` (bool, optional): Whether to use the plus variant. Defaults to False.
        - `c2x2` (bool, optional): Whether to use 2x2 convolution. Defaults to False.
    """

    def __init__(
        self,
        nf: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: str = "zero",
        norm_type: str = None,
        act_type: str = "leakyrelu",
        mode: ConvMode = "CNA",
        _convtype: str = "Conv2D",
        _spectral_norm: bool = False,
        plus: bool = False,
        c2x2: bool = False,
    ) -> None:
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the RRDB.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """#### Residual Dense Block with 5 Convolutions (ResidualDenseBlock_5C) class.

    #### Args:
        - `nf` (int, optional): Number of filters. Defaults to 64.
        - `kernel_size` (int, optional): Kernel size. Defaults to 3.
        - `gc` (int, optional): Growth channel. Defaults to 32.
        - `stride` (int, optional): Stride. Defaults to 1.
        - `bias` (bool, optional): Whether to use bias. Defaults to True.
        - `pad_type` (str, optional): Padding type. Defaults to "zero".
        - `norm_type` (str, optional): Normalization type. Defaults to None.
        - `act_type` (str, optional): Activation type. Defaults to "leakyrelu".
        - `mode` (ConvMode, optional): Convolution mode. Defaults to "CNA".
        - `plus` (bool, optional): Whether to use the plus variant. Defaults to False.
        - `c2x2` (bool, optional): Whether to use 2x2 convolution. Defaults to False.
    """

    def __init__(
        self,
        nf: int = 64,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: str = "zero",
        norm_type: str = None,
        act_type: str = "leakyrelu",
        mode: ConvMode = "CNA",
        plus: bool = False,
        c2x2: bool = False,
    ) -> None:
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1x1 = None

        self.conv1 = conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        last_act = None
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            c2x2=c2x2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the ResidualDenseBlock_5C.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBNet(nn.Module):
    """#### Residual in Residual Dense Block Network (RRDBNet) class.

    #### Args:
        - `state_dict` (dict): State dictionary.
        - `norm` (str, optional): Normalization type. Defaults to None.
        - `act` (str, optional): Activation type. Defaults to "leakyrelu".
        - `upsampler` (str, optional): Upsampler type. Defaults to "upconv".
        - `mode` (ConvMode, optional): Convolution mode. Defaults to "CNA".
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        norm: str = None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: ConvMode = "CNA",
    ) -> None:
        super(RRDBNet, self).__init__()
        self.model_arch = "ESRGAN"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch _internal
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())

        self.state = self.new_to_old_arch(self.state)

        self.key_arr = list(self.state.keys())

        self.in_nc: int = self.state[self.key_arr[0]].shape[1]
        self.out_nc: int = self.state[self.key_arr[-1]].shape[0]

        self.scale: int = self.get_scale()
        self.num_filters: int = self.state[self.key_arr[0]].shape[0]

        c2x2 = False

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        self.shuffle_factor = None

        upsample_block = {
            "upconv": upconv_block,
        }.get(self.upsampler)
        upsample_blocks = [
            upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                act_type=self.act,
                c2x2=c2x2,
            )
            for _ in range(int(math.log(self.scale, 2)))
        ]

        self.model = sequential(
            # fea conv
            conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            ShortcutBlock(
                sequential(
                    # rrdb blocks
                    *[
                        RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                            c2x2=c2x2,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """#### Convert new architecture state dictionary to old architecture.

        #### Args:
            - `state` (dict): State dictionary.

        #### Returns:
            - `dict`: Converted state dictionary.
        """
        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        # upconv layers
        max_upconv = 0
        for key in state.keys():
            match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
            if match is not None:
                _, key_num, key_type = match.groups()
                old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
                max_upconv = max(max_upconv, int(key_num) * 3)

        # final layers
        for key in state.keys():
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key in ("conv_last.weight",):
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key in ("conv_last.bias",):
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        # Sort by first numeric value of each layer
        def compare(item1: str, item2: str) -> int:
            parts1 = item1.split(".")
            parts2 = item2.split(".")
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2

        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

        # Rebuild the output dict in the right order
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

        return out_dict

    def get_scale(self, min_part: int = 6) -> int:
        """#### Get the scale factor.

        #### Args:
            - `min_part` (int, optional): Minimum part. Defaults to 6.

        #### Returns:
            - `int`: Scale factor.
        """
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        """#### Get the number of blocks.

        #### Returns:
            - `int`: Number of blocks.
        """
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the RRDBNet.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        return self.model(x)


PyTorchSRModels = (RRDBNet,)
PyTorchSRModel = Union[RRDBNet,]

PyTorchModels = (*PyTorchSRModels,)
PyTorchModel = Union[PyTorchSRModel]

import contextlib
import torch


class ModuleFactory:
    """#### Base class for module factories."""

    def get_converted_kwargs(self) -> dict:
        """#### Get the converted keyword arguments.

        #### Returns:
            - `dict`: The converted keyword arguments.
        """
        return self.converted_kwargs


class BaseModelApplyModelModule(torch.nn.Module):
    """#### Module for applying a model function."""

    def __init__(self, func: callable, module: torch.nn.Module):
        """#### Initialize the BaseModelApplyModelModule.

        #### Args:
            - `func` (callable): The function to apply.
            - `module` (torch.nn.Module): The module to apply the function to.
        """
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        c_concat: torch.Tensor = None,
        c_crossattn: torch.Tensor = None,
        y: torch.Tensor = None,
        control: torch.Tensor = None,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        """#### Forward pass of the module.

        #### Args:
            - `input_x` (torch.Tensor): The input tensor.
            - `timestep` (torch.Tensor): The timestep tensor.
            - `c_concat` (torch.Tensor, optional): The concatenated conditioning tensor. Defaults to None.
            - `c_crossattn` (torch.Tensor, optional): The cross-attention conditioning tensor. Defaults to None.
            - `y` (torch.Tensor, optional): The target tensor. Defaults to None.
            - `control` (torch.Tensor, optional): The control tensor. Defaults to None.
            - `transformer_options` (dict, optional): The transformer options. Defaults to {}.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        kwargs = {"y": y}

        new_transformer_options = {}

        return self.func(
            input_x,
            timestep,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=new_transformer_options,
            **kwargs,
        )


class BaseModelApplyModelModuleFactory(ModuleFactory):
    """#### Factory for creating BaseModelApplyModelModule instances."""

    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable: callable, kwargs: dict) -> None:
        """#### Initialize the BaseModelApplyModelModuleFactory.

        #### Args:
            - `callable` (callable): The callable to use.
            - `kwargs` (dict): The keyword arguments.
        """
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self) -> dict:
        """#### Generate the converted keyword arguments.

        #### Returns:
            - `dict`: The converted keyword arguments.
        """
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self) -> tuple:
        """#### Generate a cache key.

        #### Returns:
            - `tuple`: The cache key.
        """
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            key_kwargs[k] = v

        patch_module_cache_key = {}
        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.unet_config),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        """#### Context manager for the converted module.

        #### Yields:
            - `tuple`: The module and the converted keyword arguments.
        """
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)

import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional


class UltraBBoxDetector:
    """#### Class to detect bounding boxes using a YOLO model."""

    bbox_model: Optional[YOLO] = None

    def __init__(self, bbox_model: YOLO):
        """#### Initialize the UltraBBoxDetector with a YOLO model.

        #### Args:
            - `bbox_model` (YOLO): The YOLO model to use for detection.
        """
        self.bbox_model = bbox_model

    def detect(
        self,
        image: torch.Tensor,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int = 1,
        detailer_hook: Optional[callable] = None,
    ) -> Tuple[Tuple[int, int], List[SEG]]:
        """#### Detect bounding boxes in an image.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `threshold` (float): The detection threshold.
            - `dilation` (int): The dilation factor for masks.
            - `crop_factor` (float): The crop factor for bounding boxes.
            - `drop_size` (int, optional): The minimum size of bounding boxes to keep. Defaults to 1.
            - `detailer_hook` (callable, optional): A hook function for additional processing. Defaults to None.

        #### Returns:
            - `Tuple[Tuple[int, int], List[SEG]]`: The shape of the image and a list of detected segments.
        """
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(
            self.bbox_model, tensor2pil(image), threshold
        )
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if (
                x2 - x1 > drop_size and y2 - y1 > drop_size
            ):  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]

                item = SEG(
                    cropped_image,
                    cropped_mask,
                    confidence,
                    crop_region,
                    item_bbox,
                    label,
                    None,
                )

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        return segs


class UltraSegmDetector:
    """#### Class to detect segments using a YOLO model."""

    bbox_model: Optional[YOLO] = None

    def __init__(self, bbox_model: YOLO):
        """#### Initialize the UltraSegmDetector with a YOLO model.

        #### Args:
            - `bbox_model` (YOLO): The YOLO model to use for detection.
        """
        self.bbox_model = bbox_model


class NO_SEGM_DETECTOR:
    """#### Placeholder class for no segment detector."""

    pass


class UltralyticsDetectorProvider:
    """#### Class to provide YOLO models for detection."""

    def doit(self, model_name: str) -> Tuple[UltraBBoxDetector, UltraSegmDetector]:
        """#### Load a YOLO model and return detectors.

        #### Args:
            - `model_name` (str): The name of the YOLO model to load.

        #### Returns:
            - `Tuple[UltraBBoxDetector, UltraSegmDetector]`: The bounding box and segment detectors.
        """
        model = load_yolo("./_internal/yolos/" + model_name)
        return UltraBBoxDetector(model), UltraSegmDetector(model)


class BboxDetectorForEach:
    """#### Class to detect bounding boxes for each segment."""

    def doit(
        self,
        bbox_detector: UltraBBoxDetector,
        image: torch.Tensor,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        labels: Optional[str] = None,
        detailer_hook: Optional[callable] = None,
    ) -> Tuple[Tuple[int, int], List[SEG]]:
        """#### Detect bounding boxes for each segment in an image.

        #### Args:
            - `bbox_detector` (UltraBBoxDetector): The bounding box detector.
            - `image` (torch.Tensor): The input image tensor.
            - `threshold` (float): The detection threshold.
            - `dilation` (int): The dilation factor for masks.
            - `crop_factor` (float): The crop factor for bounding boxes.
            - `drop_size` (int): The minimum size of bounding boxes to keep.
            - `labels` (str, optional): The labels to filter. Defaults to None.
            - `detailer_hook` (callable, optional): A hook function for additional processing. Defaults to None.

        #### Returns:
            - `Tuple[Tuple[int, int], List[SEG]]`: The shape of the image and a list of detected segments.
        """
        segs = bbox_detector.detect(
            image, threshold, dilation, crop_factor, drop_size, detailer_hook
        )

        if labels is not None and labels != "":
            labels = labels.split(",")
            if len(labels) > 0:
                segs, _ = SEGSLabelFilter.filter(segs, labels)

        return segs


class WildcardChooser:
    """#### Class to choose wildcards for segments."""

    def __init__(self, items: List[Tuple[None, str]], randomize_when_exhaust: bool):
        """#### Initialize the WildcardChooser.

        #### Args:
            - `items` (List[Tuple[None, str]]): The list of items to choose from.
            - `randomize_when_exhaust` (bool): Whether to randomize when the list is exhausted.
        """
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg: SEG) -> Tuple[None, str]:
        """#### Get the next item from the list.

        #### Args:
            - `seg` (SEG): The segment.

        #### Returns:
            - `Tuple[None, str]`: The next item from the list.
        """
        item = self.items[self.i]
        self.i += 1

        return item


def process_wildcard_for_segs(wildcard: str) -> Tuple[None, WildcardChooser]:
    """#### Process a wildcard for segments.

    #### Args:
        - `wildcard` (str): The wildcard.

    #### Returns:
        - `Tuple[None, WildcardChooser]`: The processed wildcard and a WildcardChooser.
    """
    return None, WildcardChooser([(None, wildcard)], False)


import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn



class DiagonalGaussianDistribution(object):
    """#### Represents a diagonal Gaussian distribution parameterized by mean and log-variance.

    #### Attributes:
        - `parameters` (torch.Tensor): The concatenated mean and log-variance of the distribution.
        - `mean` (torch.Tensor): The mean of the distribution.
        - `logvar` (torch.Tensor): The log-variance of the distribution, clamped between -30.0 and 20.0.
        - `std` (torch.Tensor): The standard deviation of the distribution, computed as exp(0.5 * logvar).
        - `var` (torch.Tensor): The variance of the distribution, computed as exp(logvar).
        - `deterministic` (bool): If True, the distribution is deterministic.

    #### Methods:
        - `sample() -> torch.Tensor`:
            Samples from the distribution using the reparameterization trick.
        - `kl(other: DiagonalGaussianDistribution = None) -> torch.Tensor`:
            Computes the Kullback-Leibler divergence between this distribution and a standard normal distribution.
            If `other` is provided, computes the KL divergence between this distribution and `other`.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        """#### Samples from the distribution using the reparameterization trick.

        #### Returns:
            - `torch.Tensor`: A sample from the distribution.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """#### Computes the Kullback-Leibler divergence between this distribution and a standard normal distribution.

        If `other` is provided, computes the KL divergence between this distribution and `other`.

        #### Args:
            - `other` (DiagonalGaussianDistribution, optional): Another distribution to compute the KL divergence with.

        #### Returns:
            - `torch.Tensor`: The KL divergence.
        """
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )


class DiagonalGaussianRegularizer(torch.nn.Module):
    """#### Regularizer for diagonal Gaussian distributions."""

    def __init__(self, sample: bool = True):
        """#### Initialize the regularizer.

        #### Args:
            - `sample` (bool, optional): Whether to sample from the distribution. Defaults to True.
        """
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """#### Forward pass for the regularizer.

        #### Args:
            - `z` (torch.Tensor): The input tensor.

        #### Returns:
            - `Tuple[torch.Tensor, dict]`: The regularized tensor and a log dictionary.
        """
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AutoencodingEngine(nn.Module):
    """#### Class representing an autoencoding engine."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, regularizer: nn.Module, flux: bool = False):
        """#### Initialize the autoencoding engine.

        #### Args:
            - `encoder` (nn.Module): The encoder module.
            - `decoder` (nn.Module): The decoder module.
            - `regularizer` (nn.Module): The regularizer module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularization = regularizer
        if not flux:
            self.post_quant_conv = disable_weight_init.Conv2d(4, 4, 1)
            self.quant_conv = disable_weight_init.Conv2d(8, 8, 1)
        
    def get_last_layer(self):
        return self.decoder.get_last_layer()
    
    def decode(self, z: torch.Tensor, flux:bool = False, **kwargs) -> torch.Tensor:
        """#### Decode the latent tensor.

        #### Args:
            - `z` (torch.Tensor): The latent tensor.
            - `decoder_kwargs` (dict): Additional arguments for the decoder.

        #### Returns:
            - `torch.Tensor`: The decoded tensor.
        """
        if flux:
            x = self.decoder(z, **kwargs)
            return x
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **kwargs)
        return dec


    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        flux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """#### Encode the input tensor.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `return_reg_log` (bool, optional): Whether to return the regularization log. Defaults to False.

        #### Returns:
            - `Union[torch.Tensor, Tuple[torch.Tensor, dict]]`: The encoded tensor and optionally the regularization log.
        """
        z = self.encoder(x)
        if not flux:
            z = self.quant_conv(z)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

ops = disable_weight_init

if xformers_enabled_vae():
    pass


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """#### Apply the swish nonlinearity.

    #### Args:
        - `x` (torch.Tensor): The input tensor.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    """#### Class representing an upsample layer."""

    def __init__(self, in_channels: int, with_conv: bool):
        """#### Initialize the upsample layer.

        #### Args:
            - `in_channels` (int): The number of input channels.
            - `with_conv` (bool): Whether to use convolution.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the upsample layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """#### Class representing a downsample layer."""

    def __init__(self, in_channels: int, with_conv: bool):
        """#### Initialize the downsample layer.

        #### Args:
            - `in_channels` (int): The number of input channels.
            - `with_conv` (bool): Whether to use convolution.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the downsample layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """#### Class representing an encoder."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignore_kwargs,
    ):
        """#### Initialize the encoder.

        #### Args:
            - `ch` (int): The base number of channels.
            - `out_ch` (int): The number of output channels.
            - `ch_mult` (Tuple[int, ...], optional): Channel multiplier at each resolution. Defaults to (1, 2, 4, 8).
            - `num_res_blocks` (int): The number of residual blocks.
            - `attn_resolutions` (Tuple[int, ...]): The resolutions at which to apply attention.
            - `dropout` (float, optional): The dropout rate. Defaults to 0.0.
            - `resamp_with_conv` (bool, optional): Whether to use convolution for resampling. Defaults to True.
            - `in_channels` (int): The number of input channels.
            - `resolution` (int): The resolution of the input.
            - `z_channels` (int): The number of latent channels.
            - `double_z` (bool, optional): Whether to double the latent channels. Defaults to True.
            - `use_linear_attn` (bool, optional): Whether to use linear attention. Defaults to False.
            - `attn_type` (str, optional): The type of attention. Defaults to "vanilla".
        """
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = ops.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def to(self, device=None, dtype=None):
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return super().to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the encoder.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The encoded tensor.
        """
        if x.device != self._device or x.dtype != self._dtype:
            self.to(device=x.device, dtype=x.dtype)
        # timestep embedding
        temb = None
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """#### Class representing a decoder."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        conv_out_op: nn.Module = ops.Conv2d,
        resnet_op: nn.Module = ResnetBlock,
        attn_op: nn.Module = AttnBlock,
        **ignorekwargs,
    ):
        """#### Initialize the decoder.

        #### Args:
            - `ch` (int): The base number of channels.
            - `out_ch` (int): The number of output channels.
            - `ch_mult` (Tuple[int, ...], optional): Channel multiplier at each resolution. Defaults to (1, 2, 4, 8).
            - `num_res_blocks` (int): The number of residual blocks.
            - `attn_resolutions` (Tuple[int, ...]): The resolutions at which to apply attention.
            - `dropout` (float, optional): The dropout rate. Defaults to 0.0.
            - `resamp_with_conv` (bool, optional): Whether to use convolution for resampling. Defaults to True.
            - `in_channels` (int): The number of input channels.
            - `resolution` (int): The resolution of the input.
            - `z_channels` (int): The number of latent channels.
            - `give_pre_end` (bool, optional): Whether to give pre-end. Defaults to False.
            - `tanh_out` (bool, optional): Whether to use tanh activation at the output. Defaults to False.
            - `use_linear_attn` (bool, optional): Whether to use linear attention. Defaults to False.
            - `conv_out_op` (nn.Module, optional): The convolution output operation. Defaults to ops.Conv2d.
            - `resnet_op` (nn.Module, optional): The residual block operation. Defaults to ResnetBlock.
            - `attn_op` (nn.Module, optional): The attention block operation. Defaults to AttnBlock.
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = ops.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resnet_op(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """#### Forward pass for the decoder.

        #### Args:
            - `z` (torch.Tensor): The input tensor.
            - `**kwargs`: Additional arguments.

        #### Returns:
            - `torch.Tensor`: The output tensor.

        """
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        return h


class VAE:
    """#### Class representing a Variational Autoencoder (VAE)."""

    def __init__(
        self,
        sd: Optional[dict] = None,
        device: Optional[torch.device] = None,
        config: Optional[dict] = None,
        dtype: Optional[torch.dtype] = None,
        flux: Optional[bool] = False,
    ):
        """#### Initialize the VAE.

        #### Args:
            - `sd` (dict, optional): The state dictionary. Defaults to None.
            - `device` (torch.device, optional): The device to use. Defaults to None.
            - `config` (dict, optional): The configuration dictionary. Defaults to None.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
        """
        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * dtype_size(
            dtype
        )  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        self.working_dtypes = [torch.bfloat16, torch.float32]

        if config is None:
            if "decoder.conv_in.weight" in sd:
                # default SD1.x/SD2.x VAE parameters
                ddconfig = {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                }

                if (
                    "encoder.down.2.downsample.conv.weight" not in sd
                    and "decoder.up.3.upsample.conv.weight" not in sd
                ):  # Stable diffusion x4 upscaler VAE
                    ddconfig["ch_mult"] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig["z_channels"] = sd[
                    "decoder.conv_in.weight"
                ].shape[1]
                # Initialize model
                self.first_stage_model = AutoencodingEngine(
                    Encoder(**ddconfig),
                    Decoder(**ddconfig), 
                    DiagonalGaussianRegularizer(),
                    flux=flux
                )
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

        if device is None:
            device = vae_device()
        self.device = device
        offload_device = vae_offload_device()
        if dtype is None:
            dtype = vae_dtype()
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )
        logging.debug(
            "VAE load device: {}, offload device: {}, dtype: {}".format(
                self.device, offload_device, self.vae_dtype
            )
        )


    def vae_encode_crop_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """#### Crop the input pixels to be compatible with the VAE.

        #### Args:
            - `pixels` (torch.Tensor): The input pixel tensor.

        #### Returns:
            - `torch.Tensor`: The cropped pixel tensor.
        """
        (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        return pixels

    def decode(self, samples_in: torch.Tensor, flux:bool = False) -> torch.Tensor:
        """#### Decode the latent samples to pixel samples.

        #### Args:
            - `samples_in` (torch.Tensor): The input latent samples.

        #### Returns:
            - `torch.Tensor`: The decoded pixel samples.
        """
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        pixel_samples = torch.empty(
            (
                samples_in.shape[0],
                3,
                round(samples_in.shape[2] * self.upscale_ratio),
                round(samples_in.shape[3] * self.upscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, samples_in.shape[0], batch_number):
            samples = (
                samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
            )
            pixel_samples[x : x + batch_number] = self.process_output(
                self.first_stage_model.decode(samples, flux=flux).to(self.output_device).float()
            )
        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples


    def encode(self, pixel_samples: torch.Tensor, flux:bool = False) -> torch.Tensor:
        """#### Encode the pixel samples to latent samples.

        #### Args:
            - `pixel_samples` (torch.Tensor): The input pixel samples.

        #### Returns:
            - `torch.Tensor`: The encoded latent samples.
        """
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)
        samples = torch.empty(
            (
                pixel_samples.shape[0],
                self.latent_channels,
                round(pixel_samples.shape[2] // self.downscale_ratio),
                round(pixel_samples.shape[3] // self.downscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = (
                self.process_input(pixel_samples[x : x + batch_number])
                .to(self.vae_dtype)
                .to(self.device)
            )
            samples[x : x + batch_number] = (
                self.first_stage_model.encode(pixels_in, flux=flux).to(self.output_device).float()
            )

        return samples

    def get_sd(self):
        return self.first_stage_model.state_dict()


class VAEDecode:
    """#### Class for decoding VAE samples."""

    def decode(self, vae: VAE, samples: dict, flux:bool = False) -> Tuple[torch.Tensor]:
        """#### Decode the VAE samples.

        #### Args:
            - `vae` (VAE): The VAE instance.
            - `samples` (dict): The samples dictionary.

        #### Returns:
            - `Tuple[torch.Tensor]`: The decoded samples.
        """
        return (vae.decode(samples["samples"], flux=flux),)


class VAEEncode:
    """#### Class for encoding VAE samples."""

    def encode(self, vae: VAE, pixels: torch.Tensor, flux:bool = False) -> Tuple[dict]:
        """#### Encode the VAE samples.

        #### Args:
            - `vae` (VAE): The VAE instance.
            - `pixels` (torch.Tensor): The input pixel tensor.

        #### Returns:
            - `Tuple[dict]`: The encoded samples dictionary.
        """
        t = vae.encode(pixels[:, :, :, :3], flux=flux)
        return ({"samples": t},)


class VAELoader:
    # TODO: scale factor?
    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = "./_internal/vae/" + vae_name
            sd = load_torch_file(vae_path)
        vae = VAE(sd=sd)
        return (vae,)




from enum import Enum
import logging
import torch



class CLIPAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPAttention module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.k_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.v_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

        self.out_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        optimized_attention: callable = None,
    ) -> torch.Tensor:
        """#### Forward pass for the CLIPAttention module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `optimized_attention` (callable, optional): The optimized attention function. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}


class CLIPMLP(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPMLP module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `intermediate_size` (int): The intermediate size.
            - `activation` (str): The activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.fc1 = operations.Linear(
            embed_dim, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(
            intermediate_size, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the CLIPMLP module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        intermediate_size: int,
        intermediate_activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPLayer module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `intermediate_size` (int): The intermediate size.
            - `intermediate_activation` (str): The intermediate activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(
            embed_dim,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        optimized_attention: callable = None,
    ) -> torch.Tensor:
        """#### Forward pass for the CLIPLayer module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `optimized_attention` (callable, optional): The optimized attention function. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        heads: int,
        intermediate_size: int,
        intermediate_activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPEncoder module.

        #### Args:
            - `num_layers` (int): The number of layers.
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `intermediate_size` (int): The intermediate size.
            - `intermediate_activation` (str): The intermediate activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                CLIPLayer(
                    embed_dim,
                    heads,
                    intermediate_size,
                    intermediate_activation,
                    dtype,
                    device,
                    operations,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        intermediate_output: int = None,
    ) -> tuple:
        """#### Forward pass for the CLIPEncoder module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `intermediate_output` (int, optional): The intermediate output layer. Defaults to None.

        #### Returns:
            - `tuple`: The output tensor and the intermediate output tensor.
        """
        optimized_attention = optimized_attention_for_device()

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, length in enumerate(self.layers):
            x = length(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = 49408,
        num_positions: int = 77,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: object = torch.nn,
    ):
        """#### Initialize the CLIPEmbeddings module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `vocab_size` (int, optional): The vocabulary size. Defaults to 49408.
            - `num_positions` (int, optional): The number of positions. Defaults to 77.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device to use. Defaults to None.
        """
        super().__init__()
        self.token_embedding = operations.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = operations.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """#### Forward pass for the CLIPEmbeddings module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.
            - `dtype` (torch.dtype, optional): The data type. Defaults to torch.float32.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return self.token_embedding(input_tokens, out_dtype=dtype) + cast_to(
            self.position_embedding.weight, dtype=dtype, device=input_tokens.device
        )



class CLIP:
    def __init__(
        self,
        target: object = None,
        embedding_directory: str = None,
        no_init: bool = False,
        tokenizer_data={},
        parameters=0,
        model_options={},
    ):
        """#### Initialize the CLIP class.

        #### Args:
            - `target` (object, optional): The target object. Defaults to None.
            - `embedding_directory` (str, optional): The embedding directory. Defaults to None.
            - `no_init` (bool, optional): Whether to skip initialization. Defaults to False.
        """
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", text_encoder_device())
        offload_device = model_options.get(
            "offload_device", text_encoder_offload_device()
        )
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = text_encoder_dtype(load_device)

        params["dtype"] = dtype
        params["device"] = model_options.get(
            "initial_device",
            text_encoder_initial_device(
                load_device, offload_device, parameters * dtype_size(dtype)
            ),
        )
        params["model_options"] = model_options

        self.cond_stage_model = clip(**(params))

        # for dt in self.cond_stage_model.dtypes:
        #     if not supports_cast(load_device, dt):
        #         load_device = offload_device
        #         if params["device"] != offload_device:
        #             self.cond_stage_model.to(offload_device)
        #             logging.warning("Had to shift TE back.")

        try:
            self.tokenizer = tokenizer(
                embedding_directory=embedding_directory, tokenizer_data=tokenizer_data
            )
        except TypeError:
            self.tokenizer = tokenizer(
                embedding_directory=embedding_directory
            )
        self.patcher = ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        if params["device"] == load_device:
            load_models_gpu([self.patcher], force_full_load=True, flux_enabled=True)
        self.layer_idx = None
        logging.debug(
            "CLIP model load device: {}, offload device: {}, current: {}".format(
                load_device, offload_device, params["device"]
            )
        )

    def clone(self) -> "CLIP":
        """#### Clone the CLIP object.

        #### Returns:
            - `CLIP`: The cloned CLIP object.
        """
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(
        self, patches: list, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> None:
        """#### Add patches to the model.

        #### Args:
            - `patches` (list): The patches to add.
            - `strength_patch` (float, optional): The strength of the patches. Defaults to 1.0.
            - `strength_model` (float, optional): The strength of the model. Defaults to 1.0.
        """
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx: int) -> None:
        """#### Set the clip layer.

        #### Args:
            - `layer_idx` (int): The layer index.
        """
        self.layer_idx = layer_idx

    def tokenize(self, text: str, return_word_ids: bool = False) -> list:
        """#### Tokenize the input text.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `list`: The tokenized text.
        """
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens: list, return_pooled: bool = False, return_dict: bool = False, flux_enabled:bool = False) -> tuple:
        """#### Encode the input tokens.

        #### Args:
            - `tokens` (list): The input tokens.
            - `return_pooled` (bool, optional): Whether to return the pooled output. Defaults to False.
            - `flux_enabled` (bool, optional): Whether to enable flux. Defaults to False.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model(flux_enabled=flux_enabled)
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def load_sd(self, sd: dict, full_model: bool = False) -> None:
        """#### Load the state dictionary.

        #### Args:
            - `sd` (dict): The state dictionary.
            - `full_model` (bool, optional): Whether to load the full model. Defaults to False.
        """
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def load_model(self, flux_enabled:bool = False) -> ModelPatcher:
        """#### Load the model.

        #### Returns:
            - `ModelPatcher`: The model patcher.
        """
        load_model_gpu(self.patcher, flux_enabled=flux_enabled)
        return self.patcher
    
    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    SD3 = 3
    FLUX = 6

def load_text_encoder_state_dicts(
    state_dicts=[],
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "text_projection" in clip_data[i]:
            clip_data[i]["text_projection.weight"] = clip_data[i][
                "text_projection"
            ].transpose(
                0, 1
            )  # old models saved with the CLIPSave node

    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 2:
        if clip_type == CLIPType.FLUX:
            weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
            weight = clip_data[0].get(weight_name, clip_data[1].get(weight_name, None))
            dtype_t5 = None
            if weight is not None:
                dtype_t5 = weight.dtype

            clip_target.clip = flux_clip(dtype_t5=dtype_t5)
            clip_target.tokenizer = FluxTokenizer

    parameters = 0
    tokenizer_data = {}
    for c in clip_data:
        parameters += calculate_parameters(c)
        tokenizer_data, model_options = model_options_long_clip(
            c, tokenizer_data, model_options
        )

    clip = CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        tokenizer_data=tokenizer_data,
        model_options=model_options,
    )
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip

class CLIPTextEncode:
    def encode(self, clip: CLIP, text: str, flux_enabled: bool = False) -> tuple:
        """#### Encode the input text.

        #### Args:
            - `clip` (CLIP): The CLIP object.
            - `text` (str): The input text.
            - `flux_enabled` (bool, optional): Whether to enable flux. Defaults to False.

        #### Returns:
            - `tuple`: The encoded text and the pooled output.
        """
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True, flux_enabled=flux_enabled)
        return ([[cond, {"pooled_output": pooled}]],)


class CLIPSetLastLayer:
    def set_last_layer(self, clip: CLIP, stop_at_clip_layer: int) -> tuple:
        """#### Set the last layer of the CLIP model.

        #### Args:
            - `clip` (CLIP): The CLIP object.
            - `stop_at_clip_layer` (int): The layer to stop at.

        #### Returns:
            - `tuple`: Thefrom enum import Enum
        """
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)


class ClipTarget:
    """#### Target class for the CLIP model."""

    def __init__(self, tokenizer: object, clip: object):
        """#### Initialize the ClipTarget class.

        #### Args:
            - `tokenizer` (object): The tokenizer.
            - `clip` (object): The CLIP model.
        """
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}


import torch

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora: dict, to_load: dict) -> dict:
    """#### Load a LoRA model.

    #### Args:
        - `lora` (dict): The LoRA model state dictionary.
        - `to_load` (dict): The keys to load from the LoRA model.

    #### Returns:
        - `dict`: The loaded LoRA model.
    """
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        "{}.dora_scale".format(x)
        dora_scale = None

        regular_lora = "{}.lora_up.weight".format(x)
        "{}_lora.up.weight".format(x)
        "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            "{}.lora_mid.weight".format(x)

        if A_name is not None:
            mid = None
            patch_dict[to_load[x]] = (
                "lora",
                (lora[A_name], lora[B_name], alpha, mid, dora_scale),
            )
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
    return patch_dict


def model_lora_keys_clip(model: torch.nn.Module, key_map: dict = {}) -> dict:
    """#### Get the keys for a LoRA model's CLIP component.

    #### Args:
        - `model` (torch.nn.Module): The LoRA model.
        - `key_map` (dict, optional): The key map. Defaults to {}.

    #### Returns:
        - `dict`: The keys for the CLIP component.
    """
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(
                    b, LORA_CLIP_MAP[c]
                )  # SDXL base
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                    b, c
                )  # diffusers lora
                key_map[lora_key] = k
    return key_map


def model_lora_keys_unet(model: torch.nn.Module, key_map: dict = {}) -> dict:
    """#### Get the keys for a LoRA model's UNet component.

    #### Args:
        - `model` (torch.nn.Module): The LoRA model.
        - `key_map` (dict, optional): The key map. Defaults to {}.

    #### Returns:
        - `dict`: The keys for the UNet component.
    """
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = k  # cascade lora:

    diffusers_keys = unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", ""]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(
                    p, k[: -len(".weight")].replace(".to_", ".processor.to_")
                )
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key
    return key_map


def load_lora_for_models(
    model: object, clip: object, lora: dict, strength_model: float, strength_clip: float
) -> tuple:
    """#### Load a LoRA model for the given models.

    #### Args:
        - `model` (object): The model.
        - `clip` (object): The CLIP model.
        - `lora` (dict): The LoRA model state dictionary.
        - `strength_model` (float): The strength of the model.
        - `strength_clip` (float): The strength of the CLIP model.

    #### Returns:
        - `tuple`: The new model patcher and CLIP model.
    """
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = load_lora(lora, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)

    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)

    return (new_modelpatcher, new_clip)


class LoraLoader:
    """#### Class for loading LoRA models."""

    def __init__(self):
        """#### Initialize the LoraLoader class."""
        self.loaded_lora = None

    def load_lora(
        self,
        model: object,
        clip: object,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
    ) -> tuple:
        """#### Load a LoRA model.

        #### Args:
            - `model` (object): The model.
            - `clip` (object): The CLIP model.
            - `lora_name` (str): The name of the LoRA model.
            - `strength_model` (float): The strength of the model.
            - `strength_clip` (float): The strength of the CLIP model.

        #### Returns:
            - `tuple`: The new model patcher and CLIP model.
        """
        lora_path = get_full_path("loras", lora_name)
        lora = None
        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


# Original code can be found on: https://github.com/black-forest-labs/flux


from dataclasses import dataclass
from einops import rearrange, repeat
import torch
import torch.nn as nn



def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    if is_device_mps(pos.device) or is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device
    )
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega
    )
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.in_layer = operations.Linear(
            in_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )
        self.silu = nn.SiLU()
        self.out_layer = operations.Linear(
            hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.scale, 1e-6)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = RMSNorm(
            dim, dtype=dtype, device=device, operations=operations
        )
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = operations.Linear(
            dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device
        )
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(
        self, dim: int, double: bool, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(
            dim, self.multiplier * dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, vec: torch.Tensor) -> tuple:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(
            hidden_size, double=True, dtype=dtype, device=device, operations=operations
        )
        self.img_norm1 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.img_norm2 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.img_mlp = nn.Sequential(
            operations.Linear(
                hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device
            ),
            nn.GELU(approximate="tanh"),
            operations.Linear(
                mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

        self.txt_mod = Modulation(
            hidden_size, double=True, dtype=dtype, device=device, operations=operations
        )
        self.txt_norm1 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.txt_norm2 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.txt_mlp = nn.Sequential(
            operations.Linear(
                hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device
            ),
            nn.GELU(approximate="tanh"),
            operations.Linear(
                mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim,
            dtype=dtype,
            device=device,
        )
        # proj and mlp_out
        self.linear2 = operations.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device
        )

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(
            hidden_size, double=False, dtype=dtype, device=device, operations=operations
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.linear = operations.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if (
        padding_mode == "circular"
        and torch.jit.is_tracing()
        or torch.jit.is_scripting()
    ):
        padding_mode = "reflect"
    pad_h = (patch_size[0] - img.shape[-2] % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - img.shape[-1] % patch_size[1]) % patch_size[1]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)


try:
    rms_norm_torch = torch.nn.functional.rms_norm
except:
    rms_norm_torch = None


def rms_norm(x, weight, eps=1e-6):
    if rms_norm_torch is not None and not (
        torch.jit.is_tracing() or torch.jit.is_scripting()
    ):
        return rms_norm_torch(
            x,
            weight.shape,
            weight=cast_to(weight, dtype=x.dtype, device=x.device),
            eps=eps,
        )
    else:
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x * rrms) * cast_to(weight, dtype=x.dtype, device=x.device)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux3(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        image_model=None,
        final_layer=True,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = operations.Linear(
            self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.vector_in = MLPEmbedder(
            params.vec_in_dim,
            self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = operations.Linear(
            params.context_in_dim, self.hidden_size, dtype=dtype, device=device
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(
                self.hidden_size,
                1,
                self.out_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
    ) -> torch.Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding_flux(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                timestep_embedding_flux(guidance, 256).to(img.dtype)
            )

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(
            x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
        )

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[
                :, None
            ]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[
                None, :
            ]
        )
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control
        )
        return rearrange(
            out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2
        )[:, :, :h, :w]


class Flux2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=Flux3, flux=True)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = CONDRegular(cross_attn)
        out["guidance"] = CONDRegular(torch.FloatTensor([kwargs.get("guidance", 3.5)]))
        return out


class Flux(BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {}

    unet_extra_config = {}
    latent_format = Flux1

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = Flux2(self, device=device)
        return out


models = [Flux]

import logging as logger
import torch
from PIL import Image



def load_state_dict(state_dict: dict) -> PyTorchModel:
    """#### Load a state dictionary into a PyTorch model.

    #### Args:
        - `state_dict` (dict): The state dictionary.

    #### Returns:
        - `PyTorchModel`: The loaded PyTorch model.
    """
    logger.debug("Loading state dict into pytorch model arch")
    state_dict_keys = list(state_dict.keys())
    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    model = RRDBNet(state_dict)
    return model


class UpscaleModelLoader:
    """#### Class for loading upscale models."""

    def load_model(self, model_name: str) -> tuple:
        """#### Load an upscale model.

        #### Args:
            - `model_name` (str): The name of the model.

        #### Returns:
            - `tuple`: The loaded model.
        """
        model_path = f"_internal/ESRGAN/{model_name}"
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)


class ImageUpscaleWithModel:
    """#### Class for upscaling images with a model."""

    def upscale(self, upscale_model: torch.nn.Module, image: torch.Tensor) -> tuple:
        """#### Upscale an image using a model.

        #### Args:
            - `upscale_model` (torch.nn.Module): The upscale model.
            - `image` (torch.Tensor): The input image tensor.

        #### Returns:
            - `tuple`: The upscaled image tensor.
        """
        device = torch.device(torch.cuda.current_device())
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        get_free_memory(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            steps = in_img.shape[0] * get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = ProgressBar(steps)
            s = tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


def torch_gc() -> None:
    """#### Perform garbage collection for PyTorch."""
    pass


class Script:
    """#### Class representing a script."""
    pass


class Options:
    """#### Class representing options."""

    img2img_background_color: str = "#ffffff"  # Set to white for now


class State:
    """#### Class representing the state."""

    interrupted: bool = False

    def begin(self) -> None:
        """#### Begin the state."""
        pass

    def end(self) -> None:
        """#### End the state."""
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
actual_upscaler = None

# Batch of images to upscale
batch = None


if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:
    """#### Class for upscaling images."""

    def _upscale(self, img: Image.Image, scale: float) -> Image.Image:
        """#### Upscale an image.

        #### Args:
            - `img` (Image.Image): The input image.
            - `scale` (float): The scale factor.

        #### Returns:
            - `Image.Image`: The upscaled image.
        """
        global actual_upscaler
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image.Image, scale: float, selected_model: str = None) -> Image.Image:
        """#### Upscale an image with a selected model.

        #### Args:
            - `img` (Image.Image): The input image.
            - `scale` (float): The scale factor.
            - `selected_model` (str, optional): The selected model. Defaults to None.

        #### Returns:
            - `Image.Image`: The upscaled image.
        """
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    """#### Class for storing upscaler data."""

    name: str = ""
    data_path: str = ""

    def __init__(self):
        self.scaler = Upscaler()

from dataclasses import dataclass
import logging
import torch
import functools



logger = logging.getLogger()

try:
    from sfast.compilers.diffusion_pipeline_compiler import (
        _enable_xformers,
        _modify_model,
    )
    from sfast.cuda.graphs import make_dynamic_graphed_callable
    from sfast.jit import utils as jit_utils
    from sfast.jit.trace_helper import trace_with_kwargs
except ImportError:
    pass


@dataclass
class TracedModuleCacheItem:
    """#### Data class for storing traced module cache items.

    #### Attributes:
        - `module` (object): The traced module.
        - `patch_id` (int): The patch ID.
        - `device` (str): The device.
    """
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    """#### Class for lazy tracing of modules."""

    traced_modules: dict = {}

    def __init__(self, config: object = None, patch_id: int = None, **kwargs) -> None:
        """#### Initialize the LazyTraceModule.

        #### Args:
            - `config` (object, optional): The configuration object. Defaults to None.
            - `patch_id` (int, optional): The patch ID. Defaults to None.
            - `**kwargs`: Additional keyword arguments.
        """
        self.config = config
        self.patch_id = patch_id
        self.kwargs = kwargs
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )
        self.cuda_graph_modules = {}

    def ts_compiler(self, m: torch.nn.Module) -> torch.nn.Module:
        """#### TorchScript compiler for the module.

        #### Args:
            - `m` (torch.nn.Module): The module to compile.

        #### Returns:
            - `torch.nn.Module`: The compiled module.
        """
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function: callable, **kwargs) -> callable:
        """#### Call the LazyTraceModule.

        #### Args:
            - `model_function` (callable): The model function.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `callable`: The traced module.
        """
        module_factory = BaseModelApplyModelModuleFactory(
            model_function, kwargs
        )
        kwargs = module_factory.get_converted_kwargs()
        key = module_factory.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                logger.info(
                    f'Tracing {getattr(m_model, "__name__", m_model.__class__.__name__)}'
                )
                traced_m, call_helper = trace_with_kwargs(
                    m_model, None, m_kwargs, **self.kwargs
                )

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            self.cuda_graph_modules[key] = traced_module

        return traced_module(**kwargs)


def build_lazy_trace_module(config: object, device: torch.device, patch_id: int) -> LazyTraceModule:
    """#### Build a LazyTraceModule.

    #### Args:
        - `config` (object): The configuration object.
        - `device` (torch.device): The device.
        - `patch_id` (int): The patch ID.

    #### Returns:
        - `LazyTraceModule`: The LazyTraceModule instance.
    """
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )


import contextlib
import importlib
import itertools
import logging
import math
import sys
from functools import partial
from typing import TYPE_CHECKING, Callable, NamedTuple

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


class ModelType1(StrEnum):
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


def guess_model_type(model: object) -> ModelType1 | None:
    """#### Guess the model type.

    #### Args:
        - `model` (object): The model object.

    #### Returns:
        - `Optional[ModelType1]`: The guessed model type.
    """
    latent_format = model.get_model_object("latent_format")
    if isinstance(latent_format, SD15):
        return ModelType1.SD15
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
        return bislerp(samples, width, height)
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

import glob


def CheckAndDownload():
    """#### Check and download all the necessary safetensors and checkpoints models"""
    if glob.glob("./_internal/checkpoints/*.safetensors") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="Meina/MeinaMix",
            filename="Meina V10 - baked VAE.safetensors",
            local_dir="./_internal/checkpoints/",
        )
    if glob.glob("./_internal/yolos/*.pt") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="hand_yolov9c.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="face_yolov9c.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="person_yolov8m-seg.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="segments-arnaud/sam_vit_b",
            filename="sam_vit_b_01ec64.pth",
            local_dir="./_internal/yolos/",
        )
    if glob.glob("./_internal/ESRGAN/*.pth") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="lllyasviel/Annotators",
            filename="RealESRGAN_x4plus.pth",
            local_dir="./_internal/ESRGAN/",
        )
    if glob.glob("./_internal/loras/*.safetensors") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="EvilEngine/add_detail",
            filename="add_detail.safetensors",
            local_dir="./_internal/loras/",
        )
    if glob.glob("./_internal/embeddings/*.pt") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="EvilEngine/badhandv4",
            filename="badhandv4.pt",
            local_dir="./_internal/embeddings/",
        )
        # hf_hub_download(
        #     repo_id="segments-arnaud/sam_vit_b",
        #     filename="EasyNegative.safetensors",
        #     local_dir="./_internal/embeddings/",
        # )
    if glob.glob("./_internal/vae_approx/*.pth") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="madebyollin/taesd",
            filename="taesd_decoder.safetensors",
            local_dir="./_internal/vae_approx/",
        )

def CheckAndDownloadFlux():
    if glob.glob("./_internal/embeddings/*.pt") == []:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="EvilEngine/badhandv4",
            filename="badhandv4.pt",
            local_dir="./_internal/embeddings",
        )
    if glob.glob("./_internal/unet/*.gguf") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q8_0.gguf",
            local_dir="./_internal/unet",
        )
    if glob.glob("./_internal/clip/*.gguf") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="city96/t5-v1_1-xxl-encoder-gguf",
            filename="t5-v1_1-xxl-encoder-Q8_0.gguf",
            local_dir="./_internal/clip",
        )
        hf_hub_download(
            repo_id="comfyanonymous/flux_text_encoders",
            filename="clip_l.safetensors",
            local_dir="./_internal/clip",
        )
    if glob.glob("./_internal/vae/*.safetensors") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            filename="ae.safetensors",
            local_dir="./_internal/vae",
        )
        
    if glob.glob("./_internal/vae_approx/*.pth") == []:
        from huggingface_hub import hf_hub_download
        
        hf_hub_download(
            repo_id="madebyollin/taef1",
            filename="diffusion_pytorch_model.safetensors",
            local_dir="./_internal/vae_approx/",
        )

import os
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch



def sam_predict(
    predictor: SamPredictor, points: list, plabs: list, bbox: list, threshold: float
) -> list:
    """#### Predict masks using 

    #### Args:
        - `predictor` (SamPredictor): The SAM predictor.
        - `points` (list): List of points.
        - `plabs` (list): List of point labels.
        - `bbox` (list): Bounding box.
        - `threshold` (float): Threshold for mask selection.

    #### Returns:
        - `list`: List of predicted masks.
    """
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, box=box
    )

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


def is_same_device(a: torch.device, b: torch.device) -> bool:
    """#### Check if two devices are the same.

    #### Args:
        - `a` (torch.device): The first device.
        - `b` (torch.device): The second device.

    #### Returns:
        - `bool`: Whether the devices are the same.
    """
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index


class SafeToGPU:
    """#### Class to safely move objects to GPU."""

    def __init__(self, size: int):
        self.size = size

    def to_device(self, obj: torch.nn.Module, device: torch.device) -> None:
        """#### Move an object to a device.

        #### Args:
            - `obj` (torch.nn.Module): The object to move.
            - `device` (torch.device): The target device.
        """
        if is_same_device(device, "cpu"):
            obj.to(device)
        else:
            if is_same_device(obj.device, "cpu"):  # cpu to gpu
                free_memory(self.size * 1.3, device)
                if get_free_memory(device) > self.size * 1.3:
                    try:
                        obj.to(device)
                    except:
                        print(
                            f"WARN: The model is not moved to the '{device}' due to insufficient memory. [1]"
                        )
                else:
                    print(
                        f"WARN: The model is not moved to the '{device}' due to insufficient memory. [2]"
                    )


class SAMWrapper:
    """#### Wrapper class for SAM model."""

    def __init__(
        self, model: torch.nn.Module, is_auto_mode: bool, safe_to_gpu: SafeToGPU = None
    ):
        self.model = model
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self) -> None:
        """#### Prepare the device for the model."""
        if self.is_auto_mode:
            device = get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self) -> None:
        """#### Release the device from the model."""
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(
        self, image: np.ndarray, points: list, plabs: list, bbox: list, threshold: float
    ) -> list:
        """#### Predict masks using the SAM model.

        #### Args:
            - `image` (np.ndarray): The input image.
            - `points` (list): List of points.
            - `plabs` (list): List of point labels.
            - `bbox` (list): Bounding box.
            - `threshold` (float): Threshold for mask selection.

        #### Returns:
            - `list`: List of predicted masks.
        """
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)


class SAMLoader:
    """#### Class to load SAM models."""

    def load_model(self, model_name: str, device_mode: str = "auto") -> tuple:
        """#### Load a SAM model.

        #### Args:
            - `model_name` (str): The name of the model.
            - `device_mode` (str, optional): The device mode. Defaults to "auto".

        #### Returns:
            - `tuple`: The loaded SAM model.
        """
        modelname = "./_internal/yolos/" + model_name

        if "vit_h" in model_name:
            model_kind = "vit_h"
        elif "vit_l" in model_name:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        size = os.path.getsize(modelname)
        safe_to = SafeToGPU(size)

        # Unless user explicitly wants to use CPU, we use GPU
        device = get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            safe_to.to_device(sam, device)

        is_auto_mode = device_mode == "AUTO"

        sam_obj = SAMWrapper(sam, is_auto_mode=is_auto_mode, safe_to_gpu=safe_to)
        sam.sam_wrapper = sam_obj

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam,)


def make_sam_mask(
    sam: SAMWrapper,
    segs: tuple,
    image: torch.Tensor,
    detection_hint: bool,
    dilation: int,
    threshold: float,
    bbox_expansion: int,
    mask_hint_threshold: float,
    mask_hint_use_negative: bool,
) -> torch.Tensor:
    """#### Create a SAM mask.

    #### Args:
        - `sam` (SAMWrapper): The SAM wrapper.
        - `segs` (tuple): Segmentation information.
        - `image` (torch.Tensor): The input image.
        - `detection_hint` (bool): Whether to use detection hint.
        - `dilation` (int): Dilation value.
        - `threshold` (float): Threshold for mask selection.
        - `bbox_expansion` (int): Bounding box expansion value.
        - `mask_hint_threshold` (float): Mask hint threshold.
        - `mask_hint_use_negative` (bool): Whether to use negative mask hint.

    #### Returns:
        - `torch.Tensor`: The created SAM mask.
    """
    sam_obj = sam.sam_wrapper
    sam_obj.prepare_device()

    try:
        image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []
        # seg_shape = segs[0]
        segs = segs[1]
        for i in range(len(segs)):
            bbox = segs[i].bbox
            center = center_of_bbox(bbox)
            x1 = max(bbox[0] - bbox_expansion, 0)
            y1 = max(bbox[1] - bbox_expansion, 0)
            x2 = min(bbox[2] + bbox_expansion, image.shape[1])
            y2 = min(bbox[3] + bbox_expansion, image.shape[0])
            dilated_bbox = [x1, y1, x2, y2]
            points = []
            plabs = []
            points.append(center)
            plabs = [1]  # 1 = foreground point, 0 = background point
            detected_masks = sam_obj.predict(
                image, points, plabs, dilated_bbox, threshold
            )
            total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)

        mask = make_3d_mask(mask)
        return mask
    else:
        return None


class SAMDetectorCombined:
    """#### Class to combine SAM detection."""

    def doit(
        self,
        sam_model: SAMWrapper,
        segs: tuple,
        image: torch.Tensor,
        detection_hint: bool,
        dilation: int,
        threshold: float,
        bbox_expansion: int,
        mask_hint_threshold: float,
        mask_hint_use_negative: bool,
    ) -> tuple:
        """#### Combine SAM detection.

        #### Args:
            - `sam_model` (SAMWrapper): The SAM wrapper.
            - `segs` (tuple): Segmentation information.
            - `image` (torch.Tensor): The input image.
            - `detection_hint` (bool): Whether to use detection hint.
            - `dilation` (int): Dilation value.
            - `threshold` (float): Threshold for mask selection.
            - `bbox_expansion` (int): Bounding box expansion value.
            - `mask_hint_threshold` (float): Mask hint threshold.
            - `mask_hint_use_negative` (bool): Whether to use negative mask hint.

        #### Returns:
            - `tuple`: The combined SAM detection result.
        """
        sam = make_sam_mask(
            sam_model,
            segs,
            image,
            detection_hint,
            dilation,
            threshold,
            bbox_expansion,
            mask_hint_threshold,
            mask_hint_use_negative,
        )
        if sam is not None:
            return (sam,)
        else:
            return None


import math
import torch
from typing import Any, Dict, Optional, Tuple


# FIXME: Improve slow inference times


class DifferentialDiffusion:
    """#### Class for applying differential diffusion to a model."""

    def apply(self, model: torch.nn.Module) -> Tuple[torch.nn.Module]:
        """#### Apply differential diffusion to a model.

        #### Args:
            - `model` (torch.nn.Module): The input model.

        #### Returns:
            - `Tuple[torch.nn.Module]`: The modified model.
        """
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: Dict[str, Any],
    ) -> torch.Tensor:
        """#### Forward function for differential diffusion.

        #### Args:
            - `sigma` (torch.Tensor): The sigma tensor.
            - `denoise_mask` (torch.Tensor): The denoise mask tensor.
            - `extra_options` (Dict[str, Any]): Additional options.

        #### Returns:
            - `torch.Tensor`: The processed denoise mask tensor.
        """
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


def to_latent_image(pixels: torch.Tensor, vae: VAE) -> torch.Tensor:
    """#### Convert pixels to a latent image using a VAE.

    #### Args:
        - `pixels` (torch.Tensor): The input pixel tensor.
        - `vae` (VAE): The VAE model.

    #### Returns:
        - `torch.Tensor`: The latent image tensor.
    """
    pixels.shape[1]
    pixels.shape[2]
    return VAEEncode().encode(vae, pixels)[0]


def calculate_sigmas2(
    model: torch.nn.Module, sampler: str, scheduler: str, steps: int
) -> torch.Tensor:
    """#### Calculate sigmas for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `sampler` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `steps` (int): The number of steps.

    #### Returns:
        - `torch.Tensor`: The calculated sigmas.
    """
    return calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps
    )


def get_noise_sampler(
    x: torch.Tensor, cpu: bool, total_sigmas: torch.Tensor, **kwargs
) -> Optional[BrownianTreeNoiseSampler]:
    """#### Get a noise sampler.

    #### Args:
        - `x` (torch.Tensor): The input tensor.
        - `cpu` (bool): Whether to use CPU.
        - `total_sigmas` (torch.Tensor): The total sigmas tensor.
        - `kwargs` (dict): Additional arguments.

    #### Returns:
        - `Optional[BrownianTreeNoiseSampler]`: The noise sampler.
    """
    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs["extra_args"].get("seed", None)
        return BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=cpu
        )
    return None


def ksampler2(
    sampler_name: str,
    total_sigmas: torch.Tensor,
    extra_options: Dict[str, Any] = {},
    inpaint_options: Dict[str, Any] = {},
    pipeline: bool = False,
) -> KSAMPLER:
    """#### Get a ksampler.

    #### Args:
        - `sampler_name` (str): The sampler name.
        - `total_sigmas` (torch.Tensor): The total sigmas tensor.
        - `extra_options` (Dict[str, Any], optional): Additional options. Defaults to {}.
        - `inpaint_options` (Dict[str, Any], optional): Inpaint options. Defaults to {}.
        - `pipeline` (bool, optional): Whether to use  Defaults to False.

    #### Returns:
        - `KSAMPLER`: The ksampler.
    """
    if sampler_name == "dpmpp_2m_sde":

        def sample_dpmpp_sde(model, x, sigmas, pipeline, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs["noise_sampler"] = noise_sampler

            return sample_dpmpp_2m_sde(
                model, x, sigmas, pipeline=pipeline, **kwargs
            )

        sampler_function = sample_dpmpp_sde

    else:
        return sampler_object(sampler_name, pipeline=pipeline)

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


class Noise_RandomNoise:
    """#### Class for generating random noise."""

    def __init__(self, seed: int):
        """#### Initialize the Noise_RandomNoise class.

        #### Args:
            - `seed` (int): The seed for random noise.
        """
        self.seed = seed

    def generate_noise(self, input_latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """#### Generate random noise.

        #### Args:
            - `input_latent` (Dict[str, torch.Tensor]): The input latent tensor.

        #### Returns:
            - `torch.Tensor`: The generated noise tensor.
        """
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return prepare_noise(latent_image, self.seed, batch_inds)


def sample_with_custom_noise(
    model: torch.nn.Module,
    add_noise: bool,
    noise_seed: int,
    cfg: int,
    positive: Any,
    negative: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Dict[str, torch.Tensor],
    noise: Optional[torch.Tensor] = None,
    callback: Optional[callable] = None,
    pipeline: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """#### Sample with custom noise.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `add_noise` (bool): Whether to add noise.
        - `noise_seed` (int): The noise seed.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `sampler` (Any): The sampler.
        - `sigmas` (torch.Tensor): The sigmas tensor.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `callback` (Optional[callable], optional): The callback function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use  Defaults to False.

    #### Returns:
        - `Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]`: The sampled and denoised tensors.
    """
    latent = latent_image
    latent_image = latent["samples"]

    out = latent.copy()
    out["samples"] = latent_image

    if noise is None:
        noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    disable_pbar = not PROGRESS_BAR_ENABLED

    device = get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    samples = sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        disable_pbar=disable_pbar,
        seed=noise_seed,
        pipeline=pipeline,
    )

    samples = samples.to(intermediate_device())

    out["samples"] = samples
    out_denoised = out
    return out, out_denoised


def separated_sample(
    model: torch.nn.Module,
    add_noise: bool,
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: Dict[str, torch.Tensor],
    start_at_step: Optional[int],
    end_at_step: Optional[int],
    return_with_leftover_noise: bool,
    sigma_ratio: float = 1.0,
    sampler_opt: Optional[Dict[str, Any]] = None,
    noise: Optional[torch.Tensor] = None,
    callback: Optional[callable] = None,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Dict[str, torch.Tensor]:
    """#### Perform separated 

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `add_noise` (bool): Whether to add noise.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `start_at_step` (Optional[int]): The step to start at.
        - `end_at_step` (Optional[int]): The step to end at.
        - `return_with_leftover_noise` (bool): Whether to return with leftover noise.
        - `sigma_ratio` (float, optional): The sigma ratio. Defaults to 1.0.
        - `sampler_opt` (Optional[Dict[str, Any]], optional): The sampler options. Defaults to None.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `callback` (Optional[callable], optional): The callback function. Defaults to None.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use  Defaults to False.

    #### Returns:
        - `Dict[str, torch.Tensor]`: The sampled tensor.
    """
    total_sigmas = calculate_sigmas2(model, sampler_name, scheduler, steps)

    sigmas = total_sigmas

    if start_at_step is not None:
        sigmas = sigmas[start_at_step:] * sigma_ratio

    impact_sampler = ksampler2(sampler_name, total_sigmas, pipeline=pipeline)

    res = sample_with_custom_noise(
        model,
        add_noise,
        seed,
        cfg,
        positive,
        negative,
        impact_sampler,
        sigmas,
        latent_image,
        noise=noise,
        callback=callback,
        pipeline=pipeline,
    )

    return res[1]


def ksampler_wrapper(
    model: torch.nn.Module,
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: Dict[str, torch.Tensor],
    denoise: float,
    refiner_ratio: Optional[float] = None,
    refiner_model: Optional[torch.nn.Module] = None,
    refiner_clip: Optional[Any] = None,
    refiner_positive: Optional[Any] = None,
    refiner_negative: Optional[Any] = None,
    sigma_factor: float = 1.0,
    noise: Optional[torch.Tensor] = None,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Dict[str, torch.Tensor]:
    """#### Wrapper for ksampler.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `denoise` (float): The denoise factor.
        - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
        - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
        - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
        - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
        - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
        - `sigma_factor` (float, optional): The sigma factor. Defaults to 1.0.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use  Defaults to False.

    #### Returns:
        - `Dict[str, torch.Tensor]`: The refined latent tensor.
    """
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    refined_latent = separated_sample(
        model,
        True,
        seed,
        advanced_steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        False,
        sigma_ratio=sigma_factor,
        noise=noise,
        scheduler_func=scheduler_func,
        pipeline=pipeline,
    )

    return refined_latent


def enhance_detail(
    image: torch.Tensor,
    model: torch.nn.Module,
    clip: Any,
    vae: VAE,
    guide_size: int,
    guide_size_for_bbox: bool,
    max_size: int,
    bbox: Tuple[int, int, int, int],
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    denoise: float,
    noise_mask: Optional[torch.Tensor],
    force_inpaint: bool,
    wildcard_opt: Optional[Any] = None,
    wildcard_opt_concat_mode: Optional[Any] = None,
    detailer_hook: Optional[callable] = None,
    refiner_ratio: Optional[float] = None,
    refiner_model: Optional[torch.nn.Module] = None,
    refiner_clip: Optional[Any] = None,
    refiner_positive: Optional[Any] = None,
    refiner_negative: Optional[Any] = None,
    control_net_wrapper: Optional[Any] = None,
    cycle: int = 1,
    inpaint_model: bool = False,
    noise_mask_feather: int = 0,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """#### Enhance detail of an image.

    #### Args:
        - `image` (torch.Tensor): The input image tensor.
        - `model` (torch.nn.Module): The model.
        - `clip` (Any): The clip model.
        - `vae` (VAE): The VAE model.
        - `guide_size` (int): The guide size.
        - `guide_size_for_bbox` (bool): Whether to use guide size for 
        - `max_size` (int): The maximum size.
        - `bbox` (Tuple[int, int, int, int]): The bounding box.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `denoise` (float): The denoise factor.
        - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
        - `force_inpaint` (bool): Whether to force inpaint.
        - `wildcard_opt` (Optional[Any], optional): The wildcard options. Defaults to None.
        - `wildcard_opt_concat_mode` (Optional[Any], optional): The wildcard concat mode. Defaults to None.
        - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
        - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
        - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
        - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
        - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
        - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
        - `control_net_wrapper` (Optional[Any], optional): The control net wrapper. Defaults to None.
        - `cycle` (int, optional): The number of cycles. Defaults to 1.
        - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
        - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use  Defaults to False.

    #### Returns:
        - `Tuple[torch.Tensor, Optional[Any]]`: The refined image tensor and optional cnet_pils.
    """
    if noise_mask is not None:
        noise_mask = tensor_gaussian_blur_mask(
            noise_mask, noise_mask_feather
        )
        noise_mask = noise_mask.squeeze(3)

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # for cropped_size
    upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        print("Detailer: force inpaint")
        upscale = 1.0
        new_w = w
        new_h = h

    print(
        f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}"
    )

    # upscale
    upscaled_image = tensor_resize(image, new_w, new_h)

    cnet_pils = None

    # prepare mask
    latent_image = to_latent_image(upscaled_image, vae)
    if noise_mask is not None:
        latent_image["noise_mask"] = noise_mask

    refined_latent = latent_image

    # ksampler
    for i in range(0, cycle):
        (
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            _upscaled_latent2,
            denoise2,
        ) = (
            model,
            seed + i,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )
        noise = None

        refined_latent = ksampler_wrapper(
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            refined_latent,
            denoise2,
            refiner_ratio,
            refiner_model,
            refiner_clip,
            refiner_positive,
            refiner_negative,
            noise=noise,
            scheduler_func=scheduler_func,
            pipeline=pipeline,
        )

    # non-latent downscale - latent downscale cause bad quality
    try:
        # try to decode image normally
        refined_image = vae.decode(refined_latent["samples"])
    except Exception:
        # usually an out-of-memory exception from the decode, so try a tiled approach
        refined_image = vae.decode_tiled(
            refined_latent["samples"],
            tile_x=64,
            tile_y=64,
        )

    # downscale
    refined_image = tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pils


class DetailerForEach:
    """#### Class for detailing each segment of an image."""

    @staticmethod
    def do_detail(
        image: torch.Tensor,
        segs: Tuple[torch.Tensor, Any],
        model: torch.nn.Module,
        clip: Any,
        vae: VAE,
        guide_size: int,
        guide_size_for_bbox: bool,
        max_size: int,
        seed: int,
        steps: int,
        cfg: int,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        denoise: float,
        feather: int,
        noise_mask: Optional[torch.Tensor],
        force_inpaint: bool,
        wildcard_opt: Optional[Any] = None,
        detailer_hook: Optional[callable] = None,
        refiner_ratio: Optional[float] = None,
        refiner_model: Optional[torch.nn.Module] = None,
        refiner_clip: Optional[Any] = None,
        refiner_positive: Optional[Any] = None,
        refiner_negative: Optional[Any] = None,
        cycle: int = 1,
        inpaint_model: bool = False,
        noise_mask_feather: int = 0,
        scheduler_func_opt: Optional[callable] = None,
        pipeline: bool = False,
    ) -> Tuple[torch.Tensor, list, list, list, list, Tuple[torch.Tensor, list]]:
        """#### Perform detailing on each segment of an image.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `segs` (Tuple[torch.Tensor, Any]): The segments.
            - `model` (torch.nn.Module): The model.
            - `clip` (Any): The clip model.
            - `vae` (VAE): The VAE model.
            - `guide_size` (int): The guide size.
            - `guide_size_for_bbox` (bool): Whether to use guide size for 
            - `max_size` (int): The maximum size.
            - `seed` (int): The seed for random noise.
            - `steps` (int): The number of steps.
            - `cfg` (int): Classifier-Free Guidance Scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler name.
            - `positive` (Any): The positive prompt.
            - `negative` (Any): The negative prompt.
            - `denoise` (float): The denoise factor.
            - `feather` (int): The feather value.
            - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
            - `force_inpaint` (bool): Whether to force inpaint.
            - `wildcard_opt` (Optional[Any], optional): The wildcard options. Defaults to None.
            - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
            - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
            - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
            - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
            - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
            - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
            - `cycle` (int, optional): The number of cycles. Defaults to 1.
            - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
            - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
            - `scheduler_func_opt` (Optional[callable], optional): The scheduler function. Defaults to None.
            - `pipeline` (bool, optional): Whether to use  Defaults to False.

        #### Returns:
            - `Tuple[torch.Tensor, list, list, list, list, Tuple[torch.Tensor, list]]`: The detailed image tensor, cropped list, enhanced list, enhanced alpha list, cnet PIL list, and new segments.
        """
        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        wmode, wildcard_chooser = process_wildcard_for_segs(wildcard_opt)

        ordered_segs = segs[1]

        if (
            noise_mask_feather > 0
            and "denoise_mask_function" not in model.model_options
        ):
            model = DifferentialDiffusion().apply(model)[0]

        for i, seg in enumerate(ordered_segs):
            cropped_image = crop_ndarray4(
                image.cpu().numpy(), seg.crop_region
            )  # Never use seg.cropped_image to handle overlapping area
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print("Detailer: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask

            seg_seed, wildcard_item = wildcard_chooser.get(seg)

            seg_seed = seed + i if seg_seed is None else seg_seed

            cropped_positive = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]

            cropped_negative = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]

            orig_cropped_image = cropped_image.clone()
            enhanced_image, cnet_pils = enhance_detail(
                cropped_image,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for_bbox,
                max_size,
                seg.bbox,
                seg_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                cropped_positive,
                cropped_negative,
                denoise,
                cropped_mask,
                force_inpaint,
                wildcard_opt=wildcard_item,
                wildcard_opt_concat_mode=wildcard_concat_mode,
                detailer_hook=detailer_hook,
                refiner_ratio=refiner_ratio,
                refiner_model=refiner_model,
                refiner_clip=refiner_clip,
                refiner_positive=refiner_positive,
                refiner_negative=refiner_negative,
                control_net_wrapper=seg.control_net_wrapper,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func=scheduler_func_opt,
                pipeline=pipeline,
            )

            if enhanced_image is not None:
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                tensor_paste(
                    image,
                    enhanced_image,
                    (seg.crop_region[0], seg.crop_region[1]),
                    mask,
                )  # this code affecting to `cropped_image`.
                enhanced_list.append(enhanced_image)

            # Convert enhanced_pil_alpha to RGBA mode
            enhanced_image_alpha = tensor_convert_rgba(enhanced_image)
            new_seg_image = (
                enhanced_image.numpy()
            )  # alpha should not be applied to seg_image
            # Apply the mask
            mask = tensor_resize(
                mask, *tensor_get_size(enhanced_image)
            )
            tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)

            cropped_list.append(orig_cropped_image)  # NOTE: Don't use `cropped_image`

            new_seg = SEG(
                new_seg_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        image_tensor = tensor_convert_rgb(image)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return (
            image_tensor,
            cropped_list,
            enhanced_list,
            enhanced_alpha_list,
            cnet_pil_list,
            (segs[0], new_segs),
        )


def empty_pil_tensor(w: int = 64, h: int = 64) -> torch.Tensor:
    """#### Create an empty PIL tensor.

    #### Args:
        - `w` (int, optional): The width of the tensor. Defaults to 64.
        - `h` (int, optional): The height of the tensor. Defaults to 64.

    #### Returns:
        - `torch.Tensor`: The empty tensor.
    """
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


class DetailerForEachTest(DetailerForEach):
    """#### Test class for DetailerForEach."""

    def doit(
        self,
        image: torch.Tensor,
        segs: Any,
        model: torch.nn.Module,
        clip: Any,
        vae: VAE,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        seed: int,
        steps: int,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        denoise: float,
        feather: int,
        noise_mask: Optional[torch.Tensor],
        force_inpaint: bool,
        wildcard: Optional[Any],
        detailer_hook: Optional[callable] = None,
        cycle: int = 1,
        inpaint_model: bool = False,
        noise_mask_feather: int = 0,
        scheduler_func_opt: Optional[callable] = None,
        pipeline: bool = False,
    ) -> Tuple[torch.Tensor, list, list, list, list]:
        """#### Perform detail enhancement for testing.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `segs` (Any): The segments.
            - `model` (torch.nn.Module): The model.
            - `clip` (Any): The clip model.
            - `vae` (VAE): The VAE model.
            - `guide_size` (int): The guide size.
            - `guide_size_for` (bool): Whether to use guide size for.
            - `max_size` (int): The maximum size.
            - `seed` (int): The seed for random noise.
            - `steps` (int): The number of steps.
            - `cfg` (Any): The configuration.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler name.
            - `positive` (Any): The positive prompt.
            - `negative` (Any): The negative prompt.
            - `denoise` (float): The denoise factor.
            - `feather` (int): The feather value.
            - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
            - `force_inpaint` (bool): Whether to force inpaint.
            - `wildcard` (Optional[Any]): The wildcard options.
            - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
            - `cycle` (int, optional): The number of cycles. Defaults to 1.
            - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
            - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
            - `scheduler_func_opt` (Optional[callable], optional): The scheduler function. Defaults to None.
            - `pipeline` (bool, optional): Whether to use  Defaults to False.

        #### Returns:
            - `Tuple[torch.Tensor, list, list, list, list]`: The enhanced image tensor, cropped list, cropped enhanced list, cropped enhanced alpha list, and cnet PIL list.
        """
        (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
            new_segs,
        ) = DetailerForEach.do_detail(
            image,
            segs,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            feather,
            noise_mask,
            force_inpaint,
            wildcard,
            detailer_hook,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func_opt=scheduler_func_opt,
            pipeline=pipeline,
        )

        cnet_pil_list = [empty_pil_tensor()]

        return (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
        )



import copy
import logging
import gguf
import torch



TORCH_COMPATIBLE_QTYPES = {
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


def is_torch_compatible(tensor):
    return (
        tensor is None
        or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES
    )


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


dequantize_functions = {
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
}


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    torch_compatible_tensor_types = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(
            f"{prefix}bias"
        )
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias
        return

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # apply patches
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = (
                    dtype if self.patch_dtype == "target" else self.patch_dtype
                )
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.forward_ggml_cast_weights(input, *args, **kwargs)
        return super().forward_comfy_cast_weights(input, *args, **kwargs)


class GGMLOps(manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """

    class Linear(GGMLLayer, manual_cast.Linear):
        def __init__(
            self, in_features, out_features, bias=True, device=None, dtype=None
        ):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Embedding(GGMLLayer, manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(
                self, device=input.device, dtype=out_dtype
            )
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)


def gguf_sd_loader_get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}"
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, *args, **kwargs):
        # Intel Arc fix, ref#50
        new = super().__deepcopy__(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


def gguf_sd_loader(path, handle_prefix="model.diffusion_model."):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = None
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        if (
            len(arch_field.types) != 1
            or arch_field.types[0] != gguf.GGUFValueType.STRING
        ):
            raise TypeError(
                f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}"
            )
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
        if arch_str not in {"flux", "sd1", "sdxl", "t5", "t5encoder"}:
            raise ValueError(
                f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}"
            )

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
        torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = gguf_sd_loader_get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any(
                    [
                        tensor_name.endswith(x)
                        for x in (".proj_in.weight", ".proj_out.weight")
                    ]
                ):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape
        )
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k, v in qtype_dict.items():
        print(f" {k:30}{v:3}")

    return state_dict


class GGUFModelPatcher(ModelPatcher):
    patch_on_device = False

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(
            device_to=device_to, unpatch_weights=unpatch_weights
        )

    mmap_released = False

    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        n = GGUFModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n


class UnetLoaderGGUF:

    def load_unet(
        self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None
    ):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = "./_internal/unet/" + unet_name
        sd = gguf_sd_loader(unet_path)
        model = load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path)
            )
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

clip_sd_map = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

clip_name_dict = {
    "stable_diffusion": CLIPType.STABLE_DIFFUSION,
    "sdxl": CLIPType.STABLE_DIFFUSION,
    "sd3": CLIPType.SD3,
    "flux": CLIPType.FLUX,
}


def gguf_clip_loader(path):
    raw_sd = gguf_sd_loader(path)
    assert "enc.blk.23.ffn_up.weight" in raw_sd, "Invalid Text Encoder!"
    sd = {}
    for k, v in raw_sd.items():
        for s, d in clip_sd_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


class CLIPLoaderGGUF:
    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                clip_data.append(gguf_clip_loader(p))
            else:
                sd = load_torch_file(p, safe_load=True)
                clip_data.append(
                    {
                        k: GGMLTensor(
                            v,
                            tensor_type=gguf.GGMLQuantizationType.F16,
                            tensor_shape=v.shape,
                        )
                        for k, v in sd.items()
                    }
                )
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = load_text_encoder_state_dicts(
            clip_type=clip_type,
            state_dicts=clip_data,
            model_options={
                "custom_operations": GGMLOps,
                "initial_device": text_encoder_offload_device(),
            },
            embedding_directory="models/embeddings",
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        # for some reason this is just missing in some SAI checkpoints
        if getattr(clip.cond_stage_model, "clip_l", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_l.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_l.transformer.text_projection = (
                    manual_cast.Linear(768, 768)
                )
        if getattr(clip.cond_stage_model, "clip_g", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_g.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_g.transformer.text_projection = (
                    manual_cast.Linear(1280, 1280)
                )

        return clip


class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = "./_internal/clip/" + clip_name1
        clip_path2 = "./_internal/clip/" + clip_name2
        clip_paths = (clip_path1, clip_path2)
        clip_type = clip_name_dict.get(type, CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class CLIPTextEncodeFlux:
    def encode(self, clip, clip_l, t5xxl, guidance, flux_enabled:bool = False):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True, flux_enabled=flux_enabled)
        cond = output.pop("cond")
        output["guidance"] = guidance
        return ([[cond, output]],)


class ConditioningZeroOut:

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c,)

import logging
import torch


def load_checkpoint_guess_config(
    ckpt_path: str,
    output_vae: bool = True,
    output_clip: bool = True,
    output_clipvision: bool = False,
    embedding_directory: str = None,
    output_model: bool = True,
) -> tuple:
    """#### Load a checkpoint and guess the configuration.

    #### Args:
        - `ckpt_path` (str): The path to the checkpoint file.
        - `output_vae` (bool, optional): Whether to output the VAE. Defaults to True.
        - `output_clip` (bool, optional): Whether to output the CLIP. Defaults to True.
        - `output_clipvision` (bool, optional): Whether to output the CLIP vision. Defaults to False.
        - `embedding_directory` (str, optional): The embedding directory. Defaults to None.
        - `output_model` (bool, optional): Whether to output the model. Defaults to True.

    #### Returns:
        - `tuple`: The model patcher, CLIP, VAE, and CLIP vision.
    """
    sd = load_torch_file(ckpt_path)
    sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    load_device = get_torch_device()

    model_config = model_config_from_unet(sd, "model.diffusion_model.")
    unet_dtype = unet_dtype1(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
    )
    manual_cast_dtype = unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if output_model:
        inital_load_device = unet_inital_load_device(parameters, unet_dtype)
        unet_offload_device()
        model = model_config.get_model(
            sd, "model.diffusion_model.", device=inital_load_device
        )
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
        )
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(
                        filter(
                            lambda a: ".logit_scale" not in a
                            and ".transformer.text_projection.weight" not in a,
                            m,
                        )
                    )
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = ModelPatcher(
            model,
            load_device=load_device,
            offload_device=unet_offload_device(),
            current_device=inital_load_device,
        )
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)


class CheckpointLoaderSimple:
    """#### Class for loading checkpoints."""

    def load_checkpoint(
        self, ckpt_name: str, output_vae: bool = True, output_clip: bool = True
    ) -> tuple:
        """#### Load a checkpoint.

        #### Args:
            - `ckpt_name` (str): The name of the checkpoint.
            - `output_vae` (bool, optional): Whether to output the VAE. Defaults to True.
            - `output_clip` (bool, optional): Whether to output the CLIP. Defaults to True.

        #### Returns:
            - `tuple`: The model patcher, CLIP, and VAE.
        """
        ckpt_path = f"{ckpt_name}"
        out = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=output_vae,
            output_clip=output_clip,
            embedding_directory="./_internal/embeddings/",
        )
        print("loading", ckpt_path)
        return out[:3]


import torch


class sm_SD15(BASE):
    """#### Class representing the SD15 model.

    #### Args:
        - `BASE` (BASE): The base model class.
    """

    unet_config: dict = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config: dict = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format: SD15 = SD15

    def process_clip_state_dict(self, state_dict: dict) -> dict:
        """#### Process the state dictionary for the CLIP model.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                "cond_stage_model.transformer.text_model."
            ):
                y = x.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                state_dict[y] = state_dict.pop(x)

        if (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
            in state_dict
        ):
            ids = state_dict[
                "cond_stage_model.transformer.text_model.embeddings.position_ids"
            ]
            if ids.dtype == torch.float32:
                state_dict[
                    "cond_stage_model.transformer.text_model.embeddings.position_ids"
                ] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )
        return state_dict

    def clip_target(self) -> ClipTarget:
        """#### Get the target CLIP model.

        #### Returns:
            - `ClipTarget`: The target CLIP model.
        """
        return ClipTarget(SD1Tokenizer, SD1ClipModel)
    
models = [
    sm_SD15, Flux
]

import torch
from PIL import ImageFilter, ImageDraw, Image
from enum import Enum
import math

# taken from https://github.com/ssitu/ComfyUI_UltimateSDUpscale

state = state

class UnsupportedModel(Exception):
    """#### Exception raised for unsupported models."""
    pass


class StableDiffusionProcessing:
    """#### Class representing the processing of Stable Diffusion images."""

    def __init__(
        self,
        init_img: Image.Image,
        model: torch.nn.Module,
        positive: str,
        negative: str,
        vae: VAE,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        upscale_by: float,
        uniform_tile_mode: bool,
    ):
        """
        #### Initialize the StableDiffusionProcessing class.

        #### Args:
            - `init_img` (Image.Image): The initial image.
            - `model` (torch.nn.Module): The model.
            - `positive` (str): The positive prompt.
            - `negative` (str): The negative prompt.
            - `vae` (VAE): The variational autoencoder.
            - `seed` (int): The seed.
            - `steps` (int): The number of steps.
            - `cfg` (float): The CFG scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler.
            - `denoise` (float): The denoise strength.
            - `upscale_by` (float): The upscale factor.
            - `uniform_tile_mode` (bool): Whether to use uniform tile mode.
        """
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:
    """#### Class representing the processed images."""

    def __init__(
        self, p: StableDiffusionProcessing, images: list, seed: int, info: str
    ):
        """
        #### Initialize the Processed class.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `images` (list): The list of images.
            - `seed` (int): The seed.
            - `info` (str): The information string.
        """
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index: int) -> str:
        """
        #### Get the information text.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `index` (int): The index.

        #### Returns:
            - `str`: The information text.
        """
        return None


def fix_seed(p: StableDiffusionProcessing) -> None:
    """
    #### Fix the seed for reproducibility.

    #### Args:
        - `p` (StableDiffusionProcessing): The processing object.
    """
    pass


def process_images(p: StableDiffusionProcessing, pipeline: bool = False) -> Processed:
    """
    #### Process the images.

    #### Args:
        - `p` (StableDiffusionProcessing): The processing object.

    #### Returns:
        - `Processed`: The processed images.
    """
    # Where the main image generation happens in A1111

    # Setup
    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    x1, y1, x2, y2 = crop_region
    crop_width = x2 - x1
    crop_height = y2 - y1
    crop_ratio = crop_width / crop_height
    p_ratio = p.width / p.height
    if crop_ratio > p_ratio:
        target_width = crop_width
        target_height = round(crop_width / p_ratio)
    else:
        target_width = round(crop_height * p_ratio)
        target_height = crop_height
    crop_region, _ = expand_crop(
        crop_region,
        image_mask.width,
        image_mask.height,
        target_width,
        target_height,
    )
    tile_size = p.width, p.height

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(
        p.positive, crop_region, p.init_size, init_image.size, tile_size
    )
    negative_cropped = crop_cond(
        p.negative, crop_region, p.init_size, init_image.size, tile_size
    )

    # Encode the image
    vae_encoder = VAEEncode()
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    (samples,) = common_ksampler(
        p.model,
        p.seed,
        p.steps,
        p.cfg,
        p.sampler_name,
        p.scheduler,
        positive_cropped,
        negative_cropped,
        latent,
        denoise=p.denoise,
        pipeline=pipeline
    )

    # Decode the sample
    vae_decoder = VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(
                initial_tile_size, Image.Resampling.LANCZOS
            )

        # Put the tile into position
        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        image_mask = image_mask.resize(temp.size)
        temp.putalpha(image_mask)
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert("RGB")
        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed


class USDUMode(Enum):
    """#### Enum representing the modes for Ultimate SD Upscale."""
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    """#### Enum representing the seam fix modes for Ultimate SD Upscale."""
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler:
    """#### Class representing the Ultimate SD Upscaler."""

    def __init__(
        self,
        p: StableDiffusionProcessing,
        image: Image.Image,
        upscaler_index: int,
        save_redraw: bool,
        save_seams_fix: bool,
        tile_width: int,
        tile_height: int,
    ) -> None:
        """
        #### Initialize the USDUpscaler class.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `upscaler_index` (int): The upscaler index.
            - `save_redraw` (bool): Whether to save the redraw.
            - `save_seams_fix` (bool): Whether to save the seams fix.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
        """
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(
            max(p.width, p.height) / max(image.width, image.height)
        )
        self.upscaler = sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num: int) -> int:
        """
        #### Get the factor for a given number.

        #### Args:
            - `num` (int): The number.

        #### Returns:
            - `int`: The factor.
        """
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self) -> None:
        """
        #### Get the list of scale factors.
        """
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
        self.scales = enumerate(scales)

    def upscale(self) -> None:
        """
        #### Upscale the image.
        """
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(
                self.image, value, self.upscaler.data_path
            )
        # Resize image to set values
        self.image = self.image.resize(
            (self.p.width, self.p.height), resample=Image.LANCZOS
        )

    def setup_redraw(self, redraw_mode: int, padding: int, mask_blur: int) -> None:
        """
        #### Set up the redraw.

        #### Args:
            - `redraw_mode` (int): The redraw mode.
            - `padding` (int): The padding.
            - `mask_blur` (int): The mask blur.
        """
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(
        self, padding: int, denoise: float, mask_blur: int, width: int, mode: int
    ) -> None:
        """
        #### Set up the seams fix.

        #### Args:
            - `padding` (int): The padding.
            - `denoise` (float): The denoise strength.
            - `mask_blur` (int): The mask blur.
            - `width` (int): The width.
            - `mode` (int): The mode.
        """
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def calc_jobs_count(self) -> None:
        """
        #### Calculate the number of jobs.
        """
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self) -> None:
        """
        #### Print the information.
        """
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self) -> None:
        """
        #### Add extra information.
        """
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = (
            self.upscaler.name
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = (
            self.redraw.tile_width
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = (
            self.redraw.tile_height
        )
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = (
            self.p.mask_blur
        )
        self.p.extra_generation_params["Ultimate SD upscale padding"] = (
            self.redraw.padding
        )

    def process(self, pipeline) -> None:
        """
        #### Process the image.
        """
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols, pipeline)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols, pipeline)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
        state.end()


class USDURedraw:
    """#### Class representing the redraw functionality for Ultimate SD Upscale."""

    def init_draw(self, p: StableDiffusionProcessing, width: int, height: int) -> tuple:
        """
        #### Initialize the draw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `width` (int): The width.
            - `height` (int): The height.

        #### Returns:
            - `tuple`: The mask and draw objects.
        """
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi: int, yi: int) -> tuple:
        """
        #### Calculate the rectangle coordinates.

        #### Args:
            - `xi` (int): The x index.
            - `yi` (int): The y index.

        #### Returns:
            - `tuple`: The rectangle coordinates.
        """
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """
        #### Perform linear processing.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p, pipeline)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False) -> Image.Image:
        """#### Start the redraw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.
            
        #### Returns:
            - `Image.Image`: The processed image.
        """
        self.initial_info = None
        return self.linear_process(p, image, rows, cols, pipeline=pipeline)


class USDUSeamsFix:
    """#### Class representing the seams fix functionality for Ultimate SD Upscale."""

    def init_draw(self, p: StableDiffusionProcessing) -> None:
        """#### Initialize the draw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
        """
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """#### Perform half-tile processing.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
        global state
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(
            gradient.resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        row_gradient.paste(
            gradient.rotate(180).resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, self.tile_height // 2),
        )
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(
            gradient.rotate(90).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        col_gradient.paste(
            gradient.rotate(270).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (self.tile_width // 2, 0),
        )

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    row_gradient,
                    (
                        xi * self.tile_width,
                        yi * self.tile_height + self.tile_height // 2,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p, pipeline)
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    col_gradient,
                    (
                        xi * self.tile_width + self.tile_width // 2,
                        yi * self.tile_height,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p, pipeline)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """#### Start the seams fix process.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
        return self.half_tile_process(p, image, rows, cols, pipeline=pipeline)


class Script(Script):
    """#### Class representing the script for Ultimate SD Upscale."""

    def run(
        self,
        p: StableDiffusionProcessing,
        _: None,
        tile_width: int,
        tile_height: int,
        mask_blur: int,
        padding: int,
        seams_fix_width: int,
        seams_fix_denoise: float,
        seams_fix_padding: int,
        upscaler_index: int,
        save_upscaled_image: bool,
        redraw_mode: int,
        save_seams_fix_image: bool,
        seams_fix_mask_blur: int,
        seams_fix_type: int,
        target_size_type: int,
        custom_width: int,
        custom_height: int,
        custom_scale: float,
        pipeline: bool = False,
    ) -> Processed:
        """#### Run the script.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `_` (None): Unused parameter.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
            - `mask_blur` (int): The mask blur.
            - `padding` (int): The padding.
            - `seams_fix_width` (int): The seams fix width.
            - `seams_fix_denoise` (float): The seams fix denoise strength.
            - `seams_fix_padding` (int): The seams fix padding.
            - `upscaler_index` (int): The upscaler index.
            - `save_upscaled_image` (bool): Whether to save the upscaled image.
            - `redraw_mode` (int): The redraw mode.
            - `save_seams_fix_image` (bool): Whether to save the seams fix image.
            - `seams_fix_mask_blur` (int): The seams fix mask blur.
            - `seams_fix_type` (int): The seams fix type.
            - `target_size_type` (int): The target size type.
            - `custom_width` (int): The custom width.
            - `custom_height` (int): The custom height.
            - `custom_scale` (float): The custom scale.

        #### Returns:
            - `Processed`: The processed images.
        """
        # Init
        fix_seed(p)
        torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        init_img = flatten(
            init_img, opts.img2img_background_color
        )

        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
        )
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(
            seams_fix_padding,
            seams_fix_denoise,
            seams_fix_mask_blur,
            seams_fix_width,
            seams_fix_type,
        )
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process(pipeline=pipeline)
        result_images = upscaler.result_images

        return Processed(
            p,
            result_images,
            seed,
            upscaler.initial_info if upscaler.initial_info is not None else "",
        )


# Upscaler
old_init = USDUpscaler.__init__


def new_init(
    self: USDUpscaler,
    p: StableDiffusionProcessing,
    image: Image.Image,
    upscaler_index: int,
    save_redraw: bool,
    save_seams_fix: bool,
    tile_width: int,
    tile_height: int,
) -> None:
    """#### Initialize the USDUpscaler class with new settings.

    #### Args:
        - `self` (USDUpscaler): The USDUpscaler instance.
        - `p` (StableDiffusionProcessing): The processing object.
        - `image` (Image.Image): The image.
        - `upscaler_index` (int): The upscaler index.
        - `save_redraw` (bool): Whether to save the redraw.
        - `save_seams_fix` (bool): Whether to save the seams fix.
        - `tile_width` (int): The tile width.
        - `tile_height` (int): The tile height.
    """
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(
        self,
        p,
        image,
        upscaler_index,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    )


USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = USDURedraw.init_draw


def new_setup_redraw(
    self: USDURedraw, p: StableDiffusionProcessing, width: int, height: int
) -> tuple:
    """#### Set up the redraw with new settings.

    #### Args:
        - `self` (USDURedraw): The USDURedraw instance.
        - `p` (StableDiffusionProcessing): The processing object.
        - `width` (int): The width.
        - `height` (int): The height.

    #### Returns:
        - `tuple`: The mask and draw objects.
    """
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self: USDUSeamsFix, p: StableDiffusionProcessing) -> None:
    """#### Set up the seams fix with new settings.

    #### Args:
        - `self` (USDUSeamsFix): The USDUSeamsFix instance.
        - `p` (StableDiffusionProcessing): The processing object.
    """
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

# Make the script upscale on a batch of images instead of one image
old_upscale = USDUpscaler.upscale


def new_upscale(self: USDUpscaler) -> None:
    """#### Upscale a batch of images.

    #### Args:
        - `self` (USDUpscaler): The USDUpscaler instance.
    """
    old_upscale(self)
    batch = [self.image] + [
        img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        for img in batch[1:]
    ]


USDUpscaler.upscale = new_upscale
MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


class UltimateSDUpscale:
    """#### Class representing the Ultimate SD Upscale functionality."""

    def upscale(
        self,
        image: torch.Tensor,
        model: torch.nn.Module,
        positive: str,
        negative: str,
        vae: VAE,
        upscale_by: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        upscale_model: any,
        mode_type: str,
        tile_width: int,
        tile_height: int,
        mask_blur: int,
        tile_padding: int,
        seam_fix_mode: str,
        seam_fix_denoise: float,
        seam_fix_mask_blur: int,
        seam_fix_width: int,
        seam_fix_padding: int,
        force_uniform_tiles: bool,
        pipeline: bool = False,
    ) -> tuple:
        """#### Upscale the image.

        #### Args:
            - `image` (torch.Tensor): The image tensor.
            - `model` (torch.nn.Module): The model.
            - `positive` (str): The positive prompt.
            - `negative` (str): The negative prompt.
            - `vae` (VAE): The variational autoencoder.
            - `upscale_by` (float): The upscale factor.
            - `seed` (int): The seed.
            - `steps` (int): The number of steps.
            - `cfg` (float): The CFG scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler.
            - `denoise` (float): The denoise strength.
            - `upscale_model` (any): The upscale model.
            - `mode_type` (str): The mode type.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
            - `mask_blur` (int): The mask blur.
            - `tile_padding` (int): The tile padding.
            - `seam_fix_mode` (str): The seam fix mode.
            - `seam_fix_denoise` (float): The seam fix denoise strength.
            - `seam_fix_mask_blur` (int): The seam fix mask blur.
            - `seam_fix_width` (int): The seam fix width.
            - `seam_fix_padding` (int): The seam fix padding.
            - `force_uniform_tiles` (bool): Whether to force uniform tiles.

        #### Returns:
            - `tuple`: The resulting tensor.
        """
        # Set up A1111 patches

        # Upscaler
        # An object that the script works with
        sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        actual_upscaler = upscale_model

        # Set the batch of images
        batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image),
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_by,
            force_uniform_tiles,
        )

        # Running the script
        script = Script()
        script.run(
            p=sdprocessing,
            _=None,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            padding=tile_padding,
            seams_fix_width=seam_fix_width,
            seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding,
            upscaler_index=0,
            save_upscaled_image=False,
            redraw_mode=MODES[mode_type],
            save_seams_fix_image=False,
            seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
            target_size_type=2,
            custom_width=None,
            custom_height=None,
            custom_scale=upscale_by,
            pipeline=pipeline,
        )

        # Return the resulting images
        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

import torch

try:
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
except ImportError:
    pass

# Taken from https://github.com/gameltb/ComfyUI_stable_fast


def gen_stable_fast_config() -> CompilationConfig:
    """#### Generate the StableFast configuration.

    #### Returns:
        - `CompilationConfig`: The StableFast configuration.
    """
    config = CompilationConfig.Default()
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = False
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    """#### Class representing a StableFast patch."""

    def __init__(self, model: torch.nn.Module, config: CompilationConfig):
        """#### Initialize the StableFastPatch.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `config` (CompilationConfig): The configuration.
        """
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function: callable, params: dict) -> torch.Tensor:
        """#### Call the StableFastPatch.

        #### Args:
            - `model_function` (callable): The model function.
            - `params` (dict): The parameters.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device: torch.device) -> object:
        """#### Move the model to a specific device.

        #### Args:
            - `device` (torch.device): The device.

        #### Returns:
            - `object`: The StableFastPatch instance.
        """
        if isinstance(device, torch.device):
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disabling enable_cuda_graph.\33[0m"
                    )
        return self


class ApplyStableFastUnet:
    """#### Class for applying StableFast to a UNet model."""

    def apply_stable_fast(self, model: torch.nn.Module, enable_cuda_graph: bool) -> tuple:
        """#### Apply StableFast to the model.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `enable_cuda_graph` (bool): Whether to enable CUDA graph.

        #### Returns:
            - `tuple`: The StableFast model.
        """
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)



import itertools
import math
from time import time
from typing import Any, NamedTuple

import torch

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
    SCALE_METHODS = ("disabled", "skip", *UPSCALE_METHODS)
    REVERSE_SCALE_METHODS = UPSCALE_METHODS
    scale_samples = scale_samples


MODULES.register_init_handler(init_integrations)

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
    ModelType1.SD15: Preset(input_blocks="1,2", output_blocks="11,10,9"),
    ModelType1.SDXL: Preset(input_blocks="4,5", output_blocks="3,4,5"),
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
        model: ModelPatcher,
        yaml_parameters: str | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[ModelPatcher]:
        """#### Patch the model with MSW-MSA attention.

        #### Args:
            - `model` (ModelPatcher): The model patcher.
            - `yaml_parameters` (str | None, optional): The YAML parameters. Defaults to None.
            - `kwargs` (dict[str, Any]): Additional keyword arguments.

        #### Returns:
            - `tuple[ModelPatcher]`: The patched model.
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
    """Class representing a simplified version of MSW-MSA """
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
        model_type: str | ModelType1,
        model: ModelPatcher,
    ) -> tuple[ModelPatcher]:
        """#### Apply the MSW-MSA attention patch.

        #### Args:
            - `model_type` (str | ModelType1): The model type.
            - `model` (ModelPatcher): The model patcher.

        #### Returns:
            - `tuple[ModelPatcher]`: The patched model.
        """
        if model_type == "auto":
            guessed_model_type = guess_model_type(model)
            if guessed_model_type not in SIMPLE_PRESETS:
                raise RuntimeError("Unable to guess model type")
            model_type = guessed_model_type
        else:
            model_type = ModelType1(model_type)
        preset = SIMPLE_PRESETS.get(model_type)
        if preset is None:
            errstr = f"Unknown model type {model_type!s}"
            raise ValueError(errstr)
        logger.info(
            f"** ApplyMSWMSAAttentionSimple: Using preset {model_type!s}: in/mid/out blocks [{preset.pretty_blocks}], start/end percent {preset.start_time:.2}/{preset.end_time:.2}",
        )
        return ApplyMSWMSAAttention.patch(model=model, **preset.as_dict)


__all__ = ("ApplyMSWMSAAttention", "ApplyMSWMSAAttentionSimple")

import os
import numpy as np
from PIL import Image

output_directory = "./_internal/output"


def get_output_directory() -> str:
    """#### Get the output directory.

    #### Returns:
        - `str`: The output directory.
    """
    global output_directory
    return output_directory


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0
) -> tuple:
    """#### Get the save image path.

    #### Args:
        - `filename_prefix` (str): The filename prefix.
        - `output_dir` (str): The output directory.
        - `image_width` (int, optional): The image width. Defaults to 0.
        - `image_height` (int, optional): The image height. Defaults to 0.

    #### Returns:
        - `tuple`: The full output folder, filename, counter, subfolder, and filename prefix.
    """

    def map_filename(filename: str) -> tuple:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)
    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


MAX_RESOLUTION = 16384


class SaveImage:
    """#### Class for saving images."""

    def __init__(self):
        """#### Initialize the SaveImage class."""
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_images(
        self,
        images: list,
        filename_prefix: str = "LD",
        prompt: str = None,
        extra_pnginfo: dict = None,
    ) -> dict:
        """#### Save images to the output directory.

        #### Args:
            - `images` (list): The list of images.
            - `filename_prefix` (str, optional): The filename prefix. Defaults to "LD".
            - `prompt` (str, optional): The prompt. Defaults to None.
            - `extra_pnginfo` (dict, optional): Additional PNG info. Defaults to None.

        #### Returns:
            - `dict`: The saved images information.
        """
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


import ollama



def enhance_prompt(p: str = None) -> str:
    """#### Enhance a text-to-image prompt using Ollama.

    #### Args:
        - `p` (str, optional): The prompt. Defaults to `None`.

    #### Returns:
        - `str`: The enhanced prompt
    """
    prompt = load_parameters_from_file()[0]
    if p is None:
        pass
    else:
        prompt = p
    print(prompt)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"""Your goal is to generate a text-to-image prompt based on a user's input, detailing their desired final outcome for an image. The user will provide specific details about the characteristics, features, or elements they want the image to include. The prompt should guide the generation of an image that aligns with the user's desired outcome.

                        Generate a text-to-image prompt by arranging the following blocks in a single string, separated by commas:

                        Image Type: [Specify desired image type]

                        Aesthetic or Mood: [Describe desired aesthetic or mood]

                        Lighting Conditions: [Specify desired lighting conditions]

                        Composition or Framing: [Provide details about desired composition or framing]

                        Background: [Specify desired background elements or setting]

                        Colors: [Mention any specific colors or color palette]

                        Objects or Elements: [List specific objects or features]

                        Style or Artistic Influence: [Mention desired artistic style or influence]

                        Subject's Appearance: [Describe appearance of main subject]

                        Ensure the blocks are arranged in order of visual importance, from the most significant to the least significant, to effectively guide image generation, a block can be surrounded by parentheses to gain additionnal significance.

                        This is an example of a user's input: "a beautiful blonde lady in lingerie sitting in seiza in a seducing way with a focus on her assets"

                        And this is an example of a desired output: "portrait| serene and mysterious| soft, diffused lighting| close-up shot, emphasizing facial features| simple and blurred background| earthy tones with a hint of warm highlights| renaissance painting| a beautiful lady with freckles and dark makeup"
                        
                        Here is the user's input: {prompt}

                        Write the prompt in the same style as the example above, in a single line , with absolutely no additional information, words or symbols other than the enhanced prompt.

                        Output:""",
            },
        ],
    )
    print("here's the enhanced prompt :", response["message"]["content"])
    return response["message"]["content"]


from typing import List
import torch


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """#### Perform bilinear interpolation on samples.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `width` (int): The target width.
        - `height` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The interpolated samples.
    """

    def slerp(b1: torch.Tensor, b2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """#### Perform spherical linear interpolation between two vectors.

        #### Args:
            - `b1` (torch.Tensor): The first vector.
            - `b2` (torch.Tensor): The second vector.
            - `r` (torch.Tensor): The interpolation ratio.

        #### Returns:
            - `torch.Tensor`: The interpolated vector.
        """

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
            1
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
            1
        ) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(
        length_old: int, length_new: int, device: torch.device
    ) -> List[torch.Tensor]:
        """#### Generate bilinear data for interpolation.

        #### Args:
            - `length_old` (int): The old length.
            - `length_new` (int): The new length.
            - `device` (torch.device): The device to use.

        #### Returns:
            - `torch.Tensor`: The ratios.
            - `torch.Tensor`: The first coordinates.
            - `torch.Tensor`: The second coordinates.
        """
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1)
        )
        coords_1 = torch.nn.functional.interpolate(
            coords_1, size=(1, length_new), mode="bilinear"
        )
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape(
                (1, 1, 1, -1)
            )
            + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2, size=(1, length_new), mode="bilinear"
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def common_upscale(samples: List, width: int, height: int) -> torch.Tensor:
    """#### Upscales the given samples to the specified width and height using the specified method and crop settings.
    #### Args:
        - `samples` (list): The list of samples to be upscaled.
        - `width` (int): The target width for the upscaled samples.
        - `height` (int): The target height for the upscaled samples.
    #### Returns:
        - `torch.Tensor`: The upscaled samples.
    """
    s = samples
    return bislerp(s, width, height)


class LatentUpscale:
    """#### A class to upscale latent codes."""

    def upscale(self, samples: dict, width: int, height: int) -> tuple:
        """#### Upscales the given latent codes.

        #### Args:
            - `samples` (dict): The latent codes to be upscaled.
            - `width` (int): The target width for the upscaled samples.
            - `height` (int): The target height for the upscaled samples.

        #### Returns:
            - `tuple`: The upscaled samples.
        """
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()
            width = max(64, width)
            height = max(64, height)

            s["samples"] = common_upscale(samples["samples"], width // 8, height // 8)
        return (s,)


import os
import random
import sys
import argparse

import numpy as np
import torch
import torch._dynamo

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))




torch._dynamo.config.suppress_errors = True
torch.compiler.allow_in_graph

last_seed = 0

CheckAndDownload()

def pipeline(
    prompt: str,
    w: int,
    h: int,
    number: int = 1,
    batch: int = 1,
    hires_fix: bool = False,
    adetailer: bool = False,
    enhance_prompt: bool = False,
    img2img: bool = False,
    stable_fast: bool = False,
    reuse_seed: bool = False,
    flux_enabled: bool = False,
) -> None:
    """#### Run the LightDiffusion 

    #### Args:
        - `prompt` (str): The prompt for the 
        - `w` (int): The width of the generated image.
        - `h` (int): The height of the generated image.
        - `hires_fix` (bool, optional): Enable high-resolution fix. Defaults to False.
        - `adetailer` (bool, optional): Enable automatic face and body enhancing. Defaults to False.
        - `enhance_prompt` (bool, optional): Enable llama3.2 prompt enhancement. Defaults to False.
        - `img2img` (bool, optional): Use LightDiffusion in Image to Image mode, the prompt input becomes the path to the input image. Defaults to False.
        - `stable_fast` (bool, optional): Enable Stable-Fast speedup offering a 70% speed improvement in return of a compilation time. Defaults to False.
        - `reuse_seed` (bool, optional): Reuse the last used seed, if False the seed will be kept random. Default to False.
        - `flux_enabled` (bool, optional): Enable the flux mode. Defaults to False.
    """
    global last_seed
    if reuse_seed:
        seed = last_seed
    else:
        seed = random.randint(1, 2**64)
        last_seed = seed
    ckpt = "./_internal/checkpoints/Meina V10 - baked VAE.safetensors"
    with torch.inference_mode():
        if not flux_enabled:
            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
                ckpt_name=ckpt
            )
            hidiffoptimizer = ApplyMSWMSAAttentionSimple()
        cliptextencode = CLIPTextEncode()
        emptylatentimage = EmptyLatentImage()
        ksampler_instance = KSampler2()
        vaedecode = VAEDecode()
        saveimage = SaveImage()
        latent_upscale = LatentUpscale()
    for _ in range(number):
        if img2img:
            img = Image.open(prompt)
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            with torch.inference_mode():
                ultimatesdupscale = UltimateSDUpscale()
                try:
                    loraloader = LoraLoader()
                    loraloader_274 = loraloader.load_lora(
                        lora_name="add_detail.safetensors",
                        strength_model=2,
                        strength_clip=2,
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                    )
                except:
                    loraloader_274 = checkpointloadersimple_241

                if stable_fast is True:

                    applystablefast = ApplyStableFastUnet()
                    applystablefast_158 = applystablefast.apply_stable_fast(
                        enable_cuda_graph=False,
                        model=loraloader_274[0],
                    )
                else:
                    applystablefast_158 = loraloader_274

                clipsetlastlayer = CLIPSetLastLayer()
                clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=loraloader_274[1]
                )

                cliptextencode_242 = cliptextencode.encode(
                    text=prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)",
                    clip=clipsetlastlayer_257[0],
                )
                upscalemodelloader = UpscaleModelLoader()
                upscalemodelloader_244 = upscalemodelloader.load_model(
                    "RealESRGAN_x4plus.pth"
                )
                ultimatesdupscale_250 = ultimatesdupscale.upscale(
                    upscale_by=2,
                    seed=random.randint(1, 2**64),
                    steps=8,
                    cfg=6,
                    sampler_name="dpmpp_2m_sde",
                    scheduler="karras",
                    denoise=0.3,
                    mode_type="Linear",
                    tile_width=512,
                    tile_height=512,
                    mask_blur=16,
                    tile_padding=32,
                    seam_fix_mode="Half Tile",
                    seam_fix_denoise=0.2,
                    seam_fix_width=64,
                    seam_fix_mask_blur=16,
                    seam_fix_padding=32,
                    force_uniform_tiles="enable",
                    image=img_tensor,
                    model=applystablefast_158[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    vae=checkpointloadersimple_241[2],
                    upscale_model=upscalemodelloader_244[0],
                    pipeline=True,
                )
                saveimage.save_images(
                    filename_prefix="LD-i2i",
                    images=ultimatesdupscale_250[0],
                )
        elif flux_enabled:
            CheckAndDownloadFlux()
            with torch.inference_mode():
                dualcliploadergguf = DualCLIPLoaderGGUF()
                emptylatentimage = EmptyLatentImage()
                vaeloader = VAELoader()
                unetloadergguf = UnetLoaderGGUF()
                cliptextencodeflux = CLIPTextEncodeFlux()
                conditioningzeroout = ConditioningZeroOut()
                ksampler = KSampler2()
                unetloadergguf_10 = unetloadergguf.load_unet(
                    unet_name="flux1-dev-Q8_0.gguf"
                )
                vaeloader_11 = vaeloader.load_vae(vae_name="ae.safetensors")
                dualcliploadergguf_19 = dualcliploadergguf.load_clip(
                    clip_name1="clip_l.safetensors",
                    clip_name2="t5-v1_1-xxl-encoder-Q8_0.gguf",
                    type="flux",
                )
                emptylatentimage_5 = emptylatentimage.generate(
                    width=w, height=h, batch_size=batch
                )
                cliptextencodeflux_15 = cliptextencodeflux.encode(
                    clip_l=prompt,
                    t5xxl=prompt,
                    guidance=3.5,
                    clip=dualcliploadergguf_19[0],
                    flux_enabled=True,
                )
                conditioningzeroout_16 = conditioningzeroout.zero_out(
                    conditioning=cliptextencodeflux_15[0]
                )
                ksampler_3 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=20,
                    cfg=1,
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=1,
                    model=unetloadergguf_10[0],
                    positive=cliptextencodeflux_15[0],
                    negative=conditioningzeroout_16[0],
                    latent_image=emptylatentimage_5[0],
                    pipeline=True,
                    flux=True,
                )

                vaedecode_8 = vaedecode.decode(
                    samples=ksampler_3[0],
                    vae=vaeloader_11[0],
                    flux=True,
                )

                saveimage.save_images(
                    filename_prefix="Flux", images=vaedecode_8[0]
                )
        else:
            if enhance_prompt:
                try:
                    prompt = enhance_prompt(prompt)
                except:
                    pass
            while prompt is None:
                pass
            if should_use_bf16():
                dtype = torch.bfloat16
            elif should_use_fp16():
                dtype = torch.float16
            else:
                dtype = torch.float32
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=dtype),
            ):
                try:
                    loraloader = LoraLoader()
                    loraloader_274 = loraloader.load_lora(
                        lora_name="add_detail.safetensors",
                        strength_model=0.7,
                        strength_clip=0.7,
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                    )
                    print("loading add_detail.safetensors")
                except:
                    loraloader_274 = checkpointloadersimple_241
                clipsetlastlayer = CLIPSetLastLayer()
                clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=loraloader_274[1]
                )
                applystablefast_158 = loraloader_274
                cliptextencode_242 = cliptextencode.encode(
                    text=prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)",
                    clip=clipsetlastlayer_257[0],
                )
                emptylatentimage_244 = emptylatentimage.generate(
                    width=w, height=h, batch_size=batch
                )
                ksampler_239 = ksampler_instance.sample(
                    seed=seed,
                    steps=40,
                    cfg=7,
                    sampler_name="dpm_adaptive",
                    scheduler="karras",
                    denoise=1,
                    pipeline=True,
                    model=hidiffoptimizer.go(
                        model_type="auto", model=applystablefast_158[0]
                    )[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=emptylatentimage_244[0],
                )
                if hires_fix:
                    latentupscale_254 = latent_upscale.upscale(
                        width=w * 2,
                        height=h * 2,
                        samples=ksampler_239[0],
                    )
                    ksampler_253 = ksampler_instance.sample(
                        seed=random.randint(1, 2**64),
                        steps=10,
                        cfg=8,
                        sampler_name="euler_ancestral",
                        scheduler="normal",
                        denoise=0.45,
                        model=hidiffoptimizer.go(
                            model_type="auto", model=applystablefast_158[0]
                        )[0],
                        positive=cliptextencode_242[0],
                        negative=cliptextencode_243[0],
                        latent_image=latentupscale_254[0],
                        pipeline=True,
                    )
                else:
                    ksampler_253 = ksampler_239

                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )

            if adetailer:
                with torch.inference_mode():
                    samloader = SAMLoader()
                    samloader_87 = samloader.load_model(
                        model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                    )
                    cliptextencode_124 = cliptextencode.encode(
                        text="royal, detailed, magnificient, beautiful, seducing",
                        clip=loraloader_274[1],
                    )
                    ultralyticsdetectorprovider = UltralyticsDetectorProvider()
                    ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                        # model_name="face_yolov8m.pt"
                        model_name="person_yolov8m-seg.pt"
                    )
                    bboxdetectorsegs = BboxDetectorForEach()
                    samdetectorcombined = SAMDetectorCombined()
                    impactsegsandmask = SegsBitwiseAndMask()
                    detailerforeachdebug = DetailerForEachTest()
                    bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                        threshold=0.5,
                        dilation=10,
                        crop_factor=2,
                        drop_size=10,
                        labels="all",
                        bbox_detector=ultralyticsdetectorprovider_151[0],
                        image=vaedecode_240[0],
                    )
                    samdetectorcombined_139 = samdetectorcombined.doit(
                        detection_hint="center-1",
                        dilation=0,
                        threshold=0.93,
                        bbox_expansion=0,
                        mask_hint_threshold=0.7,
                        mask_hint_use_negative="False",
                        sam_model=samloader_87[0],
                        segs=bboxdetectorsegs_132,
                        image=vaedecode_240[0],
                    )
                    if samdetectorcombined_139 is None:
                        return
                    impactsegsandmask_152 = impactsegsandmask.doit(
                        segs=bboxdetectorsegs_132,
                        mask=samdetectorcombined_139[0],
                    )
                    detailerforeachdebug_145 = detailerforeachdebug.doit(
                        guide_size=512,
                        guide_size_for=False,
                        max_size=768,
                        seed=random.randint(1, 2**64),
                        steps=40,
                        cfg=6.5,
                        sampler_name="dpmpp_2m_sde",
                        scheduler="karras",
                        denoise=0.5,
                        feather=5,
                        noise_mask=True,
                        force_inpaint=True,
                        wildcard="",
                        cycle=1,
                        inpaint_model=False,
                        noise_mask_feather=20,
                        image=vaedecode_240[0],
                        segs=impactsegsandmask_152[0],
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    saveimage.save_images(
                        filename_prefix="LD-refined",
                        images=detailerforeachdebug_145[0],
                    )
                    ultralyticsdetectorprovider = UltralyticsDetectorProvider()
                    ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                        model_name="face_yolov9c.pt"
                    )
                    bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                        threshold=0.5,
                        dilation=10,
                        crop_factor=2,
                        drop_size=10,
                        labels="all",
                        bbox_detector=ultralyticsdetectorprovider_151[0],
                        image=detailerforeachdebug_145[0],
                    )
                    samdetectorcombined_139 = samdetectorcombined.doit(
                        detection_hint="center-1",
                        dilation=0,
                        threshold=0.93,
                        bbox_expansion=0,
                        mask_hint_threshold=0.7,
                        mask_hint_use_negative="False",
                        sam_model=samloader_87[0],
                        segs=bboxdetectorsegs_132,
                        image=detailerforeachdebug_145[0],
                    )
                    impactsegsandmask_152 = impactsegsandmask.doit(
                        segs=bboxdetectorsegs_132,
                        mask=samdetectorcombined_139[0],
                    )
                    detailerforeachdebug_145 = detailerforeachdebug.doit(
                        guide_size=512,
                        guide_size_for=False,
                        max_size=768,
                        seed=random.randint(1, 2**64),
                        steps=40,
                        cfg=6.5,
                        sampler_name="dpmpp_2m_sde",
                        scheduler="karras",
                        denoise=0.5,
                        feather=5,
                        noise_mask=True,
                        force_inpaint=True,
                        wildcard="",
                        cycle=1,
                        inpaint_model=False,
                        noise_mask_feather=20,
                        image=detailerforeachdebug_145[0],
                        segs=impactsegsandmask_152[0],
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    saveimage.save_images(
                        filename_prefix="lD-2ndrefined",
                        images=detailerforeachdebug_145[0],
                    )
            else:
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LightDiffusion ")
    parser.add_argument("prompt", type=str, help="The prompt for the ")
    parser.add_argument("width", type=int, help="The width of the generated image.")
    parser.add_argument("height", type=int, help="The height of the generated image.")
    parser.add_argument("number", type=int, help="The number of images to generate.")
    parser.add_argument("batch", type=int, help="The batch size. aka the number of images to generate at once.")
    parser.add_argument(
        "--hires-fix", action="store_true", help="Enable high-resolution fix."
    )
    parser.add_argument(
        "--adetailer",
        action="store_true",
        help="Enable automatic face and body enhancin.g",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enable llama3.2 prompt enhancement. Make sure to have ollama with llama3.2 installed.",
    )
    parser.add_argument(
        "--img2img",
        action="store_true",
        help="Enable image-to-image mode. This will use the image as the prompt.",
    )
    parser.add_argument(
        "--stable-fast",
        action="store_true",
        help="Enable StableFast mode. This will compile the model for faster inference.",
    )
    parser.add_argument(
        "--reuse-seed",
        action="store_true",
        help="Enable to reuse last used seed for sampling, default for False is a random seed at every use.",
    )
    parser.add_argument(
        "--flux",
        action="store_true",
        help="Enable the flux mode.",
    )
    args = parser.parse_args()

    pipeline(
        args.prompt,
        args.width,
        args.height,
        args.number,
        args.batch,
        args.hires_fix,
        args.adetailer,
        args.enhance_prompt,
        args.img2img,
        args.stable_fast,
        args.reuse_seed,
        args.flux,
    )


