import importlib
from inspect import isfunction
import itertools
import logging
import math
import os
import safetensors.torch
import torch

# Global folder paths for LoRA/embeddings/etc.
# Maps folder_name -> ([list_of_paths], set_of_extensions)
folder_names_and_paths = {
    "loras": ([os.path.join(".", "include", "loras")], {".safetensors", ".ckpt", ".pt"}),
    "embeddings": ([os.path.join(".", "include", "embeddings")], {".safetensors", ".pt", ".bin"}),
    "checkpoints": ([os.path.join(".", "include", "checkpoints")], {".safetensors", ".ckpt"}),
    "vae": ([os.path.join(".", "include", "vae")], {".safetensors", ".ckpt", ".pt"}),
}


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Append dimensions to tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    expanded = x[(...,) + (None,) * dims_to_append]
    return expanded.detach().clone() if expanded.device.type == "mps" else expanded


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Convert tensor to denoised tensor: (x - denoised) / sigma."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def load_torch_file(ckpt: str, safe_load: bool = False, device: str = None) -> dict:
    """Load a PyTorch checkpoint file (.safetensors or .pt/.ckpt)."""
    from src.Device.ModelCache import get_model_cache
    cache = get_model_cache()
    prefetched = cache.get_prefetched_model(ckpt)
    if prefetched is not None:
        cache.clear_prefetch()
        return prefetched

    if device is None:
        device = torch.device("cpu")
    
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if "weights_only" not in torch.load.__code__.co_varnames:
                logging.warning("torch.load doesn't support weights_only, loading unsafely.")
                safe_load = False
        
        load_device = "cpu"
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=load_device, weights_only=True)
        else:
            kwargs = {"map_location": load_device}
            if "weights_only" in torch.load.__code__.co_varnames:
                kwargs["weights_only"] = False
            pl_sd = torch.load(ckpt, **kwargs)
        
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        
        sd = pl_sd.get("state_dict", pl_sd)
        
        if device.type == "cuda":
            for k in sd:
                if isinstance(sd[k], torch.Tensor):
                    sd[k] = sd[k].pin_memory()
    return sd


def calculate_parameters(sd: dict, prefix: str = "") -> int:
    """Count total parameters in state dict with given prefix."""
    return sum(sd[k].nelement() for k in sd.keys() if k.startswith(prefix))


def state_dict_prefix_replace(state_dict: dict, replace_prefix: dict, filter_keys: bool = False) -> dict:
    """Replace key prefixes in state dict. O(N) optimized."""
    out = {} if filter_keys else state_dict
    to_replace = []
    for k in list(state_dict.keys()):
        for rp, new_rp in replace_prefix.items():
            if k.startswith(rp):
                to_replace.append((k, rp, new_rp))
                break
    for old_k, rp, new_rp in to_replace:
        out[new_rp + old_k[len(rp):]] = state_dict.pop(old_k)
    return out


def lcm_of_list(numbers):
    """Calculate LCM of a list of numbers."""
    return math.lcm(*numbers) if numbers else 1


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int, dim: int = 0) -> torch.Tensor:
    """Repeat tensor to match batch_size along dim."""
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        repeats = dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)
        return tensor.repeat(*repeats).narrow(dim, 0, batch_size)
    return tensor


def set_attr(obj: object, attr: str, value) -> any:
    """Set nested attribute (dot-separated), return previous value."""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj: object, attr: str, value) -> any:
    """Set nested attribute as nn.Parameter."""
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))


def copy_to_param(obj: object, attr: str, value) -> None:
    """Copy value to existing parameter's data."""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    getattr(obj, attrs[-1]).data.copy_(value)


def get_obj_from_str(string: str, reload: bool = False) -> object:
    """Import and return object from 'module.class' string."""
    module, cls = string.rsplit(".", 1)
    if reload:
        importlib.reload(importlib.import_module(module))
    return getattr(importlib.import_module(module), cls)


def get_attr(obj: object, attr: str) -> any:
    """Get nested attribute (dot-separated)."""
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


def lcm(a: int, b: int) -> int:
    """Least common multiple of a and b."""
    return math.lcm(a, b)


def get_full_path(folder_name: str, filename: str) -> str:
    """Get full path of file in folder."""
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def zero_module(module: torch.nn.Module) -> torch.nn.Module:
    """Zero out all parameters of a module."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def append_zero(x: torch.Tensor) -> torch.Tensor:
    """Append a zero to tensor."""
    return torch.cat([x, x.new_zeros([1])])


def exists(val) -> bool:
    """Check if value is not None."""
    return val is not None


def default(val, d):
    """Return val if exists, else d (or d() if callable)."""
    return val if exists(val) else (d() if isfunction(d) else d)


def write_parameters_to_file(prompt_entry: str, neg: str, width: int, height: int, cfg: int) -> None:
    """Write generation parameters to file."""
    with open("./include/prompt.txt", "w") as f:
        f.write(f"prompt: {prompt_entry}\nneg: {neg}\nw: {int(width)}\nh: {int(height)}\ncfg: {int(cfg)}\n")


def load_parameters_from_file() -> tuple:
    """Load generation parameters from file."""
    with open("./include/prompt.txt", "r") as f:
        params = {}
        for line in f:
            if line.strip():
                key, value = line.split(": ", 1)
                params[key] = value.strip()
    return params["prompt"], params["neg"], int(params["w"]), int(params["h"]), int(params["cfg"])


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None


class ProgressBar:
    """Progress bar wrapper."""
    def __init__(self, total: int):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK


def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int) -> int:
    """Calculate number of tiles for tiled scaling."""
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols


@torch.inference_mode()
def tiled_scale_multidim(
    samples: torch.Tensor, function, tile: tuple = (64, 64), overlap: int = 8,
    upscale_amount: int = 4, out_channels: int = 3, output_device: str = "cpu",
    downscale: bool = False, index_formulas=None, pbar=None
):
    """Scale tensor using tiled approach with multi-dimensional support."""
    dims = len(tile)
    upscale_amount = [upscale_amount] * dims if not isinstance(upscale_amount, (tuple, list)) else upscale_amount
    overlap = [overlap] * dims if not isinstance(overlap, (tuple, list)) else overlap
    index_formulas = upscale_amount if index_formulas is None else index_formulas
    index_formulas = [index_formulas] * dims if not isinstance(index_formulas, (tuple, list)) else index_formulas

    def get_scale(dim, val):
        up = upscale_amount[dim]
        return up(val) if callable(up) else (val / up if downscale else up * val)

    def get_pos(dim, val):
        up = index_formulas[dim]
        return up(val) if callable(up) else (val / up if downscale else up * val)

    def mult_list_upscale(a):
        return [round(get_scale(i, a[i])) for i in range(len(a))]

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b:b+1]
        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b:b+1] = function(s).to(output_device)
            if pbar: pbar.update(1)
            continue

        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros_like(out)

        positions = [
            range(0, s.shape[d+2] - overlap[d], tile[d] - overlap[d]) if s.shape[d+2] > tile[d] else [0]
            for d in range(dims)
        ]

        for it in itertools.product(*positions):
            s_in, upscaled = s, []
            for d in range(dims):
                pos = max(0, min(s.shape[d+2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d+2] - pos)
                s_in = s_in.narrow(d+2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d-2, overlap[d-2]))
                if feather < mask.shape[d]:
                    for t in range(feather):
                        a = (t + 1) / feather
                        mask.narrow(d, t, 1).mul_(a)
                        mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o, o_d = out, out_div
            for d in range(dims):
                o = o.narrow(d+2, upscaled[d], mask.shape[d+2])
                o_d = o_d.narrow(d+2, upscaled[d], mask.shape[d+2])
            o.add_(ps * mask)
            o_d.add_(mask)
            if pbar: pbar.update(1)

        output[b:b+1] = out / out_div
    return output


def tiled_scale(samples: torch.Tensor, function, tile_x: int = 64, tile_y: int = 64,
                overlap: int = 8, upscale_amount: int = 4, out_channels: int = 3,
                output_device: str = "cpu", pbar=None):
    """Scale tensor using tiled approach (2D convenience wrapper)."""
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap,
                                 upscale_amount=upscale_amount, out_channels=out_channels,
                                 output_device=output_device, pbar=pbar)


def transformers_convert(
    sd: dict, prefix_from: str, prefix_to: str, number: int
) -> dict:
    """Convert transformers state dict from one prefix to another.

    Args:
        sd: State dictionary
        prefix_from: Source prefix
        prefix_to: Destination prefix
        number: Number of transformer blocks

    Returns:
        Converted state dictionary
    """
    keys_to_replace = {
        "{}positional_embedding": "{}embeddings.position_embedding.weight",
        "{}token_embedding.weight": "{}embeddings.token_embedding.weight",
        "{}ln_final.weight": "{}final_layer_norm.weight",
        "{}ln_final.bias": "{}final_layer_norm.bias",
    }

    for k in keys_to_replace:
        x = k.format(prefix_from)
        if x in sd:
            sd[keys_to_replace[k].format(prefix_to)] = sd.pop(x)

    resblock_to_replace = {
        "ln_1": "layer_norm1",
        "ln_2": "layer_norm2",
        "mlp.c_fc": "mlp.fc1",
        "mlp.c_proj": "mlp.fc2",
        "attn.out_proj": "self_attn.out_proj",
    }

    for resblock in range(number):
        for x in resblock_to_replace:
            for y in ["weight", "bias"]:
                k = "{}transformer.resblocks.{}.{}.{}".format(
                    prefix_from, resblock, x, y
                )
                k_to = "{}encoder.layers.{}.{}.{}".format(
                    prefix_to, resblock, resblock_to_replace[x], y
                )
                if k in sd:
                    sd[k_to] = sd.pop(k)

        for y in ["weight", "bias"]:
            k_from = "{}transformer.resblocks.{}.attn.in_proj_{}".format(
                prefix_from, resblock, y
            )
            if k_from in sd:
                weights = sd.pop(k_from)
                shape_from = weights.shape[0] // 3
                for x in range(3):
                    p = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    k_to = "{}encoder.layers.{}.{}.{}".format(prefix_to, resblock, p[x], y)
                    sd[k_to] = weights[shape_from * x : shape_from * (x + 1)]

    return sd


def clip_text_transformers_convert(
    sd: dict, prefix_from: str, prefix_to: str
) -> dict:
    """Convert CLIP text transformers state dict.

    Args:
        sd: State dictionary
        prefix_from: Source prefix
        prefix_to: Destination prefix

    Returns:
        Converted state dictionary
    """
    sd = transformers_convert(sd, prefix_from, "{}text_model.".format(prefix_to), 32)

    tp = "{}text_projection.weight".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = sd.pop(tp)

    tp = "{}text_projection".format(prefix_from)
    if tp in sd:
        sd["{}text_projection.weight".format(prefix_to)] = (
            sd.pop(tp).transpose(0, 1).contiguous()
        )
    return sd
