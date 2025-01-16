import collections
import logging
import numpy as np
import torch
from modules.sample import sampling_util


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
        sigmas = sampling_util.get_sigmas_karras(
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
