import collections

import torch

from modules.sample import sampling_util

def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]


def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)


def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if "area" not in x:
            cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if "area" not in x:
            uncond_other.append((x, t))
            
def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
    mask = torch.ones_like(input_x)
    mult = mask * strength

    if "mask" not in conds:
        rr = 8

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


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
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

def calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = sampling_util.get_sigmas_karras(
            n=steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model_sampling, steps)
    return sigmas

def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu",
    )