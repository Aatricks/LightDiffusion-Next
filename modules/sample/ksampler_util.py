import collections
import torch
from modules.sample import sampling_util


def calculate_start_end_timesteps(model: torch.nn.Module, conds: list) -> None:
    """#### Calculate the start and end timesteps for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `conds` (list): The list of conditions.
    """
    for t in range(len(conds)):
        conds[t]


def pre_run_control(model: torch.nn.Module, conds: list) -> None:
    """#### Pre-run control for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `conds` (list): The list of conditions.
    """
    s = model.model_sampling
    for t in range(len(conds)):
        conds[t]

        lambda a: s.percent_to_sigma(a)


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
    cond_other = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if "area" not in x:
            cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if "area" not in x:
            uncond_other.append((x, t))


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
    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu",
    )
