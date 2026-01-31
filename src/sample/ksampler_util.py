"""K-sampler utilities for diffusion models."""
import collections
import logging
import numpy as np
import scipy
import torch
from src.sample import sampling_util


def calculate_start_end_timesteps(model: torch.nn.Module, conds: list) -> None:
    """Calculate start/end timesteps for conditions."""
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]
        ts, te = x.get("start_percent"), x.get("end_percent")
        if ts is not None or te is not None:
            n = x.copy()
            if ts is not None: n["timestep_start"] = s.percent_to_sigma(ts)
            if te is not None: n["timestep_end"] = s.percent_to_sigma(te)
            conds[t] = n


def pre_run_control(model: torch.nn.Module, conds: list) -> None:
    """Pre-run control for conditions."""
    s = model.model_sampling
    for x in conds:
        if "control" in x:
            x["control"].pre_run(model, lambda a: s.percent_to_sigma(a))


def apply_empty_x_to_equal_area(conds: list, uncond: list, name: str, uncond_fill_func: callable) -> None:
    """Apply empty x to equal area."""
    cond_cnets, cond_other = [], []
    uncond_cnets, uncond_other = [], []
    
    for t, x in enumerate(conds):
        if "area" not in x:
            (cond_cnets if name in x and x[name] else cond_other).append((x[name], None) if name in x and x[name] else (x, t))
    for t, x in enumerate(uncond):
        if "area" not in x:
            (uncond_cnets if name in x and x[name] else uncond_other).append((x[name], None) if name in x and x[name] else (x, t))
    
    if uncond_cnets: return
    for i, _ in enumerate(cond_cnets):
        temp = uncond_other[i % len(uncond_other)]
        n = temp[0].copy()
        n[name] = uncond_fill_func([c[0] for c in cond_cnets if c[1] is None], i)
        if temp[1] is not None: uncond[temp[1]] = n
        else: uncond.append(n)


CondObj = collections.namedtuple("cond_obj", ["input_x", "mult", "conditioning", "area", "control", "patches", "batch_indices"])


def get_area_and_mult(conds: dict, x_in: torch.Tensor, timestep_in: int) -> CondObj:
    """Get area and multiplier from conditions."""
    x_shape, device = x_in.shape, x_in.device
    area = (x_shape[2], x_shape[3], 0, 0)
    batch_indices = conds.get("batch_index")
    if isinstance(batch_indices, int): batch_indices = [batch_indices]
    
    area_h, area_w = max(0, min(int(area[0]), x_shape[2])), max(0, min(int(area[1]), x_shape[3]))
    area = (area_h, area_w, 0, 0)
    
    if batch_indices is None:
        input_x = x_in[:, :, :area_h, :area_w]
    else:
        try:
            mapped = [(int(b) if b >= 0 else x_shape[0] + int(b)) for b in batch_indices]
            valid = [b for b in mapped if 0 <= b < x_shape[0]]
            if not valid:
                batch_indices = None
                input_x = x_in[:, :, :area_h, :area_w]
            else:
                input_x = x_in[torch.tensor(valid, dtype=torch.long, device=device), :, :area_h, :area_w]
        except Exception:
            batch_indices = None
            input_x = x_in[:, :, :area_h, :area_w]
    
    mult = torch.ones_like(input_x)
    batch_size = x_shape[0] if batch_indices is None else len(batch_indices)
    conditioning = {c: conds["model_conds"][c].process_cond(batch_size=batch_size, device=device, area=area) 
                   for c in conds["model_conds"]}
    
    return CondObj(input_x, mult, conditioning, area, conds.get("control"), None, batch_indices)


def normal_scheduler(model_sampling, steps: int, sgm: bool = False, floor: bool = False) -> torch.FloatTensor:
    """Create normal noise scheduler."""
    s = model_sampling
    timesteps = torch.linspace(s.timestep(s.sigma_max), s.timestep(s.sigma_min), steps, device=s.sigmas.device)
    return torch.cat([s.sigma(timesteps), s.sigmas.new_zeros([1])]).cpu().float()


def simple_scheduler(model_sampling, steps: int) -> torch.FloatTensor:
    """Create simple noise scheduler."""
    s = model_sampling
    if steps <= 0: return torch.FloatTensor([0.0])
    indices = (torch.arange(steps, device=s.sigmas.device) * len(s.sigmas) / steps).long()
    sigs = s.sigmas.flip(0)[indices]
    return torch.cat([sigs, sigs.new_zeros([1])]).cpu().float()


def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6) -> torch.FloatTensor:
    """Create beta distribution noise scheduler."""
    total = len(model_sampling.sigmas) - 1
    ts = scipy.stats.beta.ppf(1 - np.linspace(0, 1, steps, endpoint=False), alpha, beta)
    ts_indices = np.rint(ts * total).astype(np.int32)
    unique_ts, indices = np.unique(ts_indices, return_index=True)
    ordered = unique_ts[np.argsort(indices)]
    sigs = model_sampling.sigmas[torch.from_numpy(ordered).to(model_sampling.sigmas.device, torch.long)]
    return torch.cat([sigs, sigs.new_zeros([1])]).cpu().float()


def calculate_sigmas(model_sampling, scheduler_name: str, steps: int) -> torch.Tensor:
    """Calculate sigmas for scheduler."""
    if scheduler_name == "karras":
        return sampling_util.get_sigmas_karras(steps, float(model_sampling.sigma_min), float(model_sampling.sigma_max))
    elif scheduler_name == "normal":
        return normal_scheduler(model_sampling, steps)
    elif scheduler_name == "simple":
        return simple_scheduler(model_sampling, steps)
    elif scheduler_name == "beta":
        return beta_scheduler(model_sampling, steps)
    elif scheduler_name in ["ays", "ays_sd15", "ays_sdxl", "ays_flux"]:
        from src.sample import ays_scheduler as ays
        model_type = {"ays_sdxl": "SDXL", "ays_flux": "FLUX", "ays_sd15": "SD15"}.get(scheduler_name)
        if not model_type:
            try:
                config = getattr(getattr(model_sampling, 'model_config', None), 'unet_config', {})
                model_type = "SDXL" if config.get('context_dim', 768) == 2048 else "SD15"
            except: model_type = "SD15"
        return ays.ays_scheduler(model_sampling, steps, model_type)
    logging.error(f"Invalid scheduler: {scheduler_name}")
    return None


def prepare_noise(latent_image: torch.Tensor, seed: int, noise_inds: list = None, 
                  seeds_per_sample: list | None = None) -> torch.Tensor:
    """Prepare noise for latent image."""
    if seeds_per_sample is not None:
        sps = np.array(seeds_per_sample)
        if sps.shape[0] != latent_image.size(0):
            raise ValueError("seeds_per_sample length must match latent batch size")
        unique_seeds, inverse = np.unique(sps, return_inverse=True)
        noises = []
        for us in unique_seeds:
            g = torch.Generator()
            g.manual_seed(int(us))
            noises.append(torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype,
                                      layout=latent_image.layout, generator=g, device=latent_image.device))
        return torch.cat([noises[i] for i in inverse], axis=0)

    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                          generator=generator, device=latent_image.device)

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype,
                           layout=latent_image.layout, generator=generator, device=latent_image.device)
        if i in unique_inds: noises.append(noise)
    return torch.cat([noises[i] for i in inverse], axis=0)
