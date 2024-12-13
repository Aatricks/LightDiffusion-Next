from enum import Enum
import torch.nn as nn

class TimestepBlock1(nn.Module):
    pass


class TimestepEmbedSequential1(nn.Sequential, TimestepBlock1):
    pass


import math
import torch

from modules import Device, cond, samplers, sampling_util, util


class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
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
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = sampling_util.make_beta_schedule(
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

    def set_sigmas(self, sigmas):
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
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




import threading


class InterruptProcessingException(Exception):
    pass


interrupt_processing_mutex = threading.RLock()

interrupt_processing = False

import torch


import collections


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


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    return True


def can_concat_cond(c1, c2):
    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

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


def calc_cond_batch(model, conds, x_in, timestep, model_options):
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

        free_memory = Device.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
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

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

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
            out_conds[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += (
                output[o] * mult[o]
            )
            out_counts[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds


def cfg_function(
    model,
    cond_pred,
    uncond_pred,
    cond_scale,
    x,
    timestep,
    model_options={},
    cond=None,
    uncond=None,
):
    cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    return cfg_result


def sampling_function(
    model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None
):
    uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)
    return cfg_function(
        model,
        out[0],
        out[1],
        cond_scale,
        x,
        timestep,
        model_options=model_options,
        cond=cond,
        uncond=uncond_,
    )


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas

    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        return out


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


def resolve_areas_and_cond_masks(conditions, h, w, device):
    for i in range(len(conditions)):
        c = conditions[i]


def create_cond_with_same_area_if_none(conds, c):
    if "area" not in c:
        return


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


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        params["width"] = params.get("width", noise.shape[3] * 8)
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


class Sampler:
    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


KSAMPLER_NAMES = [
    "euler_ancestral",
    "dpm_adaptive",
    "dpmpp_2m_sde",
]


class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None
        total_steps = len(sigmas) - 1

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_adaptive":

        def dpm_adaptive_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return samplers.sample_dpm_adaptive(
                model,
                noise,
                sigma_min,
                sigmas[0],
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpm_adaptive_function
    elif sampler_name == "dpmpp_2m_sde":

        def dpmpp_sde_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return samplers.sample_dpmpp_2m_sde(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpmpp_sde_function
    elif sampler_name == "euler_ancestral":

        def euler_ancestral_function(
            model, noise, sigmas, extra_args, callback, disable
        ):
            return samplers.sample_euler_ancestral(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = euler_ancestral_function

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(
    model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None
):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks(conds[k], noise.shape[2], noise.shape[3], device)

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
                            lambda c: c.get("control_apply_to_uncond", False) == True,
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


class CFGGuider:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = cond.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
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
    ):
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
    ):
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = cond.prepare_sampling(
            self.model_patcher, noise.shape, self.conds
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
        )

        cond.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output


def sample(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options={},
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
    )


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


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


def sampler_object(name):
    sampler = ksampler(name)
    return sampler


class KSampler1:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(
        ("dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2")
    )

    def __init__(
        self,
        model,
        steps,
        device,
        sampler=None,
        scheduler=None,
        denoise=None,
        model_options={},
    ):
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        sigmas = calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )

        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1) :]

    def sample(
        self,
        noise,
        positive,
        negative,
        cfg,
        latent_image=None,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        denoise_mask=None,
        sigmas=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        if sigmas is None:
            sigmas = self.sigmas

        sampler = sampler_object(self.sampler)

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
        )


def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu",
    )


def sample1(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    noise_mask=None,
    sigmas=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    sampler = KSampler1(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
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
    )
    samples = samples.to(Device.intermediate_device())
    return samples

def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None

    disable_pbar = not util.PROGRESS_BAR_ENABLED
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
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler2:
    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
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
        )
        
class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
        

def model_sampling(model_config, model_type):
    s = ModelSamplingDiscrete
    if model_type == ModelType.EPS:
        c = EPS

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)