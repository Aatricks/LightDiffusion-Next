from __future__ import annotations
import logging
import math
import os
import torch

import packaging.version

from typing import Callable, List, Optional, Protocol, TypedDict

from modules.Utilities import Latent
from modules.Device import Device
from modules.cond import cast, cond, cond_util
from modules.sample import CFG, ksampler_util, sampling


if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.12.0"):
    torch.backends.cuda.matmul.allow_tf32 = True

supported_pt_extensions = set([".ckpt", ".pt", ".bin", ".pth", ".safetensors", ".pkl"])

folder_names_and_paths = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "_internal")
folder_names_and_paths["checkpoints"] = (
    [os.path.join(models_dir, "checkpoints")],
    supported_pt_extensions,
)

folder_names_and_paths["loras"] = (
    [os.path.join(models_dir, "loras")],
    supported_pt_extensions,
)

folder_names_and_paths["ERSGAN"] = (
    [os.path.join(models_dir, "ERSGAN")],
    supported_pt_extensions,
)

output_directory = "./_internal/output"

filename_list_cache = {}

class Empty:
    pass


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None

logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)


KSAMPLER_NAMES = [
    "euler_ancestral",
    "dpm_adaptive",
    "dpmpp_2m_sde",
]


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
        return CFG.sampling_function(
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
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = cond.process_conds(
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
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = cond_util.prepare_sampling(
            self.model_patcher, noise.shape, self.conds, flux_enabled=True
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

        cond_util.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output


def sample(
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
    pipeline=False,
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
        pipeline=pipeline,
    )
    samples = samples.to(Device.intermediate_device())
    return samples


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


def sampler_object(name, pipeline=False):
    sampler = sampling.ksampler(name, pipeline=pipeline)
    return sampler


def sample1(
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
    pipeline=False,
):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, pipeline=pipeline
    )

ops = cast.disable_weight_init

_ATTN_PRECISION = "fp32"

    
ops = cast.manual_cast


MAX_RESOLUTION = 16384


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
    pipeline=False,
):
    latent_image = latent["samples"]
    latent_image = Latent.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = ksampler_util.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]
    samples = sample(
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


ops = cast.disable_weight_init


KSAMPLER_NAMES = [
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
]

SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

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
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = ksampler_util.calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
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
        pipeline=False,
    ):
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

        return sample1(
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
        )


class KSampler:
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
        pipeline=False,
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
            pipeline=pipeline,
        )