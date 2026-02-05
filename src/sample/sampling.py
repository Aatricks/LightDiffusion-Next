"""Sampling implementation for diffusion models - Simplified architecture."""
import math
from enum import Enum
import torch
import torch.nn as nn
from src.Device import Device
from src.sample import CFG, ksampler_util, sampling_util
from src.sample.BaseSampler import (
    EulerSampler, EulerAncestralSampler, DPMPP2MSampler, DPMPPSDESampler
)
from src.Utilities import Latent


class TimestepEmbedSequential1(nn.Sequential):
    """Sequential module that passes timestep embeddings to children that need them."""
    def forward(self, x, emb=None, context=None, transformer_options={}, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
        for layer in self:
            if hasattr(layer, 'forward'):
                import inspect
                sig = inspect.signature(layer.forward)
                params = list(sig.parameters.keys())
                if 'emb' in params or 'temb' in params:
                    x = layer(x, emb)
                elif 'context' in params:
                    x = layer(x, context=context, transformer_options=transformer_options)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x


# Noise prediction strategies
class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            return noise * torch.sqrt(1.0 + sigma**2.0) + latent_image
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise * sigma + latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent


def reshape_sigma(sigma, noise_dim):
    """Reshape sigma for broadcasting with noise tensor.
    
    Matches ComfyUI's implementation to handle both scalar and batch sigmas.
    """
    if sigma.nelement() == 1:
        return sigma.view(())
    else:
        return sigma.view(sigma.shape[:1] + (1,) * (noise_dim - 1))


class CONST:
    """CONST noise prediction for flow matching models (Flux)."""
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = reshape_sigma(sigma, model_output.ndim)
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = reshape_sigma(sigma, noise.ndim)
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        sigma = reshape_sigma(sigma, latent.ndim)
        return latent / (1.0 - sigma)


def time_snr_shift(alpha, t):
    """SNR shift function for FLOW models (not Flux).
    
    Used by ModelSamplingDiscreteFlow, NOT ModelSamplingFlux.
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def flux_time_shift(mu, sigma, t):
    """Time shift function for Flux models (matches ComfyUI exactly).
    
    This is the correct formula for Flux1 and Flux2 models.
    
    Args:
        mu: Shift parameter (1.15 for Flux1, 2.02 for Flux2)
        sigma: Sigma parameter (typically 1.0)
        t: Timestep normalized to [0, 1]
    
    Returns:
        Shifted sigma value
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux(torch.nn.Module):
    """Model sampling for Flux1 models."""
    def __init__(self, model_config=None):
        super().__init__()
        shift = model_config.sampling_settings.get("shift", 1.15) if model_config else 1.15
        self.shift = shift
        # Use 10000 timesteps like ComfyUI ModelSamplingFlux
        ts = self.sigma(torch.arange(1, 10001, 1) / 10000)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        # Flux returns sigma directly as timestep (no multiplier)
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)
    
    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        s = model_config.sampling_settings
        betas = sampling_util.make_beta_schedule(
            s.get("beta_schedule", "linear"), 1000,
            linear_start=s.get("linear_start", 0.00085),
            linear_end=s.get("linear_end", 0.012))
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())
        self.sigma_data = 1.0

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log().to(self.log_sigmas.device)
        idx = torch.searchsorted(self.log_sigmas, log_sigma)
        idx_high = idx.clamp(0, len(self.log_sigmas) - 1)
        idx_low = (idx - 1).clamp(0, len(self.log_sigmas) - 1)
        return torch.where(
            (log_sigma - self.log_sigmas[idx_high]).abs() < (log_sigma - self.log_sigmas[idx_low]).abs(),
            idx_high, idx_low).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(timestep.float().to(self.log_sigmas.device), 0, len(self.sigmas) - 1)
        low, high, w = t.floor().long(), t.ceil().long(), t.frac()
        return ((1 - w) * self.log_sigmas[low] + w * self.log_sigmas[high]).exp().to(timestep.device)

    def percent_to_sigma(self, percent):
        if percent <= 0.0: return 999999999.9
        if percent >= 1.0: return 0.0
        t = (1.0 - percent) * 999.0
        t = max(0.0, min(t, len(self.sigmas) - 1))
        low, w = int(t), t - int(t)
        high = min(low + 1, len(self.sigmas) - 1)
        return math.exp((1 - w) * self.log_sigmas[low].item() + w * self.log_sigmas[high].item())


# Sampler wrapper using class-based samplers
class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
        self.latent_image = None
        self.noise = None

    def __call__(self, x, sigma, denoise_mask=None, model_options={}, seed=None):
        return self.inner_model(x, sigma, model_options=model_options, seed=seed)


class KSAMPLER:
    def __init__(self, sampler_class, extra_options={}):
        self.sampler_class = sampler_class
        self.extra_options = extra_options

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        return math.isclose(max_sigma, float(sigmas[0]), rel_tol=1e-05) or float(sigmas[0]) > max_sigma

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False, pipeline=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))
        
        # Create sampler instance with options
        sampler = self.sampler_class(pipeline=pipeline, **self.extra_options)
        samples = sampler.sample(model_k, noise, sigmas, extra_args=extra_args, callback=callback, disable=disable_pbar)
        return model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)


# Sampler registry - using class-based samplers
SAMPLER_CLASSES = {
    "euler": EulerSampler,
    "euler_ancestral": EulerAncestralSampler,
    "euler_cfgpp": EulerSampler,
    "euler_ancestral_cfgpp": EulerAncestralSampler,
    "dpmpp_2m": DPMPP2MSampler,
    "dpmpp_2m_cfgpp": DPMPP2MSampler,
    "dpmpp_sde": DPMPPSDESampler,
    "dpmpp_sde_cfgpp": DPMPPSDESampler,
}


def ksampler(sampler_name, pipeline=False, extra_options={}):
    sampler_class = SAMPLER_CLASSES.get(sampler_name, EulerSampler)
    if sampler_name not in SAMPLER_CLASSES:
        print(f"Warning: Unknown sampler '{sampler_name}', using euler")
    return KSAMPLER(sampler_class, extra_options)


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={},
           latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None, pipeline=False,
           flux=False, cfg_free_enabled=False, cfg_free_start_percent=70.0, batched_cfg=True,
           dynamic_cfg_rescaling=False, dynamic_cfg_method="variance", dynamic_cfg_percentile=95,
           dynamic_cfg_target_scale=1.0, adaptive_noise_enabled=False, adaptive_noise_method="complexity"):
    model_options = model_options.copy()
    model_options["batched_cfg"] = batched_cfg
    cfg_guider = CFG.CFGGuider(model, flux=flux, dynamic_cfg_rescaling=dynamic_cfg_rescaling,
                                dynamic_cfg_method=dynamic_cfg_method, dynamic_cfg_percentile=dynamic_cfg_percentile,
                                dynamic_cfg_target_scale=dynamic_cfg_target_scale, adaptive_noise_enabled=adaptive_noise_enabled,
                                adaptive_noise_method=adaptive_noise_method)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    cfg_guider.set_cfg_free_params(cfg_free_enabled, cfg_free_start_percent)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, pipeline=pipeline)


class KSampler:
    def __init__(self, model=None, steps=None, sampler=None, scheduler=None, denoise=1.0, model_options={}, pipeline=False):
        self.model = model
        self.device = model.load_device if model else None
        self.scheduler = scheduler
        self.sampler_name = sampler
        self.denoise = denoise
        self.model_options = model_options
        self.pipeline = pipeline
        if model and steps:
            self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps):
        return ksampler_util.calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        elif denoise <= 0.0:
            self.sigmas = torch.FloatTensor([])
        else:
            new_steps = int(steps / denoise)
            self.sigmas = self.calculate_sigmas(new_steps).to(self.device)[-(steps + 1):]

    def direct_sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None,
                      force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False,
                      seed=None, flux=False, cfg_free_enabled=False, cfg_free_start_percent=70.0):
        sigmas = sigmas if sigmas is not None else self.sigmas
        if last_step is not None and last_step < len(sigmas) - 1:
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise: sigmas[-1] = 0
        if start_step is not None and start_step < len(sigmas):
            sigmas = sigmas[start_step:]
        if start_step is not None and start_step >= len(sigmas) - 1:
            return latent_image if latent_image is not None else torch.zeros_like(noise)
        return sample(self.model, noise, positive, negative, cfg, self.device, ksampler(self.sampler_name, self.pipeline),
                      sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback,
                      disable_pbar=disable_pbar, seed=seed, pipeline=self.pipeline, flux=flux,
                      cfg_free_enabled=cfg_free_enabled, cfg_free_start_percent=cfg_free_start_percent)

    def sample(self, model=None, seed=None, steps=None, cfg=None, sampler_name=None, scheduler=None,
               positive=None, negative=None, latent_image=None, denoise=None, start_step=None, last_step=None,
               force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=False, disable_noise=False,
               pipeline=False, flux=False, flux2=False, enable_multiscale=True, multiscale_factor=0.5,
               multiscale_fullres_start=3, multiscale_fullres_end=8, multiscale_intermittent_fullres=False,
               cfg_free_enabled=False, cfg_free_start_percent=70.0, batched_cfg=True, dynamic_cfg_rescaling=False,
               dynamic_cfg_method="variance", dynamic_cfg_percentile=95.0, dynamic_cfg_target_scale=7.0,
               adaptive_noise_enabled=False, adaptive_noise_method="complexity"):
        if model is None:
            if latent_image is None:
                raise ValueError("latent_image must be provided when using pre-initialized model")
            return (self.direct_sample(None, positive, negative, cfg, latent_image, start_step, last_step,
                                        force_full_denoise, noise_mask, None, callback, disable_pbar, seed, flux,
                                        cfg_free_enabled, cfg_free_start_percent),)
        latent = latent_image if isinstance(latent_image, dict) else {"samples": latent_image}
        return common_ksampler(model, seed, steps, cfg, sampler_name or self.sampler_name, scheduler or self.scheduler,
                               positive, negative, latent, denoise or self.denoise, disable_noise, start_step, last_step,
                               force_full_denoise, pipeline or self.pipeline, flux, flux2, enable_multiscale, multiscale_factor,
                               multiscale_fullres_start, multiscale_fullres_end, multiscale_intermittent_fullres,
                               cfg_free_enabled, cfg_free_start_percent, batched_cfg, dynamic_cfg_rescaling,
                               dynamic_cfg_method, dynamic_cfg_percentile, dynamic_cfg_target_scale,
                               adaptive_noise_enabled, adaptive_noise_method)
        return sample(self.model, noise, positive, negative, cfg, self.device, ksampler(self.sampler_name, self.pipeline),
                      sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback,
                      disable_pbar=disable_pbar, seed=seed, pipeline=self.pipeline, flux=flux,
                      cfg_free_enabled=cfg_free_enabled, cfg_free_start_percent=cfg_free_start_percent)


MULTISCALE_SAMPLERS = ["dpmpp_sde_cfgpp", "euler_ancestral", "euler", "dpmpp_2m_cfgpp"]


def sample1(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
            disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None,
            sigmas=None, callback=None, disable_pbar=False, seed=None, pipeline=False, flux=False, flux2=False,
            enable_multiscale=True, multiscale_factor=0.5, multiscale_fullres_start=3, multiscale_fullres_end=8,
            multiscale_intermittent_fullres=False, cfg_free_enabled=False, cfg_free_start_percent=70.0,
            batched_cfg=True, dynamic_cfg_rescaling=False, dynamic_cfg_method="variance", dynamic_cfg_percentile=95,
            dynamic_cfg_target_scale=1.0, adaptive_noise_enabled=False, adaptive_noise_method="complexity",
            model_options=None):
    
    # Auto-detect Flux/Flux2 to disable multi-scale (DiT architecture compatibility)
    model_sampling_obj = getattr(model.model, "model_sampling", None)
    is_flux_sampling = isinstance(model_sampling_obj, (ModelSamplingFlux, ModelSamplingFlux2))
    if flux or flux2 or is_flux_sampling:
        enable_multiscale = False
        flux = True # Ensure flux mode is enabled if detected via sampling object

    extra_options = {"enable_multiscale": enable_multiscale, "multiscale_factor": multiscale_factor,
                     "multiscale_fullres_start": multiscale_fullres_start, "multiscale_fullres_end": multiscale_fullres_end,
                     "multiscale_intermittent_fullres": multiscale_intermittent_fullres}

    sampler_obj = ksampler(sampler_name, pipeline=pipeline, extra_options=extra_options)
    
    # For Flux2, calculate sigmas using resolution-aware scheduler (matches ComfyUI Flux2Scheduler)
    if flux2:
        # Flux2 uses 16x16 patches, but the VAE latent in the pipeline is 8x downscaled (32 channels)
        # Calculate original pixel dimensions: H/8 * 8 = H
        height = latent_image.shape[2] * 8
        width = latent_image.shape[3] * 8
        sigmas = ksampler_util.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps, 
                                                 width=width, height=height, is_flux2=True)
    else:
        sigmas = ksampler_util.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)
    
    if denoise is not None and denoise <= 0.9999:
        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            # For Flux2, use resolution-aware scheduler even with partial denoise
            if flux2:
                height = latent_image.shape[2] * 8
                width = latent_image.shape[3] * 8
                sigmas = ksampler_util.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, 
                                                        int(steps / denoise), width=width, height=height, is_flux2=True)[-(steps + 1):]
            else:
                sigmas = ksampler_util.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, int(steps / denoise))[-(steps + 1):]
    
    if last_step is not None and last_step < len(sigmas) - 1:
        sigmas = sigmas[:last_step + 1]
        if force_full_denoise: sigmas[-1] = 0
    if start_step is not None and start_step < len(sigmas):
        sigmas = sigmas[start_step:]

    # Use provided model_options or default to model's own
    m_opts = model_options if model_options is not None else model.model_options

    load_device = model.load_device
    if not isinstance(load_device, (torch.device, str)):
        load_device = Device.get_torch_device() # Fallback

    samples = sample(model, noise, positive, negative, cfg, load_device, sampler_obj, sigmas.to(load_device),
                     m_opts, latent_image=latent_image, denoise_mask=noise_mask, callback=callback,  
                     disable_pbar=disable_pbar, seed=seed, pipeline=pipeline, flux=flux or flux2,    
                     cfg_free_enabled=cfg_free_enabled, cfg_free_start_percent=cfg_free_start_percent,
                     batched_cfg=batched_cfg, dynamic_cfg_rescaling=dynamic_cfg_rescaling,
                     dynamic_cfg_method=dynamic_cfg_method, dynamic_cfg_percentile=dynamic_cfg_percentile,
                     dynamic_cfg_target_scale=dynamic_cfg_target_scale, adaptive_noise_enabled=adaptive_noise_enabled,
                     adaptive_noise_method=adaptive_noise_method)
    return samples.to(Device.intermediate_device())


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    EDM = 3
    FLUX = 8
    FLUX2 = 9  # Flux2 Klein


class ModelSamplingFlux2(torch.nn.Module):
    """Model sampling for Flux2 (Klein) models with different shift default.
    
    Uses flux_time_shift formula matching ComfyUI's ModelSamplingFlux.
    The shift parameter for Flux2 is 2.02 (different from Flux1's 1.15).
    """
    def __init__(self, model_config=None, shift=None):
        super().__init__()
        # Flux2 default shift is 2.02 (different from Flux1's 1.15)
        if shift is not None:
            self.shift = shift
        elif model_config and hasattr(model_config, 'sampling_settings'):
            self.shift = model_config.sampling_settings.get("shift", 2.02)
        else:
            self.shift = 2.02  # Flux2 default
        # Use 10000 timesteps like ComfyUI ModelSamplingFlux
        ts = self.sigma(torch.arange(1, 10001, 1) / 10000)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        # Flux returns sigma directly as timestep (no multiplier)
        # Shift is already applied in the scheduler (Flux2Scheduler)
        return sigma

    def sigma(self, timestep):
        # Use flux_time_shift formula (matching ComfyUI ModelSamplingFlux)
        return flux_time_shift(self.shift, 1.0, timestep)
    
    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return flux_time_shift(self.shift, 1.0, 1.0 - percent)


def model_sampling(model_config, model_type, flux=False, flux2=False):
    if flux2:
        class ModelSampling(ModelSamplingFlux2, CONST):
            pass
        return ModelSampling(model_config)
    elif flux:
        class ModelSampling(ModelSamplingFlux, CONST):
            pass
        return ModelSampling(model_config)
    else:
        class ModelSampling(ModelSamplingDiscrete, EPS):
            pass
        return ModelSampling(model_config)


def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None,
                  callback=None, disable_pbar=False, seed=None, pipeline=False):
    return sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas,
                  model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask,
                  callback=callback, disable_pbar=disable_pbar, seed=seed, pipeline=pipeline).to(Device.intermediate_device())


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0,
                    disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, pipeline=False,
                    flux=False, flux2=False, enable_multiscale=True, multiscale_factor=0.5, multiscale_fullres_start=3,
                    multiscale_fullres_end=8, multiscale_intermittent_fullres=False, cfg_free_enabled=False,
                    cfg_free_start_percent=70.0, batched_cfg=True, dynamic_cfg_rescaling=False,
                    dynamic_cfg_method="variance", dynamic_cfg_percentile=95.0, dynamic_cfg_target_scale=7.0,
                    adaptive_noise_enabled=False, adaptive_noise_method="complexity", model_options=None):
    
    # Auto-detect Flux/Flux2 to disable multi-scale
    model_sampling_obj = getattr(model.model, "model_sampling", None)
    is_flux_sampling = isinstance(model_sampling_obj, (ModelSamplingFlux, ModelSamplingFlux2))
    if flux or flux2 or is_flux_sampling:
        enable_multiscale = False

    latent_image = Latent.fix_empty_latent_channels(model, latent["samples"])
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        noise = ksampler_util.prepare_noise(latent_image, seed, latent.get("batch_index"), seeds_per_sample=latent.get("seeds"))
    samples = sample1(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                      denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                      force_full_denoise=force_full_denoise, noise_mask=latent.get("noise_mask"), seed=seed,
                      pipeline=pipeline, flux=flux, flux2=flux2, enable_multiscale=enable_multiscale, multiscale_factor=multiscale_factor,
                      multiscale_fullres_start=multiscale_fullres_start, multiscale_fullres_end=multiscale_fullres_end,
                      multiscale_intermittent_fullres=multiscale_intermittent_fullres, cfg_free_enabled=cfg_free_enabled,
                      cfg_free_start_percent=cfg_free_start_percent, batched_cfg=batched_cfg,
                      dynamic_cfg_rescaling=dynamic_cfg_rescaling, dynamic_cfg_method=dynamic_cfg_method,
                      dynamic_cfg_percentile=dynamic_cfg_percentile, dynamic_cfg_target_scale=dynamic_cfg_target_scale,
                      adaptive_noise_enabled=adaptive_noise_enabled, adaptive_noise_method=adaptive_noise_method,
                      model_options=model_options)
    out = latent.copy()
    out["samples"] = samples
    return (out,)
