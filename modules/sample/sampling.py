from enum import Enum
import threading
import torch.nn as nn

import math
import torch

from modules.Utilities import Latent
from modules.Device import Device
from modules.sample import ksampler_util, samplers, sampling_util
from modules.sample import CFG


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

    def calculate_denoised(
        self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor
    ) -> torch.Tensor:
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

    def noise_scaling(
        self,
        sigma: torch.Tensor,
        noise: torch.Tensor,
        latent_image: torch.Tensor,
        max_denoise: bool = False,
    ) -> torch.Tensor:
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
            sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(
        self, sigma: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """#### Inverse the noise scaling.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The inversely scaled noise tensor.
        """
        return latent


class CONST:
    def calculate_input(self, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """#### Calculate the input for CONST.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `noise` (torch.Tensor): The noise tensor.

        #### Returns:
            - `torch.Tensor`: The calculated input tensor.
        """
        return noise

    def calculate_denoised(
        self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor
    ) -> torch.Tensor:
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

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """#### Scale the noise.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `max_denoise` (bool, optional): Whether to apply maximum denoising. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The scaled noise tensor.
        """
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(
        self, sigma: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """#### Inverse the noise scaling.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The inversely scaled noise tensor.
        """
        return latent / (1.0 - sigma)


def flux_time_shift(mu: float, sigma: float, t) -> float:
    """#### Calculate the flux time shift.

    #### Args:
        - `mu` (float): The mu value.
        - `sigma` (float): The sigma value.
        - `t` (float): The t value.

    #### Returns:
        - `float`: The calculated flux time shift.
    """
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
        """#### Set the parameters for the model.

        #### Args:
            - `shift` (float, optional): The shift value. Defaults to 1.15.
            - `timesteps` (int, optional): The number of timesteps. Defaults to 10000.
        """
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_max(self):
        """#### Get the maximum sigma value."""
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """#### Convert sigma to timestep.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.

        #### Returns:
            - `torch.Tensor`: The timestep value.
        """
        return sigma

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """#### Convert timestep to sigma.

        #### Args:
            - `timestep` (torch.Tensor): The timestep value.

        #### Returns:
            - `torch.Tensor`: The sigma value.
        """
        return flux_time_shift(self.shift, 1.0, timestep)


class ModelSamplingDiscrete(torch.nn.Module):
    """#### Class for discrete model sampling."""

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

    def percent_to_sigma(self, percent: float) -> float:
        """#### Convert percent to sigma.

        #### Args:
            - `percent` (float): The percent value.

        #### Returns:
            - `float`: The sigma value.
        """
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

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        model_options: dict = {},
        seed: int = None,
    ) -> torch.Tensor:
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
    """#### Class for sampling."""

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

    def __init__(
        self,
        sampler_function: callable,
        extra_options: dict = {},
        inpaint_options: dict = {},
    ):
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
            - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

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


def ksampler(
    sampler_name: str,
    pipeline: bool = False,
    extra_options: dict = {},
    inpaint_options: dict = {},
) -> KSAMPLER:
    """#### Get a KSAMPLER.

    #### Args:
        - `sampler_name` (str): The sampler name.
        - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.
        - `extra_options` (dict, optional): The extra options. Defaults to {}.
        - `inpaint_options` (dict, optional): The inpaint options. Defaults to {}.

    #### Returns:
        - `KSAMPLER`: The KSAMPLER object.
    """
    if sampler_name == "dpmpp_2m_cfgpp":
        sampler_function = samplers.sample_dpmpp_2m_cfgpp

    elif sampler_name == "euler_ancestral_cfgpp":
        sampler_function = samplers.sample_euler_ancestral_dy_cfg_pp

    elif sampler_name == "dpmpp_sde_cfgpp":
        sampler_function = samplers.sample_dpmpp_sde_cfgpp

    elif sampler_name == "euler_cfgpp":
        sampler_function = samplers.sample_euler_dy_cfg_pp

    else:
        # Default fallback
        sampler_function = samplers.sample_euler
        print(f"Warning: Unknown sampler '{sampler_name}', falling back to euler")

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
        - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    cfg_guider = CFG.CFGGuider(model, flux=flux)
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
        - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

    #### Returns:
        - `KSAMPLER`: The KSAMPLER object.
    """
    sampler = ksampler(name, pipeline=pipeline)
    return sampler


class KSampler:
    """A unified sampler class that replaces both KSampler1 and KSampler2."""

    def __init__(
        self,
        model: torch.nn.Module = None,
        steps: int = None,
        sampler: str = None,
        scheduler: str = None,
        denoise: float = 1.0,
        model_options: dict = {},
        pipeline: bool = False,
    ):
        """Initialize the KSampler class.

        Args:
            model (torch.nn.Module, optional): The model to use for sampling. Required for direct sampling.
            steps (int, optional): The number of steps. Required for direct sampling.
            sampler (str, optional): The sampler name. Defaults to None.
            scheduler (str, optional): The scheduler name. Defaults to None.
            denoise (float, optional): The denoise factor. Defaults to 1.0.
            model_options (dict, optional): The model options. Defaults to {}.
            pipeline (bool, optional): Whether to use the pipeline. Defaults to False.
        """
        self.model = model
        self.device = model.load_device if model is not None else None
        self.scheduler = scheduler
        self.sampler_name = sampler
        self.denoise = denoise
        self.model_options = model_options
        self.pipeline = pipeline

        if model is not None and steps is not None:
            self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        """Calculate the sigmas for the given steps.

        Args:
            steps (int): The number of steps.

        Returns:
            torch.Tensor: The calculated sigmas.
        """
        sigmas = ksampler_util.calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )
        return sigmas

    def set_steps(self, steps: int, denoise: float = None):
        """Set the steps and calculate the sigmas.

        Args:
            steps (int): The number of steps.
            denoise (float, optional): The denoise factor. Defaults to None.
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

    def _process_sigmas(self, sigmas, start_step, last_step, force_full_denoise):
        """Process sigmas based on start_step and last_step.

        Args:
            sigmas (torch.Tensor): The sigmas tensor.
            start_step (int, optional): The start step. Defaults to None.
            last_step (int, optional): The last step. Defaults to None.
            force_full_denoise (bool): Whether to force full denoise.

        Returns:
            torch.Tensor: The processed sigmas.
        """
        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[: last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None and start_step < (len(sigmas) - 1):
            sigmas = sigmas[start_step:]

        return sigmas

    def direct_sample(
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
        flux: bool = False,
    ) -> torch.Tensor:
        """Sample directly with the initialized model and parameters.

        Args:
            noise (torch.Tensor): The noise tensor.
            positive (torch.Tensor): The positive tensor.
            negative (torch.Tensor): The negative tensor.
            cfg (float): The CFG value.
            latent_image (torch.Tensor, optional): The latent image tensor. Defaults to None.
            start_step (int, optional): The start step. Defaults to None.
            last_step (int, optional): The last step. Defaults to None.
            force_full_denoise (bool, optional): Whether to force full denoise. Defaults to False.
            denoise_mask (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            sigmas (torch.Tensor, optional): The sigmas tensor. Defaults to None.
            callback (callable, optional): The callback function. Defaults to None.
            disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
            seed (int, optional): The seed value. Defaults to None.
            flux (bool, optional): Whether to use flux mode. Defaults to False.

        Returns:
            torch.Tensor: The sampled tensor.
        """
        if self.model is None:
            raise ValueError("Model must be provided for direct sampling")

        if sigmas is None:
            sigmas = self.sigmas

        sigmas = self._process_sigmas(sigmas, start_step, last_step, force_full_denoise)

        # Early return if needed
        if start_step is not None and start_step >= (len(sigmas) - 1):
            if latent_image is not None:
                return latent_image
            else:
                return torch.zeros_like(noise)

        sampler_obj = sampler_object(self.sampler_name, pipeline=self.pipeline)

        return sample(
            self.model,
            noise,
            positive,
            negative,
            cfg,
            self.device,
            sampler_obj,
            sigmas,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
            pipeline=self.pipeline,
            flux=flux,
        )

    def sample(
        self,
        model: torch.nn.Module = None,
        seed: int = None,
        steps: int = None,
        cfg: float = None,
        sampler_name: str = None,
        scheduler: str = None,
        positive: torch.Tensor = None,
        negative: torch.Tensor = None,
        latent_image: torch.Tensor = None,
        denoise: float = None,
        start_step: int = None,
        last_step: int = None,
        force_full_denoise: bool = False,
        noise_mask: torch.Tensor = None,
        callback: callable = None,
        disable_pbar: bool = False,
        disable_noise: bool = False,
        pipeline: bool = False,
        flux: bool = False,
    ) -> tuple:
        """Unified sampling interface that works both as direct sampling and through the common_ksampler.

        This method can be used in two ways:
        1. If model is provided, it will create a temporary sampler and use that
        2. If model is None, it will use the pre-initialized model and parameters

        Args:
            model (torch.nn.Module, optional): The model to use for sampling. If None, uses pre-initialized model.
            seed (int, optional): The seed value.
            steps (int, optional): The number of steps. If None, uses pre-initialized steps.
            cfg (float, optional): The CFG value.
            sampler_name (str, optional): The sampler name. If None, uses pre-initialized sampler.
            scheduler (str, optional): The scheduler name. If None, uses pre-initialized scheduler.
            positive (torch.Tensor, optional): The positive tensor.
            negative (torch.Tensor, optional): The negative tensor.
            latent_image (torch.Tensor, optional): The latent image tensor.
            denoise (float, optional): The denoise factor. If None, uses pre-initialized denoise.
            start_step (int, optional): The start step. Defaults to None.
            last_step (int, optional): The last step. Defaults to None.
            force_full_denoise (bool, optional): Whether to force full denoise. Defaults to False.
            noise_mask (torch.Tensor, optional): The noise mask tensor. Defaults to None.
            callback (callable, optional): The callback function. Defaults to None.
            disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
            disable_noise (bool, optional): Whether to disable noise. Defaults to False.
            pipeline (bool, optional): Whether to use the pipeline. Defaults to False.
            flux (bool, optional): Whether to use flux mode. Defaults to False.

        Returns:
            tuple: The output tuple containing either (latent_dict,) or the sampled tensor.
        """
        # Case 1: Use pre-initialized model for direct sampling
        if model is None:
            if latent_image is None:
                raise ValueError(
                    "latent_image must be provided when using pre-initialized model"
                )

            return (
                self.direct_sample(
                    None,  # noise will be generated in common_ksampler
                    positive,
                    negative,
                    cfg,
                    latent_image,
                    start_step,
                    last_step,
                    force_full_denoise,
                    noise_mask,
                    None,  # sigmas will use pre-calculated ones
                    callback,
                    disable_pbar,
                    seed,
                    flux,
                ),
            )

        # Case 2: Use common_ksampler approach with provided model
        else:
            # For backwards compatibility with KSampler2 usage pattern
            if isinstance(latent_image, dict):
                latent = latent_image
            else:
                latent = {"samples": latent_image}

            return common_ksampler(
                model,
                seed,
                steps,
                cfg,
                sampler_name or self.sampler_name,
                scheduler or self.scheduler,
                positive,
                negative,
                latent,
                denoise or self.denoise,
                disable_noise,
                start_step,
                last_step,
                force_full_denoise,
                pipeline or self.pipeline,
                flux,
            )


# Refactor sample1 to use KSampler directly
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
    """Sample using the given parameters with the unified KSampler.

    Args:
        model (torch.nn.Module): The model.
        noise (torch.Tensor): The noise tensor.
        steps (int): The number of steps.
        cfg (float): The CFG value.
        sampler_name (str): The sampler name.
        scheduler (str): The scheduler name.
        positive (torch.Tensor): The positive tensor.
        negative (torch.Tensor): The negative tensor.
        latent_image (torch.Tensor): The latent image tensor.
        denoise (float, optional): The denoise factor. Defaults to 1.0.
        disable_noise (bool, optional): Whether to disable noise. Defaults to False.
        start_step (int, optional): The start step. Defaults to None.
        last_step (int, optional): The last step. Defaults to None.
        force_full_denoise (bool, optional): Whether to force full denoise. Defaults to False.
        noise_mask (torch.Tensor, optional): The noise mask tensor. Defaults to None.
        sigmas (torch.Tensor, optional): The sigmas tensor. Defaults to None.
        callback (callable, optional): The callback function. Defaults to None.
        disable_pbar (bool, optional): Whether to disable the progress bar. Defaults to False.
        seed (int, optional): The seed value. Defaults to None.
        pipeline (bool, optional): Whether to use the pipeline. Defaults to False.
        flux (bool, optional): Whether to use flux mode. Defaults to False.

    Returns:
        torch.Tensor: The sampled tensor.
    """
    sampler = KSampler(
        model=model,
        steps=steps,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
        pipeline=pipeline,
    )

    samples = sampler.direct_sample(
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
        flux=flux,
    )
    samples = samples.to(Device.intermediate_device())
    return samples


class ModelType(Enum):
    """#### Enum for Model Types."""

    EPS = 1
    FLUX = 8


def model_sampling(
    model_config: dict, model_type: ModelType, flux: bool = False
) -> torch.nn.Module:
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
        - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

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
    samples = samples.to(Device.intermediate_device())
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
    """Common ksampler function.

    Args:
        model (torch.nn.Module): The model.
        seed (int): The seed value.
        steps (int): The number of steps.
        cfg (float): The CFG value.
        sampler_name (str): The sampler name.
        scheduler (str): The scheduler name.
        positive (torch.Tensor): The positive tensor.
        negative (torch.Tensor): The negative tensor.
        latent (dict): The latent dictionary.
        denoise (float, optional): The denoise factor. Defaults to 1.0.
        disable_noise (bool, optional): Whether to disable noise. Defaults to False.
        start_step (int, optional): The start step. Defaults to None.
        last_step (int, optional): The last step. Defaults to None.
        force_full_denoise (bool, optional): Whether to force full denoise. Defaults to False.
        pipeline (bool, optional): Whether to use the pipeline. Defaults to False.
        flux (bool, optional): Whether to use flux mode. Defaults to False.

    Returns:
        tuple: The output tuple containing the latent dictionary and samples.
    """
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
        flux=flux,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)
