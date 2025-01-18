import math
import torch
from modules.cond import cond, cond_util


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
    """#### Perform sampling with CFG.

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
    out = cond.calc_cond_batch(model, conds, x, timestep, model_options)

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
    """#### Class for guiding the sampling process with CFG."""
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
        """#### Set the conditions for CFG.

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
            self.original_conds[k] = cond.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        """#### Call the CFGGuider to predict noise.

        #### Returns:
            - `torch.Tensor`: The predicted noise.
        """
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """#### Predict noise using CFG.

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
            - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
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
        """#### Perform the sampling process with CFG.

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `sampler` (object): The sampler object.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            - `callback` (callable, optional): The callback function. Defaults to None.
            - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
            - `seed` (int, optional): The random seed. Defaults to None.
            - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = cond_util.prepare_sampling(
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

        cond_util.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output