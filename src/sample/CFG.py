"""Classifier-Free Guidance implementation."""
import math
import logging
import torch
from src.cond import cond, cond_util


def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    """Apply classifier-free guidance to predictions."""
    # Dynamic CFG rescaling
    if "cfg_guider" in model_options:
        guider = model_options["cfg_guider"]
        if hasattr(guider, "dynamic_cfg_rescaling") and guider.dynamic_cfg_rescaling:
            cond_scale = guider._apply_dynamic_cfg_rescaling(cond_pred, uncond_pred, cond_scale)
    
    # Custom sampler CFG
    if "sampler_cfg_function" in model_options:
        cfg_result = x - model_options["sampler_cfg_function"]({
            "cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale,
            "timestep": timestep, "input": x, "sigma": timestep,
            "cond_denoised": cond_pred, "uncond_denoised": uncond_pred,
            "model": model, "model_options": model_options,
        })
    elif math.isclose(cond_scale, 1.0):
        cfg_result = cond_pred
    else:
        cfg_result = torch.lerp(uncond_pred, cond_pred, cond_scale)

    # Post-CFG functions
    for fn in model_options.get("sampler_post_cfg_function", []):
        cfg_result = fn({
            "denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model,
            "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
            "sigma": timestep, "model_options": model_options, "input": x,
        })
    return cfg_result


def sampling_function(model, x, timestep, uncond, condo, cond_scale, model_options={}, seed=None):
    """Perform sampling with CFG."""
    uncond_ = None if (math.isclose(cond_scale, 1.0) and not model_options.get("disable_cfg1_optimization", False)) else uncond
    cond_outputs = cond.calc_cond_batch(model, [condo, uncond_], x, timestep, model_options)

    # Pre-CFG functions
    for fn in model_options.get("sampler_pre_cfg_function", []):
        cond_outputs = fn({
            "conds": [condo, uncond_], "conds_out": cond_outputs, "cond_scale": cond_scale,
            "timestep": timestep, "input": x, "sigma": timestep,
            "model": model, "model_options": model_options,
        })

    return cfg_function(model, cond_outputs[0], cond_outputs[1], cond_scale, x, timestep, model_options, condo, uncond_)


class CFGGuider:
    """Guidance with Classifier-Free Guidance."""

    def __init__(self, model_patcher, flux=False, dynamic_cfg_rescaling=False, dynamic_cfg_method="variance",
                 dynamic_cfg_percentile=95.0, dynamic_cfg_target_scale=7.0, 
                 adaptive_noise_enabled=False, adaptive_noise_method="complexity"):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0
        self.cfg_free_enabled = False
        self.cfg_free_start_percent = 70.0
        self.original_cfg = 1.0
        self.sigmas = None
        self.flux = flux  # Flag for FLUX model behavior
        self.dynamic_cfg_rescaling = dynamic_cfg_rescaling
        self.dynamic_cfg_method = dynamic_cfg_method
        self.dynamic_cfg_percentile = dynamic_cfg_percentile
        self.dynamic_cfg_target_scale = dynamic_cfg_target_scale
        self.adaptive_noise_enabled = adaptive_noise_enabled
        self.adaptive_noise_method = adaptive_noise_method
        self.complexity_history = []
        self.base_sigmas = None

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg
        self.original_cfg = cfg

    def set_cfg_free_params(self, enabled=False, start_percent=70.0):
        self.cfg_free_enabled = enabled
        self.cfg_free_start_percent = max(0.0, min(100.0, start_percent))
        if enabled:
            print(f"✓ CFG-Free sampling ACTIVE: CFG will reduce to 0 starting at {start_percent:.0f}% of steps")

    def _update_cfg_for_sigma(self, sigma):
        """Update CFG based on current sigma for CFG-free sampling."""
        if not self.cfg_free_enabled or self.sigmas is None or len(self.sigmas) <= 1:
            return

        total_steps = len(self.sigmas) - 1
        current_step = min(range(len(self.sigmas)), key=lambda i: abs(float(self.sigmas[i]) - float(sigma)))
        progress = (current_step / total_steps) * 100.0 if total_steps > 0 else 0

        if progress >= self.cfg_free_start_percent:
            remaining = 100.0 - self.cfg_free_start_percent
            if remaining > 0:
                self.cfg = max(0.0, self.original_cfg * (1.0 - (progress - self.cfg_free_start_percent) / remaining))
        else:
            self.cfg = self.original_cfg

    def _apply_dynamic_cfg_rescaling(self, cond_pred, uncond_pred, cond_scale):
        """Apply dynamic CFG rescaling."""
        if not self.dynamic_cfg_rescaling:
            return cond_scale
            
        diff = cond_pred - uncond_pred
        if self.dynamic_cfg_method == "variance":
            variance = min(torch.var(diff).item() / 0.1, 10.0)
            adjusted = cond_scale / (1.0 + variance * 0.1)
        elif self.dynamic_cfg_method == "range":
            low = torch.quantile(diff.flatten(), (100 - self.dynamic_cfg_percentile) / 100).item()
            high = torch.quantile(diff.flatten(), self.dynamic_cfg_percentile / 100).item()
            adjusted = min(cond_scale / max(high - low, 0.01), self.dynamic_cfg_target_scale)
        else:
            adjusted = cond_scale
        return max(1.0, min(adjusted, 20.0))

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = cond.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if self.cfg_free_enabled:
            self._update_cfg_for_sigma(timestep)
        
        opts = {**model_options, "cfg_guider": self}
        result = sampling_function(self.inner_model, x, timestep, 
                                   self.conds.get("negative"), self.conds.get("positive"),
                                   self.cfg, model_options=opts, seed=seed)
        
        if self.adaptive_noise_enabled:
            self.complexity_history.append(self._calc_complexity(result))
        return result

    def _calc_complexity(self, prediction):
        """Calculate complexity for adaptive noise."""
        if self.adaptive_noise_method == "complexity":
            dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
            dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
            h, w = min(dx.shape[2], dy.shape[2]), min(dx.shape[3], dy.shape[3])
            return (dx[:, :, :h, :w].abs() + dy[:, :, :h, :w].abs()).mean().item()
        return prediction.var(dim=[2, 3]).mean().item()

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, pipeline=False):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0:
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = cond.process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)
        self.sigmas = sigmas
        
        if self.adaptive_noise_enabled and len(self.complexity_history) > 0:
            if self.base_sigmas is None:
                self.base_sigmas = sigmas.clone()
            avg = sum(self.complexity_history) / len(self.complexity_history)
            sigmas = self.base_sigmas * (1.0 + (avg / max(0.01, avg + 0.1)) * 0.5)

        samples = sampler.sample(self, sigmas, {"model_options": self.model_options, "seed": seed},
                                 callback, noise, latent_image, denoise_mask, disable_pbar, pipeline=pipeline)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, 
               disable_pbar=False, seed=None, pipeline=False):
        self.conds = {k: [a.copy() for a in v] for k, v in self.original_conds.items()}
        self.inner_model, self.conds, self.loaded_models = cond_util.prepare_sampling(
            self.model_patcher, noise.shape, self.conds)
        device = self.model_patcher.load_device

        output = self.inner_sample(noise.to(device), latent_image.to(device), device, sampler,
                                   sigmas.to(device), denoise_mask, callback, disable_pbar, seed, pipeline)

        from src.Device.ModelCache import get_model_cache
        get_model_cache().prevent_model_cleanup(self.conds, self.loaded_models)
        del self.inner_model, self.conds, self.loaded_models
        return output
