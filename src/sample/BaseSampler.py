"""Simplified base sampler infrastructure for LightDiffusion-Next."""
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch
from tqdm.auto import trange
from src.AutoEncoders import taesd
from src.sample import sampling_util
from src.user import app_instance
from src.Utilities import util


@dataclass
class MultiscaleConfig:
    enabled: bool = True
    factor: float = 0.5
    fullres_start: int = 3
    fullres_end: int = 8
    intermittent_fullres: bool = False


class MultiscaleManager:
    """Handles resolution switching during sampling."""
    
    def __init__(self, shape: tuple, n_steps: int, config: MultiscaleConfig):
        self.orig_h, self.orig_w = shape[2], shape[3]
        self.n_steps = n_steps
        self.config = config
        
        # Calculate scaled dimensions (multiples of 8)
        self.active = config.enabled and 0.1 <= config.factor <= 1.0 and config.fullres_start >= 0 and config.fullres_end >= 0
        if self.active:
            self.scale_h = int(max(8, ((self.orig_h * config.factor) // 8) * 8))
            self.scale_w = int(max(8, ((self.orig_w * config.factor) // 8) * 8))
            self.active = self.scale_h != self.orig_h or self.scale_w != self.orig_w
        else:
            self.scale_h, self.scale_w = self.orig_h, self.orig_w
        
        if self.active:
            print(f"Multi-scale: {self.orig_h}x{self.orig_w} -> {self.scale_h}x{self.scale_w}")
        
        self._schedule = [self._should_fullres(i) for i in range(n_steps)]
    
    def _should_fullres(self, step: int) -> bool:
        if not self.active:
            return True
        if step < self.config.fullres_start or step >= self.n_steps - self.config.fullres_end:
            return True
        if self.config.intermittent_fullres:
            low_start = self.config.fullres_start
            if low_start <= step < self.n_steps - self.config.fullres_end:
                return (step - low_start) % 2 == 0
        return False
    
    def use_fullres(self, step: int) -> bool:
        return self._schedule[step] if step < len(self._schedule) else True
    
    def downscale(self, t: torch.Tensor) -> torch.Tensor:
        if not self.active or t.shape[-2:] == (self.scale_h, self.scale_w):
            return t
        return torch.nn.functional.interpolate(t, (self.scale_h, self.scale_w), mode="bilinear", align_corners=False)
    
    def upscale(self, t: torch.Tensor) -> torch.Tensor:
        if not self.active or t.shape[-2:] == (self.orig_h, self.orig_w):
            return t
        return torch.nn.functional.interpolate(t, (self.orig_h, self.orig_w), mode="bilinear", align_corners=False)


class SamplerCallback:
    """Handles progress, interruption, and preview."""
    
    def __init__(self, n_steps: int, pipeline: bool = False):
        self.n_steps = n_steps
        self.pipeline = pipeline
    
    def check_interrupt(self) -> bool:
        return getattr(getattr(app_instance, "app", None), "interrupt_flag", False)
    
    def update_progress(self, step: int):
        if not self.pipeline:
            app = getattr(app_instance, "app", None)
            if app:
                app.progress.set(step / self.n_steps)
    
    def preview(self, x: torch.Tensor, step: int):
        app = getattr(app_instance, "app", None)
        if app and app.previewer_var.get() and step % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()


def set_model_options_post_cfg_function(opts: dict, fn: Callable, disable_cfg1_optimization: bool = False) -> dict:
    opts = opts.copy()
    opts["sampler_post_cfg_function"] = opts.get("sampler_post_cfg_function", []) + [fn]
    if disable_cfg1_optimization:
        opts["disable_cfg1_optimization"] = True
    return opts


@dataclass
class CFGState:
    old_denoised: Optional[torch.Tensor] = None
    old_uncond: Optional[torch.Tensor] = None
    
    def capture(self, args: dict) -> torch.Tensor:
        self.old_uncond = args.get("uncond_denoised")
        return args["denoised"]
    
    def update(self, denoised: torch.Tensor, uncond: torch.Tensor):
        self.old_denoised = denoised
        self.old_uncond = uncond


class BaseSampler(ABC):
    """Abstract base for all samplers."""
    
    def __init__(self, enable_multiscale: bool = True, multiscale_factor: float = 0.5,
                 multiscale_fullres_start: int = 3, multiscale_fullres_end: int = 8,
                 multiscale_intermittent_fullres: bool = False, cfg_scale: float = 7.5,
                 cfg_min: float = 1.0, cfg_x0_scale: float = 1.0, pipeline: bool = False):
        self.ms_config = MultiscaleConfig(enable_multiscale, multiscale_factor,
                                          multiscale_fullres_start, multiscale_fullres_end,
                                          multiscale_intermittent_fullres)
        self.cfg_scale = cfg_scale
        self.cfg_min = cfg_min
        self.cfg_x0_scale = cfg_x0_scale
        self.pipeline = pipeline
    
    def get_cfg(self, step: int, n_steps: int) -> float:
        return self.cfg_scale + (self.cfg_min - self.cfg_scale) * (step / max(1, n_steps - 1))
    
    def apply_cfg(self, denoised: torch.Tensor, uncond: torch.Tensor, cfg: float,
                  state: CFGState, h_ratio: Optional[float] = None) -> torch.Tensor:
        if state.old_uncond is None or h_ratio is None:
            return torch.lerp(uncond, denoised, cfg)
        h1 = 1 + h_ratio
        momentum = h1 * denoised - h_ratio * state.old_denoised
        uncond_momentum = h1 * uncond - h_ratio * state.old_uncond
        return torch.lerp(uncond_momentum, momentum, cfg * self.cfg_x0_scale)
    
    @torch.no_grad()
    def sample(self, model: Any, x: torch.Tensor, sigmas: torch.Tensor,
               extra_args: Optional[dict] = None, callback: Optional[Callable] = None,
               disable: Optional[bool] = None, **kwargs) -> torch.Tensor:
        extra_args = extra_args or {}
        n_steps = len(sigmas) - 1
        if n_steps <= 0:
            return x
        
        device = x.device
        ms = MultiscaleManager(x.shape, n_steps, self.ms_config)
        cb = SamplerCallback(n_steps, self.pipeline)
        s_in = torch.ones((x.shape[0],), device=device)
        
        # Setup CFG++
        state = CFGState()
        extra_args = extra_args.copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(
            extra_args.get("model_options", {}), state.capture, disable_cfg1_optimization=True)
        
        return self._loop(model, x, sigmas, extra_args, callback, disable,
                          n_steps, device, ms, cb, s_in, state, **kwargs)
    
    @abstractmethod
    def _loop(self, model, x, sigmas, extra_args, callback, disable,
              n_steps, device, ms, cb, s_in, state, **kwargs) -> torch.Tensor:
        pass


class EulerSampler(BaseSampler):
    def _loop(self, model, x, sigmas, extra_args, callback, disable,
              n_steps, device, ms, cb, s_in, state, s_churn=0.0, s_tmin=0.0,
              s_tmax=float("inf"), s_noise=1.0, **kwargs):
        gamma_max = min(s_churn / n_steps, 2**0.5 - 1) if s_churn > 0 else 0
        
        for i in trange(n_steps, disable=disable):
            if cb.check_interrupt():
                return x
            cb.update_progress(i)
            
            sigma_hat = sigmas[i]
            if gamma_max > 0 and s_tmin <= sigmas[i] <= s_tmax:
                sigma_hat = sigmas[i] * (1 + gamma_max)
                x = x + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i]**2)**0.5
            
            if ms.use_fullres(i):
                denoised = model(x, sigma_hat * s_in, **extra_args)
            else:
                denoised = ms.upscale(model(ms.downscale(x), sigma_hat * torch.ones((ms.downscale(x).shape[0],), device=device), **extra_args))
            
            uncond = state.old_uncond if state.old_uncond is not None else denoised
            cfg_denoised = self.apply_cfg(denoised, uncond, self.get_cfg(i, n_steps), state)
            state.update(denoised, uncond)
            
            x = x + util.to_d(x, sigma_hat, cfg_denoised) * (sigmas[i + 1] - sigma_hat)
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.preview(x, i)
        return x


class EulerAncestralSampler(BaseSampler):
    def _loop(self, model, x, sigmas, extra_args, callback, disable,
              n_steps, device, ms, cb, s_in, state, eta=1.0, s_noise=1.0,
              noise_sampler=None, **kwargs):
        noise_sampler = noise_sampler or sampling_util.default_noise_sampler(x)
        
        for i in trange(n_steps, disable=disable):
            if cb.check_interrupt():
                return x
            cb.update_progress(i)
            
            if ms.use_fullres(i):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                denoised = ms.upscale(model(ms.downscale(x), sigmas[i] * torch.ones((ms.downscale(x).shape[0],), device=device), **extra_args))
            
            uncond = state.old_uncond if state.old_uncond is not None else denoised
            cfg_denoised = self.apply_cfg(denoised, uncond, self.get_cfg(i, n_steps), state)
            state.update(denoised, uncond)
            
            sigma_down, sigma_up = sampling_util.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            x = x + util.to_d(x, sigmas[i], cfg_denoised) * (sigma_down - sigmas[i])
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.preview(x, i)
        return x


class DPMPP2MSampler(BaseSampler):
    def _loop(self, model, x, sigmas, extra_args, callback, disable,
              n_steps, device, ms, cb, s_in, state, **kwargs):
        t_steps = -torch.log(sigmas)
        sigma_steps = torch.exp(-t_steps)
        ratios = sigma_steps[1:] / sigma_steps[:-1]
        h_steps = t_steps[1:] - t_steps[:-1]
        
        for i in trange(n_steps, disable=disable):
            if cb.check_interrupt():
                return x
            cb.update_progress(i)
            
            if ms.use_fullres(i):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                denoised = ms.upscale(model(ms.downscale(x), sigmas[i] * torch.ones((ms.downscale(x).shape[0],), device=device), **extra_args))
            
            uncond = state.old_uncond if state.old_uncond is not None else denoised
            h_ratio = h_steps[i - 1] / (2 * h_steps[i]) if i > 0 and state.old_denoised is not None else None
            cfg_denoised = self.apply_cfg(denoised, uncond, self.get_cfg(i, n_steps), state, h_ratio)
            state.update(denoised, uncond)
            
            x = ratios[i] * x - torch.expm1(-h_steps[i]) * cfg_denoised
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.preview(x, i)
        return x


class DPMPPSDESampler(BaseSampler):
    def _loop(self, model, x, sigmas, extra_args, callback, disable,
              n_steps, device, ms, cb, s_in, state, eta=1.0, s_noise=1.0,
              noise_sampler=None, r=0.5, seed=None, **kwargs):
        sigma_fn = lambda t: (-t).exp()
        t_fn = lambda s: -s.log()
        
        if noise_sampler is None:
            sigmas_cpu = sigmas.cpu()
            noise_sampler = sampling_util.BrownianTreeNoiseSampler(
                x, sigmas_cpu[sigmas_cpu > 0].min(), sigmas_cpu.max(), seed=seed, cpu=True)
        
        for i in trange(n_steps, disable=disable):
            if cb.check_interrupt():
                return x
            cb.update_progress(i)
            
            if ms.use_fullres(i):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                denoised = ms.upscale(model(ms.downscale(x), sigmas[i] * torch.ones((ms.downscale(x).shape[0],), device=device), **extra_args))
            
            uncond = state.old_uncond if state.old_uncond is not None else denoised
            cfg = self.get_cfg(i, n_steps)
            
            if sigmas[i + 1] == 0:
                cfg_denoised = self.apply_cfg(denoised, uncond, cfg, state)
                x = x + util.to_d(x, sigmas[i], cfg_denoised) * (sigmas[i + 1] - sigmas[i])
            else:
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                s = t + (t_next - t) * r
                sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
                s_ = t_fn(sd)
                
                h_ratio = (t - s_) / (2 * (t - t_next)) if state.old_denoised is not None else None
                cfg_denoised = self.apply_cfg(denoised, uncond, cfg, state, h_ratio)
                
                noise1 = noise_sampler(sigma_fn(t), sigma_fn(s)).to(device) * s_noise * su
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * cfg_denoised + noise1
                
                if ms.use_fullres(i):
                    denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
                else:
                    denoised_2 = ms.upscale(model(ms.downscale(x_2), sigma_fn(s) * torch.ones((ms.downscale(x_2).shape[0],), device=device), **extra_args))
                
                uncond_2 = state.old_uncond if state.old_uncond is not None else denoised_2
                cfg_denoised_2 = self.apply_cfg(denoised_2, uncond_2, cfg, state, h_ratio)
                
                sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
                t_next_ = t_fn(sd)
                noise_final = noise_sampler(sigma_fn(t), sigma_fn(t_next)).to(device) * s_noise * su
                x = ((sigma_fn(t_next_) / sigma_fn(t)) * x
                     - (t - t_next_).expm1() * ((1 - 1/(2*r)) * cfg_denoised + (1/(2*r)) * cfg_denoised_2)
                     + noise_final)
            
            state.update(denoised, uncond)
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.preview(x, i)
        return x


# Registry
SAMPLERS = {
    "euler": EulerSampler,
    "euler_ancestral": EulerAncestralSampler,
    "dpmpp_2m": DPMPP2MSampler,
    "dpmpp_2m_cfgpp": DPMPP2MSampler,
    "dpmpp_sde": DPMPPSDESampler,
    "dpmpp_sde_cfgpp": DPMPPSDESampler,
}


def get_sampler(name: str, **kwargs) -> BaseSampler:
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[name](**kwargs)
