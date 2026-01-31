"""Base sampler infrastructure for LightDiffusion-Next.

This module provides the foundational classes for all samplers:
- MultiscaleManager: Handles resolution switching during sampling
- SamplerCallback: Unified progress, preview, and interrupt handling
- BaseSampler: Abstract base class extracting common sampler logic

By extracting this shared logic, individual samplers become ~30 lines each
instead of 150+ lines with 90% duplication.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tqdm.auto import trange

from src.AutoEncoders import taesd
from src.sample import sampling_util
from src.user import app_instance
from src.Utilities import util


@dataclass
class MultiscaleConfig:
    """Configuration for multi-scale diffusion."""
    enabled: bool = True
    factor: float = 0.5
    fullres_start: int = 3
    fullres_end: int = 8
    intermittent_fullres: bool = False
    
    def validate(self) -> bool:
        """Validate config and return whether multi-scale should be active."""
        if not self.enabled:
            return False
        if not (0.1 <= self.factor <= 1.0):
            print(f"Warning: multiscale_factor {self.factor} out of range [0.1, 1.0]")
            return False
        if self.fullres_start < 0 or self.fullres_end < 0:
            print("Warning: Invalid fullres step counts")
            return False
        return True


class MultiscaleManager:
    """Manages resolution switching during sampling.
    
    Extracts the duplicated multi-scale logic from all samplers into
    a single reusable component.
    """
    
    def __init__(
        self,
        original_shape: tuple[int, int, int, int],
        n_steps: int,
        config: MultiscaleConfig,
    ):
        self.original_shape = original_shape
        self.batch_size, self.channels, self.orig_h, self.orig_w = original_shape
        self.n_steps = n_steps
        self.config = config
        
        # Calculate scaled dimensions (must be multiples of 8 for VAE)
        self.active = config.validate()
        if self.active:
            self.scale_h = int(max(8, ((self.orig_h * config.factor) // 8) * 8))
            self.scale_w = int(max(8, ((self.orig_w * config.factor) // 8) * 8))
            self.active = (self.scale_h != self.orig_h or self.scale_w != self.orig_w)
        else:
            self.scale_h = self.orig_h
            self.scale_w = self.orig_w
        
        if self.active:
            print(f"Multi-scale: {self.orig_h}x{self.orig_w} -> {self.scale_h}x{self.scale_w}")
        
        # Pre-calculate resolution schedule
        self._schedule = [self._should_use_fullres(i) for i in range(n_steps)]
    
    def _should_use_fullres(self, step: int) -> bool:
        """Determine if step should use full resolution."""
        if not self.active:
            return True
        
        # Always full res for start and end
        if step < self.config.fullres_start:
            return True
        if step >= self.n_steps - self.config.fullres_end:
            return True
        
        # Intermittent: every 2nd step in low-res region
        if self.config.intermittent_fullres:
            low_start = self.config.fullres_start
            low_end = self.n_steps - self.config.fullres_end
            if low_start <= step < low_end:
                return (step - low_start) % 2 == 0
        
        return False
    
    def use_fullres(self, step: int) -> bool:
        """Check if step should use full resolution."""
        return self._schedule[step] if step < len(self._schedule) else True
    
    def downscale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Downscale tensor to low-res."""
        if not self.active or tensor.shape[-2:] == (self.scale_h, self.scale_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(self.scale_h, self.scale_w), 
            mode="bilinear", align_corners=False
        )
    
    def upscale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Upscale tensor to full-res."""
        if not self.active or tensor.shape[-2:] == (self.orig_h, self.orig_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(self.orig_h, self.orig_w),
            mode="bilinear", align_corners=False
        )
    
    def process_at_resolution(
        self,
        x: torch.Tensor,
        step: int,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Process x at appropriate resolution for this step.
        
        Args:
            x: Input tensor at full resolution
            step: Current step index
            model_fn: Function to call with (possibly downscaled) tensor
            
        Returns:
            Result tensor at full resolution
        """
        if self.use_fullres(step):
            return model_fn(x)
        else:
            x_low = self.downscale(x)
            result_low = model_fn(x_low)
            return self.upscale(result_low)


class SamplerCallback:
    """Unified callback handling for samplers.
    
    Handles progress, interruption, and preview updates.
    """
    
    def __init__(self, n_steps: int, pipeline_mode: bool = False, preview_interval: int = 5):
        self.n_steps = n_steps
        self.pipeline_mode = pipeline_mode
        self.preview_interval = preview_interval
    
    def check_interrupt(self) -> bool:
        """Check for user interrupt. Returns True if interrupted."""
        app = getattr(app_instance, "app", None)
        if app is not None and getattr(app, "interrupt_flag", False):
            return True
        return False
    
    def update_progress(self, step: int) -> None:
        """Update progress bar (only in non-pipeline mode)."""
        if not self.pipeline_mode:
            app = getattr(app_instance, "app", None)
            if app is not None:
                app.progress.set(step / self.n_steps)
    
    def show_preview(self, x: torch.Tensor, step: int, force: bool = False) -> None:
        """Show TAESD preview if enabled and at interval."""
        app = getattr(app_instance, "app", None)
        if app is None:
            return
        
        if force or (app.previewer_var.get() and step % self.preview_interval == 0):
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()
    
    def step_callback(
        self,
        user_callback: Optional[Callable],
        step: int,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoised: torch.Tensor,
        **extra,
    ) -> bool:
        """Run all callbacks for a step.
        
        Returns True if generation should continue, False if interrupted.
        """
        if self.check_interrupt():
            return False
        
        self.update_progress(step)
        
        if user_callback is not None:
            user_callback({
                "x": x,
                "i": step,
                "sigma": sigma,
                "denoised": denoised,
                **extra,
            })
        
        self.show_preview(x, step)
        return True


def set_model_options_post_cfg_function(
    model_options: dict,
    post_cfg_function: Callable,
    disable_cfg1_optimization: bool = False,
) -> dict:
    """Add post-CFG function to model options."""
    model_options = model_options.copy()
    model_options["sampler_post_cfg_function"] = model_options.get(
        "sampler_post_cfg_function", []
    ) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


@dataclass
class CFGPPState:
    """State for CFG++ momentum tracking."""
    old_denoised: Optional[torch.Tensor] = None
    old_uncond_denoised: Optional[torch.Tensor] = None
    
    def capture_uncond(self, args: dict) -> torch.Tensor:
        """Post-CFG function to capture unconditional prediction."""
        self.old_uncond_denoised = args.get("uncond_denoised")
        return args["denoised"]
    
    def update(self, denoised: torch.Tensor, uncond_denoised: torch.Tensor) -> None:
        """Update state after step."""
        self.old_denoised = denoised
        self.old_uncond_denoised = uncond_denoised


class BaseSampler(ABC):
    """Abstract base class for all samplers.
    
    Extracts ~90% of duplicated code from individual samplers.
    Subclasses only need to implement the core step logic.
    """
    
    def __init__(
        self,
        # Multi-scale params
        enable_multiscale: bool = True,
        multiscale_factor: float = 0.5,
        multiscale_fullres_start: int = 3,
        multiscale_fullres_end: int = 8,
        multiscale_intermittent_fullres: bool = False,
        # CFG++ params
        cfg_scale: float = 7.5,
        cfg_min: float = 1.0,
        cfg_x0_scale: float = 1.0,
        # Mode
        pipeline: bool = False,
    ):
        self.multiscale_config = MultiscaleConfig(
            enabled=enable_multiscale,
            factor=multiscale_factor,
            fullres_start=multiscale_fullres_start,
            fullres_end=multiscale_fullres_end,
            intermittent_fullres=multiscale_intermittent_fullres,
        )
        self.cfg_scale = cfg_scale
        self.cfg_min = cfg_min
        self.cfg_x0_scale = cfg_x0_scale
        self.pipeline = pipeline
    
    def get_cfg_at_step(self, step: int, n_steps: int) -> float:
        """Get CFG scale for a given step (linear schedule)."""
        progress = step / max(1, n_steps - 1)
        return self.cfg_scale + (self.cfg_min - self.cfg_scale) * progress
    
    def setup_cfgpp(self, extra_args: dict) -> tuple[dict, CFGPPState]:
        """Setup CFG++ state and model options.
        
        Returns:
            Tuple of (modified extra_args, CFGPPState)
        """
        state = CFGPPState()
        model_options = extra_args.get("model_options", {}).copy()
        extra_args = extra_args.copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(
            model_options, state.capture_uncond, disable_cfg1_optimization=True
        )
        return extra_args, state
    
    def apply_cfg_denoised(
        self,
        denoised: torch.Tensor,
        uncond_denoised: torch.Tensor,
        current_cfg: float,
        state: CFGPPState,
        h_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply CFG/CFG++ to get final denoised prediction.
        
        Args:
            denoised: Conditional denoised prediction
            uncond_denoised: Unconditional denoised prediction  
            current_cfg: CFG scale for this step
            state: CFG++ state with previous predictions
            h_ratio: Momentum ratio (None for no momentum/first step)
            
        Returns:
            CFG-combined denoised prediction
        """
        if state.old_uncond_denoised is None or h_ratio is None:
            # First step or no momentum - regular CFG
            return torch.lerp(uncond_denoised, denoised, current_cfg)
        
        # CFG++ with momentum
        h_plus_1 = 1 + h_ratio
        momentum = h_plus_1 * denoised - h_ratio * state.old_denoised
        uncond_momentum = h_plus_1 * uncond_denoised - h_ratio * state.old_uncond_denoised
        
        return torch.lerp(uncond_momentum, momentum, current_cfg * self.cfg_x0_scale)
    
    @torch.no_grad()
    def sample(
        self,
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: Optional[dict] = None,
        callback: Optional[Callable] = None,
        disable: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the sampling loop.
        
        This method handles all the common setup and delegates to
        _sample_loop for the algorithm-specific logic.
        """
        extra_args = extra_args or {}
        n_steps = len(sigmas) - 1
        
        if n_steps <= 0:
            return x
        
        # Setup managers
        device = x.device
        multiscale = MultiscaleManager(x.shape, n_steps, self.multiscale_config)
        cb = SamplerCallback(n_steps, self.pipeline)
        s_in = torch.ones((x.shape[0],), device=device)
        
        # Setup CFG++
        extra_args, cfg_state = self.setup_cfgpp(extra_args)
        
        # Delegate to subclass
        return self._sample_loop(
            model=model,
            x=x,
            sigmas=sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            n_steps=n_steps,
            device=device,
            multiscale=multiscale,
            cb=cb,
            s_in=s_in,
            cfg_state=cfg_state,
            **kwargs,
        )
    
    @abstractmethod
    def _sample_loop(
        self,
        model: Any,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        extra_args: dict,
        callback: Optional[Callable],
        disable: Optional[bool],
        n_steps: int,
        device: torch.device,
        multiscale: MultiscaleManager,
        cb: SamplerCallback,
        s_in: torch.Tensor,
        cfg_state: CFGPPState,
        **kwargs,
    ) -> torch.Tensor:
        """Algorithm-specific sampling loop.
        
        Subclasses implement only the unique step logic here.
        """
        pass


class EulerSampler(BaseSampler):
    """Euler sampler with multi-scale and CFG++ support."""
    
    def _sample_loop(
        self,
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        n_steps,
        device,
        multiscale,
        cb,
        s_in,
        cfg_state,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        **kwargs,
    ):
        gamma_max = min(s_churn / n_steps, 2**0.5 - 1) if s_churn > 0 else 0
        
        for i in trange(n_steps, disable=disable):
            if not cb.step_callback(None, i, x, sigmas[i], x):
                return x
            
            current_cfg = self.get_cfg_at_step(i, n_steps)
            use_fullres = multiscale.use_fullres(i)
            
            # Sigma noise injection
            sigma_hat = sigmas[i]
            if gamma_max > 0 and s_tmin <= sigmas[i] <= s_tmax:
                sigma_hat = sigmas[i] * (1 + gamma_max)
                x = x + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i]**2)**0.5
            
            # Model inference at appropriate resolution
            if use_fullres:
                denoised = model(x, sigma_hat * s_in, **extra_args)
            else:
                x_low = multiscale.downscale(x)
                s_low = torch.ones((x_low.shape[0],), device=device)
                denoised = multiscale.upscale(model(x_low, sigma_hat * s_low, **extra_args))
            
            # Get unconditional from post-cfg capture
            uncond_denoised = cfg_state.old_uncond_denoised
            if uncond_denoised is None:
                uncond_denoised = denoised  # Fallback
            
            # Apply CFG
            cfg_denoised = self.apply_cfg_denoised(
                denoised, uncond_denoised, current_cfg, cfg_state
            )
            cfg_state.update(denoised, uncond_denoised)
            
            # Euler step
            d = util.to_d(x, sigma_hat, cfg_denoised)
            x = x + d * (sigmas[i + 1] - sigma_hat)
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.show_preview(x, i)
        
        return x


class EulerAncestralSampler(BaseSampler):
    """Euler ancestral sampler with multi-scale and CFG++ support."""
    
    def _sample_loop(
        self,
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        n_steps,
        device,
        multiscale,
        cb,
        s_in,
        cfg_state,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
        **kwargs,
    ):
        noise_sampler = noise_sampler or sampling_util.default_noise_sampler(x)
        
        for i in trange(n_steps, disable=disable):
            if not cb.step_callback(None, i, x, sigmas[i], x):
                return x
            
            current_cfg = self.get_cfg_at_step(i, n_steps)
            use_fullres = multiscale.use_fullres(i)
            
            # Model inference
            if use_fullres:
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                x_low = multiscale.downscale(x)
                s_low = torch.ones((x_low.shape[0],), device=device)
                denoised = multiscale.upscale(model(x_low, sigmas[i] * s_low, **extra_args))
            
            uncond_denoised = cfg_state.old_uncond_denoised or denoised
            cfg_denoised = self.apply_cfg_denoised(
                denoised, uncond_denoised, current_cfg, cfg_state
            )
            cfg_state.update(denoised, uncond_denoised)
            
            # Ancestral step
            sigma_down, sigma_up = sampling_util.get_ancestral_step(
                sigmas[i], sigmas[i + 1], eta=eta
            )
            d = util.to_d(x, sigmas[i], cfg_denoised)
            x = x + d * (sigma_down - sigmas[i])
            
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.show_preview(x, i)
        
        return x


class DPMPP2MSampler(BaseSampler):
    """DPM++ 2M sampler with CFG++ optimizations."""
    
    def _sample_loop(
        self,
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        n_steps,
        device,
        multiscale,
        cb,
        s_in,
        cfg_state,
        **kwargs,
    ):
        # Pre-calculate values
        t_steps = -torch.log(sigmas)
        sigma_steps = torch.exp(-t_steps)
        ratios = sigma_steps[1:] / sigma_steps[:-1]
        h_steps = t_steps[1:] - t_steps[:-1]
        
        for i in trange(n_steps, disable=disable):
            if not cb.step_callback(None, i, x, sigmas[i], x):
                return x
            
            current_cfg = self.get_cfg_at_step(i, n_steps)
            use_fullres = multiscale.use_fullres(i)
            
            # Model inference
            if use_fullres:
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                x_low = multiscale.downscale(x)
                s_low = torch.ones((x_low.shape[0],), device=device)
                denoised = multiscale.upscale(model(x_low, sigmas[i] * s_low, **extra_args))
            
            uncond_denoised = cfg_state.old_uncond_denoised or denoised
            
            # Calculate h_ratio for momentum
            h_ratio = None
            if i > 0 and cfg_state.old_denoised is not None:
                h_ratio = h_steps[i - 1] / (2 * h_steps[i])
            
            cfg_denoised = self.apply_cfg_denoised(
                denoised, uncond_denoised, current_cfg, cfg_state, h_ratio
            )
            cfg_state.update(denoised, uncond_denoised)
            
            # DPM++ 2M update
            h_expm1 = torch.expm1(-h_steps[i])
            x = ratios[i] * x - h_expm1 * cfg_denoised
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.show_preview(x, i)
        
        return x


class DPMPPSDESampler(BaseSampler):
    """DPM++ SDE sampler with CFG++ optimizations."""
    
    def _sample_loop(
        self,
        model,
        x,
        sigmas,
        extra_args,
        callback,
        disable,
        n_steps,
        device,
        multiscale,
        cb,
        s_in,
        cfg_state,
        eta=1.0,
        s_noise=1.0,
        noise_sampler=None,
        r=0.5,
        seed=None,
        **kwargs,
    ):
        # Helper functions
        def sigma_fn(t):
            return (-t).exp()
        
        def t_fn(sigma):
            return -sigma.log()
        
        # Initialize noise sampler
        if noise_sampler is None:
            sigmas_cpu = sigmas.cpu()
            noise_sampler = sampling_util.BrownianTreeNoiseSampler(
                x, sigmas_cpu[sigmas_cpu > 0].min(), sigmas_cpu.max(), seed=seed, cpu=True
            )
        
        for i in trange(n_steps, disable=disable):
            if not cb.step_callback(None, i, x, sigmas[i], x):
                return x
            
            current_cfg = self.get_cfg_at_step(i, n_steps)
            use_fullres = multiscale.use_fullres(i)
            
            # Model inference
            if use_fullres:
                denoised = model(x, sigmas[i] * s_in, **extra_args)
            else:
                x_low = multiscale.downscale(x)
                s_low = torch.ones((x_low.shape[0],), device=device)
                denoised = multiscale.upscale(model(x_low, sigmas[i] * s_low, **extra_args))
            
            uncond_denoised = cfg_state.old_uncond_denoised or denoised
            
            if sigmas[i + 1] == 0:
                # Final step - simple CFG
                cfg_denoised = self.apply_cfg_denoised(
                    denoised, uncond_denoised, current_cfg, cfg_state
                )
                x = x + util.to_d(x, sigmas[i], cfg_denoised) * (sigmas[i + 1] - sigmas[i])
            else:
                # Two-step update
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                s = t + (t_next - t) * r
                
                sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
                s_ = t_fn(sd)
                
                # h_ratio for momentum
                h_ratio = (t - s_) / (2 * (t - t_next)) if cfg_state.old_denoised is not None else None
                
                cfg_denoised = self.apply_cfg_denoised(
                    denoised, uncond_denoised, current_cfg, cfg_state, h_ratio
                )
                
                # Step 1
                noise1 = noise_sampler(sigma_fn(t), sigma_fn(s)).to(device) * s_noise * su
                x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * cfg_denoised + noise1
                
                # Step 2 inference
                if multiscale.use_fullres(i):
                    denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
                else:
                    x_2_low = multiscale.downscale(x_2)
                    s_low = torch.ones((x_2_low.shape[0],), device=device)
                    denoised_2 = multiscale.upscale(model(x_2_low, sigma_fn(s) * s_low, **extra_args))
                
                uncond_2 = cfg_state.old_uncond_denoised or denoised_2
                cfg_denoised_2 = self.apply_cfg_denoised(
                    denoised_2, uncond_2, current_cfg, cfg_state, h_ratio
                )
                
                # Final update
                sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
                t_next_ = t_fn(sd)
                
                noise_final = noise_sampler(sigma_fn(t), sigma_fn(t_next)).to(device) * s_noise * su
                x = (
                    (sigma_fn(t_next_) / sigma_fn(t)) * x
                    - (t - t_next_).expm1() * ((1 - 1/(2*r)) * cfg_denoised + (1/(2*r)) * cfg_denoised_2)
                    + noise_final
                )
            
            cfg_state.update(denoised, uncond_denoised)
            
            if callback:
                callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
            cb.show_preview(x, i)
        
        return x


# ============================================================================
# SAMPLER REGISTRY
# ============================================================================

SAMPLERS = {
    "euler": EulerSampler,
    "euler_ancestral": EulerAncestralSampler,
    "dpmpp_2m": DPMPP2MSampler,
    "dpmpp_2m_cfgpp": DPMPP2MSampler,
    "dpmpp_sde": DPMPPSDESampler,
    "dpmpp_sde_cfgpp": DPMPPSDESampler,
}


def get_sampler(name: str, **kwargs) -> BaseSampler:
    """Get a sampler instance by name.
    
    Args:
        name: Sampler name
        **kwargs: Sampler configuration
        
    Returns:
        Configured sampler instance
    """
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[name](**kwargs)
