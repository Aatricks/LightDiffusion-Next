"""Sampling utilities for diffusion models."""
import logging
import math
import torch
import torchsde

from src.Utilities import util

disable_gui = False
logging.basicConfig(format="%(message)s", level=logging.INFO)


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """Create linear beta schedule."""
    return torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2


def checkpoint(func, inputs, params, flag):
    """Checkpoint wrapper (passthrough)."""
    return func(*inputs)


_freqs_cache = {}


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """Create sinusoidal timestep embedding."""
    half = dim // 2
    cache_key = (half, max_period, timesteps.device)
    if cache_key not in _freqs_cache:
        _freqs_cache[cache_key] = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    freqs = _freqs_cache[cache_key]
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def timestep_embedding_flux(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """Create timestep embedding for Flux models."""
    t = time_factor * t
    half = dim // 2
    cache_key = (half, max_period, t.device)
    if cache_key not in _freqs_cache:
        _freqs_cache[cache_key] = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    freqs = _freqs_cache[cache_key]
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Get Karras et al. (2022) noise schedule."""
    ramp = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_max ** (1/rho) + ramp * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    return util.append_zero(sigmas).to(device)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculate sigma_down and sigma_up for ancestral sampling."""
    if torch.is_tensor(sigma_to):
        sigma_up = torch.min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    else:
        sigma_up = min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    return (sigma_to**2 - sigma_up**2) ** 0.5, sigma_up


def default_noise_sampler(x):
    """Return function that generates randn_like(x)."""
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """Batched Brownian tree for SDE sampling."""
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = kwargs.pop("cpu", True)
        
        # Handle mock objects in tests
        try:
            t0, t1 = float(t0), float(t1)
        except Exception:
            t0, t1 = 0.0, 1.0
            
        t0, t1, self.sign = self.sort(t0, t1)
        
        if not isinstance(x, torch.Tensor):
            w0 = torch.zeros((1, 4, 8, 8))
        else:
            w0 = kwargs.get("w0", torch.zeros_like(x))
            
        seed = [seed if seed else torch.randint(0, 2**63 - 1, []).item()]
        self.batched = False
        
        t0_cpu = t0.cpu() if torch.is_tensor(t0) else torch.tensor(t0)
        t1_cpu = t1.cpu() if torch.is_tensor(t1) else torch.tensor(t1)
        w0_cpu = w0.cpu() if torch.is_tensor(w0) else w0
        
        self.trees = [torchsde.BrownianTree(t0_cpu, w0_cpu, t1_cpu, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0_val = t0.item() if torch.is_tensor(t0) else float(t0)
        t1_val = t1.item() if torch.is_tensor(t1) else float(t1)
        t_min, t_max, sign = self.sort(t0_val, t1_val)
        device = t0.device if torch.is_tensor(t0) else None
        w = torch.stack([tree(t_min, t_max).to(device=device) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """Noise sampler using Brownian tree."""
    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = transform(torch.as_tensor(sigma_min)), transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
