"""
Color utility helpers for preview generation and simple tonemapping.

Functions:
- linear_to_srgb(tensor): Converts linear RGB in [0,1] to sRGB transfer (still in [0,1]).
- reinhard_tonemap(tensor): Applies a simple Reinhard tone map: x/(1+x).

These helpers are intentionally small and torch-centric so they can be applied
on CUDA tensors without forcing CPU/GIL roundtrips.
"""

from __future__ import annotations

import torch


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to sRGB transfer curve.

    Expects input values in the [0, 1] range (but will clamp). Returns values
    in [0, 1]. Operates elementwise and preserves device/dtype.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("linear_to_srgb expects a torch.Tensor")

    x = x.to(dtype=torch.float32)
    # clamp for numerical safety
    x = x.clamp(0.0, 1.0)
    a = 0.0031308
    # sRGB transfer: low linear segment and high gamma curve
    low = x <= a
    # Use torch.where to avoid advanced indexing performance issues
    return torch.where(low, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def reinhard_tonemap(x: torch.Tensor) -> torch.Tensor:
    """Simple Reinhard tonemapping: x / (1 + x).

    Useful when decoded values may exceed [0, 1]. This compresses highlights
    gracefully and keeps values in [0, 1) for positive inputs.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("reinhard_tonemap expects a torch.Tensor")

    return x / (1.0 + x)
