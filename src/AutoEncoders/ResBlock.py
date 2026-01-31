"""Residual block components for diffusion models."""
from abc import abstractmethod
from typing import Optional, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.NeuralNetwork import transformer
from src.Attention import Attention
from src.cond import cast
from src.sample import sampling_util

ops = cast.disable_weight_init


class TimestepBlock1(nn.Module):
    """Abstract timestep block."""
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        pass


def forward_timestep_embed1(ts, x, emb, context=None, transformer_options={}, output_shape=None, 
                            time_context=None, num_video_frames=None, image_only_indicator=None):
    """Forward pass for timestep embedding."""
    for layer in ts:
        if isinstance(layer, TimestepBlock1):
            x = layer(x, emb)
        elif isinstance(layer, transformer.SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample1):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class Upsample1(nn.Module):
    """Upsample layer with optional conv."""
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, 
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = operations.conv_nd(dims, channels, self.out_channels, 3, 
                                           padding=padding, dtype=dtype, device=device)

    def forward(self, x, output_shape=None):
        shape = [x.shape[2] * 2, x.shape[3] * 2] if output_shape is None else [output_shape[2], output_shape[3]]
        x = F.interpolate(x, size=shape, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample1(nn.Module):
    """Downsample layer."""
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = operations.conv_nd(dims, channels, self.out_channels, 3, stride=stride, 
                                     padding=padding, dtype=dtype, device=device)

    def forward(self, x):
        return self.op(x)


class ResBlock1(TimestepBlock1):
    """Residual block with timestep embedding."""
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False,
                 kernel_size=3, exchange_temb_dims=False, skip_t_emb=False, 
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.skip_t_emb = skip_t_emb
        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device))
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels, 
                            dtype=dtype, device=device))
        
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device))
        
        self.skip_connection = (nn.Identity() if self.out_channels == channels 
                               else operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device))

    def forward(self, x, emb):
        return sampling_util.checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            h = h + emb_out
        return self.skip_connection(x) + self.out_layers(h)


class ResnetBlock(nn.Module):
    """VAE-style ResNet block."""
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels, self.out_channels = in_channels, out_channels

        self.norm1 = Attention.Normalize(in_channels)
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = Attention.Normalize(out_channels)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.nin_shortcut = ops.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x, temb):
        h = self.swish(self.norm1(x))
        h = self.conv1(h)
        h = self.dropout(self.swish(self.norm2(h)))
        h = self.conv2(h)
        return (self.nin_shortcut(x) if self.nin_shortcut else x) + h
