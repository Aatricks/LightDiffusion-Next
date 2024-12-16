import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from modules import util
from modules.Attention import AttentionMethods
from modules.Device import Device
from modules.cond import cast, cond

def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )



if Device.xformers_enabled():
    logging.info("Using xformers cross attention")
    optimized_attention = AttentionMethods.attention_xformers
else:
    logging.info("Using pytorch cross attention")
    optimized_attention = AttentionMethods.attention_pytorch

optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
    return AttentionMethods.attention_pytorch


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=cast.disable_weight_init,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = util.default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = util.default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = optimized_attention(q, k, v, self.heads)
        return self.to_out(out)
    


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = cast.disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = cast.disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = cast.disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = cast.disable_weight_init.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        if Device.xformers_enabled_vae():
            logging.info("Using xformers attention in VAE")
            self.optimized_attention = AttentionMethods.xformers_attention
        else:
            logging.info("Using pytorch attention in VAE")
            self.optimized_attention = AttentionMethods.pytorch_attention

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    return AttnBlock(in_channels)