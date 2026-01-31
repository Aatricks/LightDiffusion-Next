"""Transformer components for diffusion models."""
from einops import rearrange
import torch
import torch.nn as nn
from src.Utilities import util
from src.Attention import Attention
from src.Device import Device
from src.cond import Activation, cast
from src.sample import sampling_util

ops = cast.disable_weight_init


class FeedForward(nn.Module):
    """FeedForward network."""
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = Activation.GEGLU(dim, inner_dim) if glu else nn.Sequential(
            operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 operations.Linear(inner_dim, dim_out, dtype=dtype, device=device))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """Basic Transformer block with self/cross attention."""
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True,
                 checkpoint=True, ff_in=False, inner_dim=None, disable_self_attn=False,
                 disable_temporal_crossattention=False, switch_temporal_ca_to_sa=False,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.ff_in = ff_in or inner_dim is not None
        inner_dim = inner_dim or dim
        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn
        self.checkpoint = checkpoint
        self.n_heads, self.d_head = n_heads, d_head

        self.attn1 = Attention.CrossAttention(
            query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if disable_self_attn else None,
            dtype=dtype, device=device, operations=operations)
        
        self.attn2 = Attention.CrossAttention(
            query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=None if switch_temporal_ca_to_sa else context_dim,
            dtype=dtype, device=device, operations=operations)
        
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff,
                              dtype=dtype, device=device, operations=operations)
        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

    def forward(self, x, context=None, transformer_options={}):
        return sampling_util.checkpoint(self._forward, (x, context, transformer_options),
                                        self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, transformer_options={}):
        n = self.norm1(x)
        n = self.attn1(n, context=None, value=None)
        x = x + n
        
        if self.attn2:
            n = self.norm2(x)
            n = self.attn2(n, context=context, value=None)
            x = x + n
        
        x_skip = x if self.is_res else None
        x = self.ff(self.norm3(x))
        return x + x_skip if x_skip is not None else x


class SpatialTransformer(nn.Module):
    """Spatial Transformer module."""
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None,
                 disable_self_attn=False, use_linear=False, use_checkpoint=True,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        inner_dim = n_heads * d_head
        context_dim = [context_dim] * depth if context_dim and not isinstance(context_dim, list) else context_dim
        
        self.norm = operations.GroupNorm(32, in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        
        if use_linear:
            self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
            self.proj_out = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Conv2d(in_channels, inner_dim, 1, dtype=dtype, device=device)
            self.proj_out = operations.Conv2d(inner_dim, in_channels, 1, dtype=dtype, device=device)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim[d] if context_dim else None,
                                  disable_self_attn=disable_self_attn, checkpoint=use_checkpoint,
                                  dtype=dtype, device=device, operations=operations)
            for d in range(depth)])
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        context = [context] * len(self.transformer_blocks) if not isinstance(context, list) else context
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


def count_blocks(state_dict_keys, prefix_string):
    """Count blocks matching prefix."""
    count = 0
    while any(k.startswith(prefix_string.format(count)) for k in state_dict_keys):
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    """Calculate transformer depth from state dict."""
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = [k for k in state_dict_keys if k.startswith(transformer_prefix)]
    if not transformer_keys:
        return None
    
    depth = count_blocks(state_dict_keys, transformer_prefix + "{}")
    context_dim = state_dict[f"{transformer_prefix}0.attn2.to_k.weight"].shape[1]
    use_linear = len(state_dict[f"{prefix}1.proj_in.weight"].shape) == 2
    time_stack = (f"{prefix}1.time_stack.0.attn1.to_q.weight" in state_dict or
                  f"{prefix}1.time_mix_blocks.0.attn1.to_q.weight" in state_dict)
    return depth, context_dim, use_linear, time_stack
