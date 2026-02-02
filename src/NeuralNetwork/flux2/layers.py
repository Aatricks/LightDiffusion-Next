"""Flux2 transformer layers for LightDiffusion-Next.

Core building blocks for the Flux2 architecture:
- Attention mechanisms
- Modulation layers  
- Transformer blocks (double and single stream)
- Embedding layers

Adapted from ComfyUI's Flux implementation for LightDiffusion-Next.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.cond import cast as ops_module
from src.Device import Device


# Get operations module
def get_ops():
    """Get the operations module for weight initialization."""
    return ops_module.disable_weight_init


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        # Use 'scale' to match Flux2 checkpoint naming convention
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS normalization
        # Ensure scale is on the same device as input
        scale = self.scale.to(x.device, x.dtype)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * scale


class EmbedND(nn.Module):
    """N-dimensional positional embedding using RoPE."""

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute rotary positional embeddings.
        
        Args:
            ids: Position IDs tensor of shape [batch, seq_len, num_axes]
            
        Returns:
            Rotary embeddings of shape [batch, seq_len, dim]
        """
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Compute rotary position embeddings.
    
    Args:
        pos: Position indices
        dim: Embedding dimension
        theta: Base frequency
        
    Returns:
        Rotary embeddings as concatenation of cos and sin
    """
    assert dim % 2 == 0
    if Device.xformers_enabled():
        device = pos.device
        dtype = torch.float32  # Compute in fp32 for precision
    else:
        device = pos.device
        dtype = torch.float64  # Higher precision fallback
        
    scale = torch.linspace(0, (dim - 2) / dim, dim // 2, dtype=dtype, device=device)
    omega = 1.0 / (theta ** scale)
    
    # Einsum for position-frequency interaction
    out = torch.einsum("...n,d->...nd", pos.to(dtype), omega)
    
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=pos.dtype)


class MLPEmbedder(nn.Module):
    """MLP for timestep and guidance embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None, ops_bias: bool = True):
        super().__init__()
        if operations is None:
            operations = get_ops()
        self.in_layer = operations.Linear(in_dim, hidden_dim, bias=ops_bias, dtype=dtype, device=device)
        self.out_layer = operations.Linear(hidden_dim, hidden_dim, bias=ops_bias, dtype=dtype, device=device)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU) for Klein models.
    
    Structure: hidden -> 2*intermediate -> SiLU gate -> intermediate -> hidden
    The first linear produces gate and value activations,
    SiLU is applied to gate, then gate * value, then final projection.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dtype=None, device=None, operations=None, ops_bias: bool = True):
        super().__init__()
        if operations is None:
            operations = get_ops()
        # First layer outputs 2x intermediate for gating
        self.gate_up_proj = operations.Linear(hidden_size, intermediate_size * 2, bias=ops_bias, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=ops_bias, dtype=dtype, device=device)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)


class QKNorm(nn.Module):
    """Query-Key normalization layer."""

    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        # Use native RMSNorm instead of operations.RMSNorm
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(nn.Module):
    """Self-attention with rotary position embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
        ops_bias: bool = True,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, bias=ops_bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Apply attention with rotary position embeddings.
    
    Args:
        q: Query tensor [batch, heads, seq, dim]
        k: Key tensor [batch, heads, seq, dim]  
        v: Value tensor [batch, heads, seq, dim]
        pe: Positional embeddings
        
    Returns:
        Attention output [batch, seq, heads*dim]
    """
    q, k = apply_rope(q, k, pe)
    
    # Efficient attention implementation
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads)
    return x


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a single tensor.
    
    Args:
        x: Input tensor [batch, heads, seq, dim]
        freqs_cis: Frequency tensor [..., dim//2, 2, 2]
        
    Returns:
        Rotated tensor [batch, heads, seq, dim]
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    
    # Apply rotation: out = freqs[..., 0] * x[..., 0] + freqs[..., 1] * x[..., 1]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]
    
    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(q: torch.Tensor, k: torch.Tensor, pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Query tensor [batch, heads, seq, dim]
        k: Key tensor [batch, heads, seq, dim]
        pe: Positional embeddings [..., dim//2, 2, 2]
        
    Returns:
        Rotated (q, k) tensors
    """
    return apply_rope1(q, pe), apply_rope1(k, pe)


def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """Optimized attention using Flash/SDPA with fallback to xformers.
    
    Performance priority: Flash > SDPA > xformers > naive
    """
    b, _, seq_q, dim = q.shape
    _, _, seq_kv, _ = k.shape
    
    # Method 1: Use native scaled_dot_product_attention (includes Flash attention when available)
    # This is the fastest path on modern PyTorch with GPU support
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        try:
            # SDPA expects [batch, heads, seq, dim] - q/k/v are already in this format
            # Just call directly - PyTorch auto-selects the best backend
            out = F.scaled_dot_product_attention(q, k, v)
            # Reshape: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
            out = out.transpose(1, 2).reshape(b, seq_q, -1)
            return out
        except Exception:
            pass  # Fall through to xformers
    
    # Method 2: Use xformers memory-efficient attention
    if Device.xformers_enabled():
        try:
            import xformers.ops as xops
            # xformers expects [batch, seq, heads, dim]
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            out = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
            # Reshape: [batch, seq, heads, dim] -> [batch, seq, heads*dim]
            out = out.reshape(b, seq_q, -1)
            return out
        except Exception:
            pass  # Fall through to naive
    
    # Method 3: Naive implementation (slowest, memory intensive)
    # Reshape for attention: [b, heads, seq, dim] -> [b, seq, heads, dim]
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    
    # Reshape back: [b, seq, heads, dim] -> [b, seq, heads*dim]
    out = out.reshape(b, seq_q, -1)
    return out


@dataclass
class ModulationOut:
    """Output of modulation layer."""
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """Adaptive layer normalization modulation.
    
    Applies shift, scale, and gate from conditioning vector.
    """

    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None, ops_bias: bool = True):
        super().__init__()
        if operations is None:
            operations = get_ops()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=ops_bias, dtype=dtype, device=device)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        
        mod1 = ModulationOut(shift=out[0], scale=out[1], gate=out[2])
        mod2 = ModulationOut(shift=out[3], scale=out[4], gate=out[5]) if self.is_double else None
        return mod1, mod2


class GlobalModulation(nn.Module):
    """Global modulation for Flux2 (Klein) double stream blocks."""

    def __init__(self, dim: int, dtype=None, device=None, operations=None, ops_bias: bool = True):
        super().__init__()
        if operations is None:
            operations = get_ops()
        # 12 outputs: 6 for img stream, 6 for txt stream
        self.lin = operations.Linear(dim, 12 * dim, bias=ops_bias, dtype=dtype, device=device)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut, ModulationOut, ModulationOut]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(12, dim=-1)
        
        mod1_img = ModulationOut(shift=out[0], scale=out[1], gate=out[2])
        mod2_img = ModulationOut(shift=out[3], scale=out[4], gate=out[5])
        mod1_txt = ModulationOut(shift=out[6], scale=out[7], gate=out[8])
        mod2_txt = ModulationOut(shift=out[9], scale=out[10], gate=out[11])
        
        return mod1_img, mod2_img, mod1_txt, mod2_txt


class DoubleStreamBlock(nn.Module):
    """Transformer block with separate image and text streams.
    
    Uses joint attention but separate MLPs for image and text.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        global_modulation: bool = False,
        dtype=None,
        device=None,
        operations=None,
        flax_compatible: bool = False,
        silu_mlp: bool = False,
        gated_mlp: bool = False,
        ops_bias: bool = True,  # Whether to use bias in linear layers
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.flax_compatible = flax_compatible
        self.silu_mlp = silu_mlp
        self.gated_mlp = gated_mlp
        
        # For gated MLP (Klein), mlp_ratio is the true ratio
        # First layer outputs 2x for gating: hidden -> 2*intermediate
        # Second layer: intermediate -> hidden
        if gated_mlp:
            mlp_intermediate = int(hidden_size * mlp_ratio)
            mlp_hidden_dim = mlp_intermediate * 2  # Double for gate+up projection
        else:
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            mlp_intermediate = mlp_hidden_dim

        if global_modulation:
            # When using global modulation at model level, don't create per-block modulation
            self.double_stream_modulation = None
            self.img_mod = None
            self.txt_mod = None
            self.use_global_modulation = True
        else:
            self.double_stream_modulation = None
            self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations, ops_bias=ops_bias)
            self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations, ops_bias=ops_bias)
            self.use_global_modulation = False

        # Image stream
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias, dtype=dtype, device=device, operations=operations, ops_bias=ops_bias)
        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        
        if gated_mlp:
            # Gated MLP with naming compatible with checkpoint: .0, .1 (identity), .2
            self.img_mlp = nn.Sequential(
                operations.Linear(hidden_size, mlp_hidden_dim, bias=ops_bias, dtype=dtype, device=device),
                nn.Identity(),  # Placeholder for index 1
                operations.Linear(mlp_intermediate, hidden_size, bias=ops_bias, dtype=dtype, device=device),
            )
        else:
            self.img_mlp = nn.Sequential(
                operations.Linear(hidden_size, mlp_hidden_dim, bias=ops_bias, dtype=dtype, device=device),
                nn.SiLU() if silu_mlp else nn.GELU(approximate="tanh"),
                operations.Linear(mlp_hidden_dim, hidden_size, bias=ops_bias, dtype=dtype, device=device),
            )

        # Text stream
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias, dtype=dtype, device=device, operations=operations, ops_bias=ops_bias)
        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        
        if gated_mlp:
            self.txt_mlp = nn.Sequential(
                operations.Linear(hidden_size, mlp_hidden_dim, bias=ops_bias, dtype=dtype, device=device),
                nn.Identity(),
                operations.Linear(mlp_intermediate, hidden_size, bias=ops_bias, dtype=dtype, device=device),
            )
        else:
            self.txt_mlp = nn.Sequential(
                operations.Linear(hidden_size, mlp_hidden_dim, bias=ops_bias, dtype=dtype, device=device),
                nn.SiLU() if silu_mlp else nn.GELU(approximate="tanh"),
                operations.Linear(mlp_hidden_dim, hidden_size, bias=ops_bias, dtype=dtype, device=device),
            )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        attn_mask=None,
        img_mod: tuple = None,  # (img_mod1, img_mod2) from global modulation
        txt_mod: tuple = None,  # (txt_mod1, txt_mod2) from global modulation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters
        if self.use_global_modulation and img_mod is not None and txt_mod is not None:
            # Use global modulation passed from model level
            img_mod1, img_mod2 = img_mod
            txt_mod1, txt_mod2 = txt_mod
        elif self.img_mod is not None and self.txt_mod is not None:
            # Use per-block modulation (Flux1 style)
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            raise ValueError("No modulation available - either provide global or use per-block modulation")

        # Prepare normed inputs
        img_normed = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_normed + img_mod1.shift
        
        txt_normed = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_normed + txt_mod1.shift

        # Run joint attention
        q_img, k_img, v_img = rearrange(
            self.img_attn.qkv(img_modulated), "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        q_txt, k_txt, v_txt = rearrange(
            self.txt_attn.qkv(txt_modulated), "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )

        q_img, k_img = self.img_attn.norm(q_img, k_img, v_img)
        q_txt, k_txt = self.txt_attn.norm(q_txt, k_txt, v_txt)

        # Concatenate for joint attention
        q = torch.cat((q_txt, q_img), dim=2)
        k = torch.cat((k_txt, k_img), dim=2)
        v = torch.cat((v_txt, v_img), dim=2)

        attn_out = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]

        # Apply residual connections with gating
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)

        # MLP with modulation
        img_mlp_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        img = img + img_mod2.gate * self._forward_mlp(self.img_mlp, img_mlp_in)

        txt_mlp_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        txt = txt + txt_mod2.gate * self._forward_mlp(self.txt_mlp, txt_mlp_in)

        return img, txt

    def _forward_mlp(self, mlp: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Forward through MLP, handling both standard and gated variants."""
        if self.gated_mlp:
            # Gated MLP: split into gate and up, apply SiLU to gate, multiply, project
            gate_up = mlp[0](x)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            return mlp[2](hidden)
        else:
            return mlp(x)


class SingleStreamBlock(nn.Module):
    """Transformer block with merged image and text stream.
    
    Used after the double stream blocks have processed both modalities.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None,
        silu_mlp: bool = False,
        gated_mlp: bool = False,
        ops_bias: bool = True,
        global_modulation: bool = False,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
            
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.silu_mlp = silu_mlp
        self.gated_mlp = gated_mlp
        self.use_global_modulation = global_modulation

        # For gated MLP, mlp_ratio gives intermediate size
        # linear1 outputs gate+up (2x intermediate), linear2 takes intermediate
        if gated_mlp:
            self.mlp_intermediate = int(hidden_size * mlp_ratio)
            self.mlp_gate_up_dim = self.mlp_intermediate * 2
            linear1_out = hidden_size * 3 + self.mlp_gate_up_dim
            linear2_in = hidden_size + self.mlp_intermediate
        else:
            self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
            linear1_out = hidden_size * 3 + self.mlp_hidden_dim
            linear2_in = hidden_size + self.mlp_hidden_dim
        
        # Joint QKV and MLP projection
        self.linear1 = operations.Linear(
            hidden_size, linear1_out, bias=ops_bias, dtype=dtype, device=device
        )
        self.linear2 = operations.Linear(
            linear2_in, hidden_size, bias=ops_bias, dtype=dtype, device=device
        )

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        # Only create per-block modulation if not using global modulation
        if not global_modulation:
            self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations, ops_bias=ops_bias)
        else:
            self.modulation = None

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        pe: torch.Tensor,
        attn_mask=None,
        modulation=None,  # ModulationOut from global modulation
    ) -> torch.Tensor:
        # Get modulation
        if self.use_global_modulation and modulation is not None:
            mod = modulation
        elif self.modulation is not None:
            mod, _ = self.modulation(vec)
        else:
            raise ValueError("No modulation available - either provide global or use per-block modulation")
        
        x_normed = self.pre_norm(x)
        x_mod = (1 + mod.scale) * x_normed + mod.shift
        
        # Joint projection - split QKV from MLP part
        qkv_mlp = self.linear1(x_mod)
        
        if self.gated_mlp:
            qkv, mlp_gate_up = qkv_mlp.split([self.hidden_size * 3, self.mlp_gate_up_dim], dim=-1)
            # Gated MLP: split into gate and up, apply SiLU to gate, multiply
            gate, up = mlp_gate_up.chunk(2, dim=-1)
            mlp = F.silu(gate) * up
        else:
            qkv, mlp = qkv_mlp.split([self.hidden_size * 3, self.mlp_hidden_dim], dim=-1)
            # Standard activation
            if self.silu_mlp:
                mlp = F.silu(mlp)
            else:
                mlp = F.gelu(mlp, approximate="tanh")
        
        # Attention
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        attn = attention(q, k, v, pe=pe)
        
        # Combine and project
        output = self.linear2(torch.cat((attn, mlp), dim=-1))
        
        return x + mod.gate * output


class LastLayer(nn.Module):
    """Final layer for unpatchifying and producing output."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=None, device=None, operations=None, ops_bias: bool = True):
        super().__init__()
        if operations is None:
            operations = get_ops()
            
        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=ops_bias, dtype=dtype, device=device
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 2 * hidden_size, bias=ops_bias, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
