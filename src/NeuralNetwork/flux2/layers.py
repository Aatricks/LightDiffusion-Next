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
    """Root Mean Square Layer Normalization.
    
    Uses native PyTorch rms_norm when available for numerical consistency with ComfyUI.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        # Use 'scale' to match Flux2 checkpoint naming convention
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
        # Check if native rms_norm is available
        self._use_native = hasattr(torch.nn.functional, 'rms_norm')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure scale is on the same device as input
        scale = self.scale.to(x.device, x.dtype)
        
        if self._use_native and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
            # Use native PyTorch rms_norm for better precision (matches ComfyUI)
            return torch.nn.functional.rms_norm(x, scale.shape, weight=scale, eps=self.eps)
        else:
            # Fallback implementation
            rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
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
    
    Matches ComfyUI's implementation exactly for numerical precision.
    
    Args:
        pos: Position indices
        dim: Embedding dimension
        theta: Base frequency
        
    Returns:
        Rotary embeddings as float32 concatenation of cos and sin
    """
    assert dim % 2 == 0
    device = pos.device
    
    # ComfyUI uses float64 for scale calculation for maximum precision
    scale = torch.linspace(0, (dim - 2) / dim, dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    
    # Einsum for position-frequency interaction - cast pos to float32 like ComfyUI
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    
    # ComfyUI always returns float32 for RoPE embeddings
    return out.to(dtype=torch.float32, device=pos.device)


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
        # Cast to v's dtype and device to match ComfyUI (crucial for numerical consistency)
        return q.to(v), k.to(v)


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


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Apply attention with rotary position embeddings.
    
    Args:
        q: Query tensor [batch, heads, seq, dim]
        k: Key tensor [batch, heads, seq, dim]  
        v: Value tensor [batch, heads, seq, dim]
        pe: Positional embeddings
        mask: Optional attention mask for padding tokens
        
    Returns:
        Attention output [batch, seq, heads*dim]
    """
    # Validate positional embedding sequence length to prevent RoPE shape errors
    if pe is not None:
        try:
            pe_seq = pe.shape[2] if pe.ndim >= 3 else None
            if pe_seq not in (1, q.shape[2]):
                raise ValueError(
                    f"RoPE sequence length mismatch: pe.seq={pe_seq} != q.seq={q.shape[2]}. "
                    "Transformer options (img_h/img_w) may not match the input token grid; check calc_cond_batch merging of transformer_options."
                )
        except Exception:
            # Re-raise as a clear ValueError for easier debugging
            raise

    q, k = apply_rope(q, k, pe)
    
    # Efficient attention implementation
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, mask=mask)
    return x


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a single tensor.
    
    Correctly applies the 2x2 rotation matrix:
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    
    Args:
        x: Input tensor [batch, heads, seq, dim]
        freqs_cis: Frequency tensor [batch, 1, seq, dim//2, 2, 2]
        
    Returns:
        Rotated tensor [batch, heads, seq, dim]
    """
    # Reshape x to match RoPE components [batch, heads, seq, dim//2, 2]
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

    # Handle differing sequence lengths between x and freqs_cis
    # freqs_cis shape: [batch, 1, seq_pe, dim//2, 2, 2]
    seq_x = x.shape[2]
    seq_pe = freqs_cis.shape[2]
    if seq_pe != seq_x:
        if seq_pe < seq_x:
            # Upsample by repeating along sequence dimension then slice to exact length
            repeat = (seq_x + seq_pe - 1) // seq_pe
            freqs_cis = freqs_cis.repeat_interleave(repeat, dim=2)[..., :seq_x, :, :, :]
        else:
            # Slice to match x sequence length
            freqs_cis = freqs_cis[..., :seq_x, :, :, :]

    # Extract rotation matrix components
    # freqs_cis is [..., dim//2, row, col]
    # row 0: [cos, -sin]
    # row 1: [sin, cos]
    cos = freqs_cis[..., 0, 0]
    msin = freqs_cis[..., 0, 1] # -sin
    sin = freqs_cis[..., 1, 0]

    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]

    # Apply rotation
    out1 = x1 * cos + x2 * msin
    out2 = x1 * sin + x2 * cos
    
    # Combine and reshape back to original
    return torch.stack([out1, out2], dim=-1).reshape(*x.shape).type_as(x)


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


def optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor = None) -> torch.Tensor:
    """Optimized attention using Flash/SDPA with fallback to xformers.
    
    Performance priority: cuDNN > Flash > SDPA > xformers > naive
    Uses SDPA backend priority from Device module for optimal dispatch.
    """
    b, _, seq_q, dim = q.shape
    _, _, seq_kv, _ = k.shape
    
    # Method 1: Use native scaled_dot_product_attention with backend priority
    # This is the fastest path on modern PyTorch with GPU support
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        try:
            # Get SDPA backend priority context manager from Device
            sdpa_context = Device.get_sdpa_context()
            
            # Process attention mask for SDPA if provided
            attn_mask = None
            if mask is not None:
                # Add dimensions as needed: [B, L] -> [B, 1, 1, L] for broadcasting
                if mask.ndim == 2:
                    attn_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
                elif mask.ndim == 3:
                    attn_mask = mask.unsqueeze(1)  # [B, 1, L, L]
                else:
                    attn_mask = mask
                # Convert mask to additive form (0 for attend, -inf for mask)
                # Input mask is 1 for valid, 0 for invalid (padding)
                attn_mask = attn_mask.to(dtype=q.dtype)
                attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
            
            # SDPA expects [batch, heads, seq, dim] - q/k/v are already in this format
            with sdpa_context:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            
            # Reshape: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
            # Use transpose + view for efficiency (avoid copy)
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
            # Note: xformers has different mask format, conversion would be needed
            out = xops.memory_efficient_attention(q_xf, k_xf, v_xf)
            del q_xf, k_xf, v_xf  # Free memory early
            # Reshape: [batch, seq, heads, dim] -> [batch, seq, heads*dim]
            out = out.reshape(b, seq_q, -1)
            return out
        except Exception:
            pass  # Fall through to naive
    
    # Method 3: Naive implementation (slowest, memory intensive)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(b, seq_q, -1)
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
        del img_normed  # Free memory early
        
        txt_normed = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_normed + txt_mod1.shift
        del txt_normed  # Free memory early

        # Run joint attention - use view+permute for efficiency instead of rearrange
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        q_img, k_img, v_img = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        q_txt, k_txt, v_txt = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv

        q_img, k_img = self.img_attn.norm(q_img, k_img, v_img)
        q_txt, k_txt = self.txt_attn.norm(q_txt, k_txt, v_txt)

        # Concatenate for joint attention
        q = torch.cat((q_txt, q_img), dim=2)
        del q_txt, q_img
        k = torch.cat((k_txt, k_img), dim=2)
        del k_txt, k_img
        v = torch.cat((v_txt, v_img), dim=2)
        del v_txt, v_img

        attn_out = attention(q, k, v, pe=pe, mask=attn_mask)
        del q, k, v
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]
        del attn_out

        # Apply residual connections with gating
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        del img_attn
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        del txt_attn

        # MLP with modulation
        img_mlp_in = (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        img = img + img_mod2.gate * self._forward_mlp(self.img_mlp, img_mlp_in)
        del img_mlp_in

        txt_mlp_in = (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        txt = txt + txt_mod2.gate * self._forward_mlp(self.txt_mlp, txt_mlp_in)
        del txt_mlp_in

        # Handle fp16 numerical issues (matches ComfyUI exactly)
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

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
        del x_normed  # Free memory early
        
        # Joint projection - split QKV from MLP part
        qkv_mlp = self.linear1(x_mod)
        del x_mod
        
        if self.gated_mlp:
            qkv, mlp_gate_up = qkv_mlp.split([self.hidden_size * 3, self.mlp_gate_up_dim], dim=-1)
            del qkv_mlp
            # Gated MLP: split into gate and up, apply SiLU to gate, multiply
            gate, up = mlp_gate_up.chunk(2, dim=-1)
            del mlp_gate_up
            mlp = F.silu(gate) * up
            del gate, up
        else:
            qkv, mlp = qkv_mlp.split([self.hidden_size * 3, self.mlp_hidden_dim], dim=-1)
            del qkv_mlp
            # Standard activation
            if self.silu_mlp:
                mlp = F.silu(mlp)
            else:
                mlp = F.gelu(mlp, approximate="tanh")
        
        # Attention - use view+permute for efficiency instead of rearrange
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k, v)
        
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
        del q, k, v
        
        # Combine and project
        output = self.linear2(torch.cat((attn, mlp), dim=-1))
        del attn, mlp
        
        result = x + mod.gate * output
        
        # Handle fp16 numerical issues (matches ComfyUI exactly)
        if result.dtype == torch.float16:
            result = torch.nan_to_num(result, nan=0.0, posinf=65504, neginf=-65504)
        
        return result


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
