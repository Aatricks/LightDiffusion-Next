"""Attention implementations supporting PyTorch, XFormers, and SageAttention."""
try:
    import sageattention
except ImportError:
    sageattention = None

try:
    import spas_sage_attn
except ImportError:
    spas_sage_attn = None

try:
    import xformers
    BROKEN_XFORMERS = xformers.__version__.startswith("0.0.2") and not xformers.__version__.startswith("0.0.20")
except ImportError:
    xformers = None
    BROKEN_XFORMERS = False

import torch
import torch.nn.functional as F

# Pre-computed padding targets for SageAttention supported dimensions
# Maps dimension -> (target_dim, padding_amount) or None if no padding needed
_SAGE_PAD_CACHE: dict[int, tuple[int, int] | None] = {}


def _get_sage_padding(dim: int) -> tuple[int, int] | None:
    """Get pre-computed padding target for a given dimension.
    
    Returns (target_dim, pad_amount) or None if no padding needed.
    """
    if dim not in _SAGE_PAD_CACHE:
        if dim in (64, 96, 128):
            _SAGE_PAD_CACHE[dim] = None  # No padding needed
        elif dim < 64:
            _SAGE_PAD_CACHE[dim] = (64, 64 - dim)
        elif dim < 128:
            _SAGE_PAD_CACHE[dim] = (128, 128 - dim)
        else:
            _SAGE_PAD_CACHE[dim] = None  # Unsupported, no padding
    return _SAGE_PAD_CACHE[dim]


def _pad_for_sage(q, k, v, dim):
    """Pad tensors to supported SageAttention dimensions (64, 96, 128)."""
    padding = _get_sage_padding(dim)
    if padding is None:
        return q, k, v, dim
    target, pad = padding
    return (F.pad(q, (0, pad)), F.pad(k, (0, pad)), F.pad(v, (0, pad)), dim)


def _reshape_for_heads(q, k, v, heads, flux=False, skip_reshape=False):
    """Reshape tensors for multi-head attention."""
    if flux and skip_reshape:
        return q, k, v, q.shape[-1]
    b = q.shape[0]
    dim_head = q.shape[-1] // heads
    reshape_fn = lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2).contiguous()
    return reshape_fn(q), reshape_fn(k), reshape_fn(v), dim_head


def _reshape_output(out, b, heads, dim_head, flux=False, skip_reshape=False):
    """Reshape attention output back to original format."""
    if flux and not skip_reshape:
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    if not flux:
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def attention_pytorch(q, k, v, heads, mask=None, skip_reshape=False, flux=False):
    """Multi-head attention using PyTorch SDPA."""
    b = q.shape[0]
    if not flux:
        seq_q, seq_kv = q.shape[1], k.shape[1]
        dim_head = q.shape[-1] // heads
        q = q.view(b, seq_q, heads, dim_head).transpose(1, 2)
        k = k.view(b, seq_kv, heads, dim_head).transpose(1, 2)
        v = v.view(b, seq_kv, heads, dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).reshape(b, seq_q, heads * dim_head)
    
    dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if not skip_reshape:
        q, k, v = [t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def attention_xformers(q, k, v, heads, mask=None, skip_reshape=False, flux=False):
    """Multi-head attention using XFormers."""
    b = q.shape[0]
    if not flux:
        dim_head = q.shape[-1] // heads
        q, k, v = [t.view(b, -1, heads, dim_head).permute(0, 2, 1, 3).reshape(b * heads, -1, dim_head).contiguous()
                   for t in (q, k, v)]
        try:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        except (NotImplementedError, RuntimeError):
            out = F.scaled_dot_product_attention(
                q.view(b, heads, -1, dim_head), k.view(b, heads, -1, dim_head), v.view(b, heads, -1, dim_head),
                attn_mask=mask, dropout_p=0.0, is_causal=False).reshape(b * heads, -1, dim_head)
        return out.view(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    
    dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if BROKEN_XFORMERS and b * heads > 65535:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape, flux)
    
    if skip_reshape:
        q, k, v = [t.reshape(b * heads, -1, dim_head) for t in (q, k, v)]
    else:
        q, k, v = [t.reshape(b, -1, heads, dim_head) for t in (q, k, v)]
    
    try:
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
    except (NotImplementedError, RuntimeError):
        out = F.scaled_dot_product_attention(
            q.view(b, heads, -1, dim_head), k.view(b, heads, -1, dim_head), v.view(b, heads, -1, dim_head),
            attn_mask=mask, dropout_p=0.0, is_causal=False).reshape(b * heads, -1, dim_head)
    
    if skip_reshape:
        return out.view(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out.reshape(b, -1, heads * dim_head)


def attention_sage(q, k, v, heads, mask=None, skip_reshape=False, flux=False):
    """Multi-head attention using SageAttention."""
    if mask is not None and mask.device != q.device:
        mask = mask.to(q.device)
    
    b = q.shape[0]
    dim_head = q.shape[-1] if (flux and skip_reshape) else q.shape[-1] // heads
    
    if not (flux and skip_reshape):
        if not flux:
            q, k, v = [t.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                       for t in (q, k, v)]
        else:
            q, k, v = [t.reshape(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    
    # Pad and compute attention
    qp, kp, vp, orig_dim = _pad_for_sage(q, k, v, dim_head)
    if orig_dim != dim_head or orig_dim in [64, 96, 128]:
        out = sageattention.sageattn(qp, kp, vp, tensor_layout="HND", attn_mask=mask, is_causal=False)
        if orig_dim != dim_head:
            out = out[..., :orig_dim]
    elif dim_head > 128:
        # Fallback for unsupported dimensions
        try:
            out = xformers.ops.memory_efficient_attention(
                q.reshape(b * heads, -1, dim_head), k.reshape(b * heads, -1, dim_head), 
                v.reshape(b * heads, -1, dim_head), attn_bias=mask)
            out = out.reshape(b, heads, -1, dim_head)
        except:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    else:
        out = sageattention.sageattn(qp, kp, vp, tensor_layout="HND", attn_mask=mask, is_causal=False)
        out = out[..., :dim_head]
    
    if not flux:
        return out.reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def attention_sparge(q, k, v, heads, mask=None, skip_reshape=False, flux=False):
    """Multi-head attention using SpargeAttn (Sparse + SageAttention)."""
    b = q.shape[0]
    dim_head = q.shape[-1] if (flux and skip_reshape) else q.shape[-1] // heads
    
    if not (flux and skip_reshape):
        if not flux:
            q, k, v = [t.unsqueeze(3).reshape(b, -1, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                       for t in (q, k, v)]
        else:
            q, k, v = [t.reshape(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    
    qp, kp, vp, orig_dim = _pad_for_sage(q, k, v, dim_head)
    sparge_kwargs = dict(simthreshd1=0.6, cdfthreshd=0.97, pvthreshd=15, is_causal=False)
    
    if orig_dim != dim_head or orig_dim in [64, 96, 128]:
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(qp, kp, vp, **sparge_kwargs)
        if orig_dim != dim_head:
            out = out[..., :orig_dim]
    elif dim_head > 128:
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
    else:
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(qp, kp, vp, **sparge_kwargs)
        out = out[..., :dim_head]
    
    if not flux:
        return out.reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


# Simple 4D attention variants (B, C, H, W format)
def sage_attention(q, k, v):
    """SageAttention for 4D tensors (B, C, H, W)."""
    B, C, H, W = q.shape
    q, k, v = [t.view(B, 1, C, -1).transpose(2, 3).contiguous() for t in (q, k, v)]
    qp, kp, vp, orig = _pad_for_sage(q, k, v, C)
    if C > 128:
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    else:
        out = sageattention.sageattn(qp, kp, vp, tensor_layout="HND", is_causal=False)
        if orig != C:
            out = out[..., :C]
    return out.transpose(2, 3).reshape(B, C, H, W)


def sparge_attention(q, k, v):
    """SpargeAttn for 4D tensors (B, C, H, W)."""
    B, C, H, W = q.shape
    q, k, v = [t.view(B, 1, C, -1).transpose(2, 3).contiguous() for t in (q, k, v)]
    qp, kp, vp, orig = _pad_for_sage(q, k, v, C)
    sparge_kwargs = dict(simthreshd1=0.6, cdfthreshd=0.97, pvthreshd=15, is_causal=False)
    if C > 128:
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    else:
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(qp, kp, vp, **sparge_kwargs)
        if orig != C:
            out = out[..., :C]
    return out.transpose(2, 3).reshape(B, C, H, W)


def xformers_attention(q, k, v):
    """XFormers attention for 4D tensors (B, C, H, W)."""
    B, C, H, W = q.shape
    q, k, v = [t.view(B, C, -1).transpose(1, 2).contiguous() for t in (q, k, v)]
    try:
        out = xformers.ops.memory_efficient_attention(q, k, v)
    except (NotImplementedError, RuntimeError):
        out = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), dropout_p=0.0, is_causal=False).squeeze(1)
    return out.transpose(1, 2).reshape(B, C, H, W)


def pytorch_attention(q, k, v):
    """PyTorch attention for 4D tensors (B, C, H, W)."""
    B, C, H, W = q.shape
    q, k, v = [t.view(B, 1, C, -1).transpose(2, 3).contiguous() for t in (q, k, v)]
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.transpose(2, 3).reshape(B, C, H, W)
