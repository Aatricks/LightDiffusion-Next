try:
    import sageattention
except ImportError:
    pass

try:
    import spas_sage_attn
except ImportError:
    pass

try :
    import xformers
except ImportError:
    pass
import torch

BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
except:
    pass


def attention_sage(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None, skip_reshape=False, flux=False
) -> torch.Tensor:
    """#### Make an attention call using SageAttention. Fastest and most accurate attention implementation.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.
        - `skip_reshape` (bool, optional): Whether to skip reshaping. Defaults to `False`.
        - `flux` (bool, optional): Whether to use flux mode. Defaults to `False`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if isinstance(mask, torch.Tensor) and mask.device != q.device:
        # Ensure mask lives on the same device as attention tensors to avoid device mismatch in sageattention.
        mask = mask.to(q.device)

    if not flux:
        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous(),
            (q, k, v),
        )

        # SageAttention expects (batch_size, head_num, seq_len, head_dim) by default
        # Pad head dimension if needed (sageattn requires 64, 96, or 128)
        head_dim_og = dim_head
        if head_dim_og in [64, 96, 128]:
            # Native support, no padding needed
            out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
        elif head_dim_og < 64:
            q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
            k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
            v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
            out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
            out = out[..., :head_dim_og]
        elif head_dim_og > 64 and head_dim_og < 128:
            q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
            k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
            v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
            out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
            out = out[..., :head_dim_og]
        else:
            # Head dimension > 128 not supported by SageAttention, fallback to xformers or PyTorch
            q = q.reshape(b * heads, -1, dim_head).contiguous()
            k = k.reshape(b * heads, -1, dim_head).contiguous()
            v = v.reshape(b * heads, -1, dim_head).contiguous()
            try:
                out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
            except:
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            out = out.reshape(b, heads, -1, dim_head)

        out = (
            out.reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out
    else:
        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        if skip_reshape:
            # Already in correct shape for SageAttention
            head_dim_og = dim_head
            if head_dim_og in [64, 96, 128]:
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
            elif head_dim_og < 64:
                q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
                out = out[..., :head_dim_og]
            elif head_dim_og > 64 and head_dim_og < 128:
                q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
                out = out[..., :head_dim_og]
            else:
                # Fallback to PyTorch for unsupported head dimensions
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        else:
            q, k, v = map(
                lambda t: t.reshape(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            head_dim_og = dim_head
            if head_dim_og in [64, 96, 128]:
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
            elif head_dim_og < 64:
                q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
                out = out[..., :head_dim_og]
            elif head_dim_og > 64 and head_dim_og < 128:
                q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
                out = out[..., :head_dim_og]
            else:
                # Fallback to PyTorch for unsupported head dimensions
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = out.transpose(1, 2).reshape(b, -1, heads * head_dim_og)
        return out


def attention_xformers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None, skip_reshape=False, flux=False
) -> torch.Tensor:
    """#### Make an attention call using xformers. Fastest attention implementation.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if not flux:
        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out
    else:
        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        disabled_xformers = False

        if BROKEN_XFORMERS:
            if b * heads > 65535:
                disabled_xformers = True

        if not disabled_xformers:
            if torch.jit.is_tracing() or torch.jit.is_scripting():
                disabled_xformers = True

        if disabled_xformers:
            return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

        if skip_reshape:
            q, k, v = map(
                lambda t: t.reshape(b * heads, -1, dim_head),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda t: t.reshape(b, -1, heads, dim_head),
                (q, k, v),
            )

        if mask is not None:
            pad = 8 - q.shape[1] % 8
            mask_out = torch.empty(
                [q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device
            )
            mask_out[:, :, : mask.shape[-1]] = mask
            mask = mask_out[:, :, : mask.shape[-1]]

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

        if skip_reshape:
            out = (
                out.unsqueeze(0)
                .reshape(b, heads, -1, dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, -1, heads * dim_head)
            )
        else:
            out = out.reshape(b, -1, heads * dim_head)

        return out


def attention_pytorch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None, skip_reshape=False, flux=False
) -> torch.Tensor:
    """#### Make an attention call using PyTorch.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same last dimension as q.
        - `v` (torch.Tensor): The value tensor, must have the same shape as k.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if not flux:
        b, seq_len_q, total_dim = q.shape
        _, seq_len_kv, _ = k.shape
        dim_head = total_dim // heads
        
        # Check if dimension is divisible
        if total_dim % heads != 0:
            import logging
            logging.error(f"ERROR: total_dim({total_dim}) not divisible by heads({heads})")
            raise RuntimeError(f"total_dim({total_dim}) must be divisible by heads({heads})")
        
        # Reshape q, k, v to separate heads
        # q: [b, seq_len_q, heads, dim_head] -> [b, heads, seq_len_q, dim_head]
        # k, v: [b, seq_len_kv, heads, dim_head] -> [b, heads, seq_len_kv, dim_head]
        q = q.view(b, seq_len_q, heads, dim_head).transpose(1, 2)
        k = k.view(b, seq_len_kv, heads, dim_head).transpose(1, 2)
        v = v.view(b, seq_len_kv, heads, dim_head).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        # out: [b, heads, seq_len_q, dim_head] -> [b, seq_len_q, heads, dim_head] -> [b, seq_len_q, total_dim]
        out = out.transpose(1, 2).reshape(b, seq_len_q, total_dim)
        return out
    else:
        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out

def sage_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using SageAttention.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.

    Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    # SageAttention expects (batch_size, head_num, seq_len, head_dim)
    head_dim_og = C
    if head_dim_og in [64, 96, 128]:
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    elif head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        out = out[..., :head_dim_og]
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
        out = sageattention.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        out = out[..., :head_dim_og]
    else:
        # Fallback to PyTorch for unsupported head dimensions
        q = q.squeeze(1).transpose(1, 2).contiguous()
        k = k.squeeze(1).transpose(1, 2).contiguous()
        v = v.squeeze(1).transpose(1, 2).contiguous()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).unsqueeze(1)
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out


def attention_sparge(q, k, v, heads, mask=None, skip_reshape=False, flux=False):
    """SpargeAttn cross attention (Sparse + SageAttention) with automatic head dimension handling."""
    if not flux:
        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .contiguous(),
            (q, k, v),
        )

        # SpargeAttn expects (batch_size, head_num, seq_len, head_dim) 
        # Only supports head dimensions [64, 96, 128] like SageAttention
        head_dim_og = dim_head
        if head_dim_og in [64, 96, 128]:
            # Native support, use SpargeAttn with default thresholds
            out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                q, k, v, 
                simthreshd1=0.6, 
                cdfthreshd=0.97, 
                pvthreshd=15, 
                is_causal=False
            )
        elif head_dim_og < 64:
            q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
            k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
            v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
            out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                q, k, v,
                simthreshd1=0.6,
                cdfthreshd=0.97,
                pvthreshd=15,
                is_causal=False
            )
            out = out[..., :head_dim_og]
        elif head_dim_og > 64 and head_dim_og < 128:
            q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
            k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
            v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
            out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                q, k, v,
                simthreshd1=0.6,
                cdfthreshd=0.97,
                pvthreshd=15,
                is_causal=False
            )
            out = out[..., :head_dim_og]
        else:
            # Head dimension > 128 not supported, fallback to SageAttention
            out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)

        out = (
            out.reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out
    else:
        if skip_reshape:
            b, _, _, dim_head = q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads

        if skip_reshape:
            head_dim_og = dim_head
            if head_dim_og in [64, 96, 128]:
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
            elif head_dim_og < 64:
                q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
                out = out[..., :head_dim_og]
            elif head_dim_og > 64 and head_dim_og < 128:
                q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
                out = out[..., :head_dim_og]
            else:
                # Fallback to SageAttention for unsupported head dimensions
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)
        else:
            q, k, v = map(
                lambda t: t.reshape(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            head_dim_og = dim_head
            if head_dim_og in [64, 96, 128]:
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
            elif head_dim_og < 64:
                q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
                out = out[..., :head_dim_og]
            elif head_dim_og > 64 and head_dim_og < 128:
                q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
                k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
                v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
                out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
                    q, k, v,
                    simthreshd1=0.6,
                    cdfthreshd=0.97,
                    pvthreshd=15,
                    is_causal=False
                )
                out = out[..., :head_dim_og]
            else:
                # Fallback to SageAttention for unsupported head dimensions
                out = sageattention.sageattn(q, k, v, tensor_layout="HND", attn_mask=mask, is_causal=False)

        out = out.transpose(1, 2).reshape(b, -1, heads * head_dim_og)
        return out


def sparge_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using SpargeAttn (Sparse + SageAttention).

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.

    Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    # SpargeAttn expects (batch_size, head_num, seq_len, head_dim)
    head_dim_og = C
    if head_dim_og in [64, 96, 128]:
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.97,
            pvthreshd=15,
            is_causal=False
        )
    elif head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.97,
            pvthreshd=15,
            is_causal=False
        )
        out = out[..., :head_dim_og]
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
        out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
            q, k, v,
            simthreshd1=0.6,
            cdfthreshd=0.97,
            pvthreshd=15,
            is_causal=False
        )
        out = out[..., :head_dim_og]
    else:
        # Fallback to SageAttention for unsupported head dimensions
        q_sage = q
        k_sage = k
        v_sage = v
        out = sageattention.sageattn(q_sage, k_sage, v_sage, tensor_layout="HND", is_causal=False)
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out


def xformers_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using xformers.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q`.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q`.

    Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


def pytorch_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """#### Compute attention using PyTorch.

    #### Args:
        - `q` (torch.Tensor): The query tensor.
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out
