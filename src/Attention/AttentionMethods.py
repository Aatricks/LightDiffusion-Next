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
        - `k` (torch.Tensor): The key tensor, must have the same shape as `q.
        - `v` (torch.Tensor): The value tensor, must have the same shape as `q.
        - `heads` (int): The number of heads, must be a divisor of the hidden dimension.
        - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
    if not flux:
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
