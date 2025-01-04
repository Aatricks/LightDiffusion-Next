try :
    import xformers
except ImportError:
    pass
import torch


def attention_xformers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None
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


def attention_pytorch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask=None
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
