import torch
import torch.nn as nn
import logging

from modules.Utilities import util
from modules.Attention import AttentionMethods
from modules.Device import Device
from modules.cond import cast


def Normalize(
    in_channels: int, dtype: torch.dtype = None, device: torch.device = None
) -> torch.nn.GroupNorm:
    """#### Normalize the input channels.

    #### Args:
        - `in_channels` (int): The input channels.
        - `dtype` (torch.dtype, optional): The data type. Defaults to `None`.
        - `device` (torch.device, optional): The device. Defaults to `None`.

    #### Returns:
        - `torch.nn.GroupNorm`: The normalized input channels
    """
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


def optimized_attention_for_device() -> AttentionMethods.attention_pytorch:
    """#### Get the optimized attention for a device.

    #### Returns:
        - `function`: The optimized attention function.
    """
    return AttentionMethods.attention_pytorch


class CrossAttention(nn.Module):
    """#### Cross attention module, which applies attention across the query and context.

    #### Args:
        - `query_dim` (int): The query dimension.
        - `context_dim` (int, optional): The context dimension. Defaults to `None`.
        - `heads` (int, optional): The number of heads. Defaults to `8`.
        - `dim_head` (int, optional): The head dimension. Defaults to `64`.
        - `dropout` (float, optional): The dropout rate. Defaults to `0.0`.
        - `dtype` (torch.dtype, optional): The data type. Defaults to `None`.
        - `device` (torch.device, optional): The device. Defaults to `None`.
        - `operations` (cast.disable_weight_init, optional): The operations. Defaults to `cast.disable_weight_init`.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: cast.disable_weight_init = cast.disable_weight_init,
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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """#### Forward pass of the cross attention module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `context` (torch.Tensor, optional): The context tensor. Defaults to `None`.
            - `value` (torch.Tensor, optional): The value tensor. Defaults to `None`.
            - `mask` (torch.Tensor, optional): The mask tensor. Defaults to `None`.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        q = self.to_q(x)
        context = util.default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = optimized_attention(q, k, v, self.heads)
        return self.to_out(out)


class AttnBlock(nn.Module):
    """#### Attention block, which applies attention to the input tensor.

    #### Args:
        - `in_channels` (int): The input channels.
    """

    def __init__(self, in_channels: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the attention block.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels: int, attn_type: str = "vanilla") -> AttnBlock:
    """#### Make an attention block.

    #### Args:
        - `in_channels` (int): The input channels.
        - `attn_type` (str, optional): The attention type. Defaults to "vanilla".

    #### Returns:
        - `AttnBlock`: A class instance of the attention block.
    """
    return AttnBlock(in_channels)
