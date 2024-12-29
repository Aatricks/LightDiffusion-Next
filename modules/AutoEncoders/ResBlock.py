from abc import abstractmethod
from typing import Optional, Any, Dict

import torch
from modules.NeuralNetwork import transformer
import torch.nn as nn
import torch.nn.functional as F

from modules.Attention import Attention
from modules.cond import cast, cond
from modules.sample import sampling_util


oai_ops = cast.disable_weight_init


class TimestepBlock1(nn.Module):
    """#### Abstract class representing a timestep block."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the timestep block.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        pass


def forward_timestep_embed1(
    ts: nn.ModuleList,
    x: torch.Tensor,
    emb: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    transformer_options: Optional[Dict[str, Any]] = {},
    output_shape: Optional[torch.Size] = None,
    time_context: Optional[torch.Tensor] = None,
    num_video_frames: Optional[int] = None,
    image_only_indicator: Optional[bool] = None,
) -> torch.Tensor:
    """#### Forward pass for timestep embedding.
    
    #### Args:
        - `ts` (nn.ModuleList): The list of timestep blocks.
        - `x` (torch.Tensor): The input tensor.
        - `emb` (torch.Tensor): The embedding tensor.
        - `context` (torch.Tensor, optional): The context tensor. Defaults to None.
        - `transformer_options` (dict, optional): The transformer options. Defaults to {}.
        - `output_shape` (torch.Size, optional): The output shape. Defaults to None.
        - `time_context` (torch.Tensor, optional): The time context tensor. Defaults to None.
        - `num_video_frames` (int, optional): The number of video frames. Defaults to None.
        - `image_only_indicator` (bool, optional): The image only indicator. Defaults to None.
    
    #### Returns:
        - `torch.Tensor`: The output tensor.
    """
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
    """#### Class representing an upsample layer."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the upsample layer.
        
        #### Args:
            - `channels` (int): The number of input channels.
            - `use_conv` (bool): Whether to use convolution.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `padding` (int, optional): The padding size. Defaults to 1.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = operations.conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
                dtype=dtype,
                device=device,
            )

    def forward(self, x: torch.Tensor, output_shape: Optional[torch.Size] = None) -> torch.Tensor:
        """#### Forward pass for the upsample layer.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `output_shape` (torch.Size, optional): The output shape. Defaults to None.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        assert x.shape[1] == self.channels
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample1(nn.Module):
    """#### Class representing a downsample layer."""
    
    def __init__(
        self,
        channels: int,
        use_conv: bool,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the downsample layer.
        
        #### Args:
            - `channels` (int): The number of input channels.
            - `use_conv` (bool): Whether to use convolution.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `padding` (int, optional): The padding size. Defaults to 1.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = operations.conv_nd(
            dims,
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the downsample layer.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock1(TimestepBlock1):
    """#### Class representing a residual block layer."""
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        kernel_size: int = 3,
        exchange_temb_dims: bool = False,
        skip_t_emb: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations: Any = oai_ops,
    ):
        """#### Initialize the residual block layer.
        
        #### Args:
            - `channels` (int): The number of input channels.
            - `emb_channels` (int): The number of embedding channels.
            - `dropout` (float): The dropout rate.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `use_conv` (bool, optional): Whether to use convolution. Defaults to False.
            - `use_scale_shift_norm` (bool, optional): Whether to use scale shift normalization. Defaults to False.
            - `dims` (int, optional): The number of dimensions. Defaults to 2.
            - `use_checkpoint` (bool, optional): Whether to use checkpointing. Defaults to False.
            - `up` (bool, optional): Whether to use upsampling. Defaults to False.
            - `down` (bool, optional): Whether to use downsampling. Defaults to False.
            - `kernel_size` (int, optional): The kernel size. Defaults to 3.
            - `exchange_temb_dims` (bool, optional): Whether to exchange embedding dimensions. Defaults to False.
            - `skip_t_emb` (bool, optional): Whether to skip embedding. Defaults to False.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device. Defaults to None.
            - `operations` (any, optional): The operations. Defaults to oai_ops.
        """
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                emb_channels,
                (2 * self.out_channels if use_scale_shift_norm else self.out_channels),
                dtype=dtype,
                device=device,
            ),
        )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(
                dims,
                self.out_channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 1, dtype=dtype, device=device
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the residual block layer.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return sampling_util.checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """#### Internal forward pass for the residual block layer.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `emb` (torch.Tensor): The embedding tensor.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if emb_out is not None:
            h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


ops = cast.disable_weight_init


class ResnetBlock(nn.Module):
    """#### Class representing a ResNet block layer."""
    
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ):
        """#### Initialize the ResNet block layer.
        
        #### Args:
            - `in_channels` (int): The number of input channels.
            - `out_channels` (int, optional): The number of output channels. Defaults to None.
            - `conv_shortcut` (bool, optional): Whether to use convolution shortcut. Defaults to False.
            - `dropout` (float): The dropout rate.
            - `temb_channels` (int, optional): The number of embedding channels. Defaults to 512.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Attention.Normalize(in_channels)
        self.conv1 = ops.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Attention.Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = ops.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ops.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the ResNet block layer.
        
        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `temb` (torch.Tensor): The embedding tensor.
        
        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h