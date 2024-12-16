from abc import abstractmethod
from modules import transformer
import torch.nn as nn
import torch.nn.functional as F

from modules.cond import cast, cond
from modules.sample import sampling_util


oai_ops = cast.disable_weight_init


class TimestepBlock1(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


def forward_timestep_embed1(
    ts,
    x,
    emb,
    context=None,
    transformer_options={},
    output_shape=None,
    time_context=None,
    num_video_frames=None,
    image_only_indicator=None,
):
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
    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
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

    def forward(self, x, output_shape=None):
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
    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
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

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock1(TimestepBlock1):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
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

    def forward(self, x, emb):
        return sampling_util.checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
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
