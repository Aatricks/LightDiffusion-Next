from einops import rearrange

from modules import util

import torch.nn as nn

from modules.Attention import Attention
from modules.Device import Device
from modules.cond import Activation
from modules.cond import cast, cond
from modules.sample import sampling_util


if Device.xformers_enabled():
    import xformers
    import xformers.ops

ops = cast.disable_weight_init

_ATTN_PRECISION = "fp32"


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = util.default(dim_out, dim)
        project_in = (
            nn.Sequential(
                operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU()
            )
            if not glu
            else Activation.GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        ff_in=False,
        inner_dim=None,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn
        self.attn1 = Attention.CrossAttention(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(
            inner_dim,
            dim_out=dim,
            dropout=dropout,
            glu=gated_ff,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        context_dim_attn2 = None
        if not switch_temporal_ca_to_sa:
            context_dim_attn2 = context_dim

        self.attn2 = Attention.CrossAttention(
            query_dim=inner_dim,
            context_dim=context_dim_attn2,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is self-attn if context is none
        self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        return sampling_util.checkpoint(
            self._forward,
            (x, context, transformer_options),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        n = self.norm1(x)
        context_attn1 = None
        value_attn1 = None

        transformer_block = (block[0], block[1], block_index)
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        n = self.attn1(n, context=context_attn1, value=value_attn1)

        x += n

        if self.attn2 is not None:
            n = self.norm2(x)
            context_attn2 = context
            value_attn2 = None

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block
            n = self.attn2(n, context=context_attn2, value=value_attn2)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        if util.exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        if not use_linear:
            self.proj_in = operations.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
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
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(
        list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
    )
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(
            state_dict_keys, transformer_prefix + "{}"
        )
        context_dim = state_dict[
            "{}0.attn2.to_k.weight".format(transformer_prefix)
        ].shape[1]
        use_linear_in_transformer = (
            len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
        )
        time_stack = (
            "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict
            or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
        )
        return (
            last_transformer_depth,
            context_dim,
            use_linear_in_transformer,
            time_stack,
        )
    return None