# Original code can be found on: https://github.com/black-forest-labs/flux


from dataclasses import dataclass
from einops import rearrange, repeat
import torch
import torch.nn as nn

from src.Attention import Attention
from src.Device import Device
from src.Model import ModelBase
from src.Utilities import Latent
from src.cond import cast, cond
from src.sample import sampling, sampling_util


# Define the attention mechanism
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """#### Compute the attention mechanism.

    #### Args:
        - `q` (Tensor): The query tensor.
        - `k` (Tensor): The key tensor.
        - `v` (Tensor): The value tensor.
        - `pe` (Tensor): The positional encoding tensor.

    #### Returns:
        - `Tensor`: The attention tensor.
    """
    q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = Attention.optimized_attention(q, k, v, heads, skip_reshape=True, flux=True)
    return x

# Define the rotary positional encoding (RoPE)
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """#### Compute the rotary positional encoding.

    #### Args:
        - `pos` (Tensor): The position tensor.
        - `dim` (int): The dimension of the tensor.
        - `theta` (int): The theta value for scaling.

    #### Returns:
        - `Tensor`: The rotary positional encoding tensor.
    """
    assert dim % 2 == 0
    if Device.is_device_mps(pos.device) or Device.is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device
    )
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega
    )
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)

# Apply the rotary positional encoding to the query and key tensors
def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple:
    """#### Apply the rotary positional encoding to the query and key tensors.

    #### Args:
        - `xq` (Tensor): The query tensor.
        - `xk` (Tensor): The key tensor.
        - `freqs_cis` (Tensor): The frequency tensor.

    #### Returns:
        - `tuple`: The modified query and key tensors.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

# Define the embedding class
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        """#### Initialize the EmbedND class.

        #### Args:
            - `dim` (int): The dimension of the tensor.
            - `theta` (int): The theta value for scaling.
            - `axes_dim` (list): The list of axis dimensions.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the EmbedND class.

        #### Args:
            - `ids` (Tensor): The input tensor.

        #### Returns:
            - `Tensor`: The embedded tensor.
        """
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

# Define the MLP embedder class
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None):
        """#### Initialize the MLPEmbedder class.

        #### Args:
            - `in_dim` (int): The input dimension.
            - `hidden_dim` (int): The hidden dimension.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.in_layer = operations.Linear(
            in_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )
        self.silu = nn.SiLU()
        self.out_layer = operations.Linear(
            hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the MLPEmbedder class.

        #### Args:
            - `x` (Tensor): The input tensor.

        #### Returns:
            - `Tensor`: The output tensor.
        """
        return self.out_layer(self.silu(self.in_layer(x)))

# Define the RMS normalization class
class RMSNorm(nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        """#### Initialize the RMSNorm class.

        #### Args:
            - `dim` (int): The dimension of the tensor.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the RMSNorm class.

        #### Args:
            - `x` (Tensor): The input tensor.

        #### Returns:
            - `Tensor`: The normalized tensor.
        """
        return rms_norm(x, self.scale, 1e-6)

# Define the query-key normalization class
class QKNorm(nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        """#### Initialize the QKNorm class.

        #### Args:
            - `dim` (int): The dimension of the tensor.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
        """#### Forward pass for the QKNorm class.

        #### Args:
            - `q` (Tensor): The query tensor.
            - `k` (Tensor): The key tensor.
            - `v` (Tensor): The value tensor.

        #### Returns:
            - `tuple`: The normalized query and key tensors.
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

# Define the self-attention class
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        """#### Initialize the SelfAttention class.

        #### Args:
            - `dim` (int): The dimension of the tensor.
            - `num_heads` (int, optional): The number of attention heads. Defaults to 8.
            - `qkv_bias` (bool, optional): Whether to use bias in QKV projection. Defaults to False.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

# Define the modulation output dataclass
@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor

# Define the modulation class
class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, dtype=None, device=None, operations=None):
        """#### Initialize the Modulation class.

        #### Args:
            - `dim` (int): The dimension of the tensor.
            - `double` (bool): Whether to use double modulation.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(dim, self.multiplier * dim, bias=True, dtype=dtype, device=device)

    def forward(self, vec: torch.Tensor) -> tuple:
        """#### Forward pass for the Modulation class.

        #### Args:
            - `vec` (Tensor): The input tensor.

        #### Returns:
            - `tuple`: The modulation output.
        """
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)

# Define the double stream block class
class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, dtype=None, device=None, operations=None):
        """#### Initialize the DoubleStreamBlock class.

        #### Args:
            - `hidden_size` (int): The hidden size.
            - `num_heads` (int): The number of attention heads.
            - `mlp_ratio` (float): The MLP ratio.
            - `qkv_bias` (bool, optional): Whether to use bias in QKV projection. Defaults to False.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.img_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)
        self.img_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.img_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype, device=device, operations=operations)
        self.txt_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations)
        self.txt_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.txt_mlp = nn.Sequential(
            operations.Linear(hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            operations.Linear(mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> tuple:
        """#### Forward pass for the DoubleStreamBlock class.

        #### Args:
            - `img` (Tensor): The image tensor.
            - `txt` (Tensor): The text tensor.
            - `vec` (Tensor): The vector tensor.
            - `pe` (Tensor): The positional encoding tensor.

        #### Returns:
            - `tuple`: The modified image and text tensors.
        """
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt

# Define the single stream block class
class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float = None, dtype=None, device=None, operations=None):
        """#### Initialize the SingleStreamBlock class.

        #### Args:
            - `hidden_size` (int): The hidden size.
            - `num_heads` (int): The number of attention heads.
            - `mlp_ratio` (float, optional): The MLP ratio. Defaults to 4.0.
            - `qk_scale` (float, optional): The QK scale. Defaults to None.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, dtype=dtype, device=device)
        # proj and mlp_out
        self.linear2 = operations.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device)

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype, device=device, operations=operations)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the SingleStreamBlock class.

        #### Args:
            - `x` (Tensor): The input tensor.
            - `vec` (Tensor): The vector tensor.
            - `pe` (Tensor): The positional encoding tensor.

        #### Returns:
            - `Tensor`: The modified tensor.
        """
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        """#### Initialize the LastLayer class.

        #### Args:
            - `hidden_size` (int): The hidden size.
            - `patch_size` (int): The patch size.
            - `out_channels` (int): The number of output channels.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
        """
        super().__init__()
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.linear = operations.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the LastLayer class.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `vec` (torch.Tensor): The vector tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def pad_to_patch_size(img: torch.Tensor, patch_size: tuple = (2, 2), padding_mode: str = "circular") -> torch.Tensor:
    """#### Pad the image to the specified patch size.

    #### Args:
        - `img` (torch.Tensor): The input image tensor.
        - `patch_size` (tuple, optional): The patch size. Defaults to (2, 2).
        - `padding_mode` (str, optional): The padding mode. Defaults to "circular".

    #### Returns:
        - `torch.Tensor`: The padded image tensor.
    """
    if (
        padding_mode == "circular"
        and torch.jit.is_tracing()
        or torch.jit.is_scripting()
    ):
        padding_mode = "reflect"
    pad_h = (patch_size[0] - img.shape[-2] % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - img.shape[-1] % patch_size[1]) % patch_size[1]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)


try:
    rms_norm_torch = torch.nn.functional.rms_norm
except Exception:
    rms_norm_torch = None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """#### Apply RMS normalization to the input tensor.

    #### Args:
        - `x` (torch.Tensor): The input tensor.
        - `weight` (torch.Tensor): The weight tensor.
        - `eps` (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    #### Returns:
        - `torch.Tensor`: The normalized tensor.
    """
    if rms_norm_torch is not None and not (
        torch.jit.is_tracing() or torch.jit.is_scripting()
    ):
        return rms_norm_torch(
            x,
            weight.shape,
            weight=cast.cast_to(weight, dtype=x.dtype, device=x.device),
            eps=eps,
        )
    else:
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x * rrms) * cast.cast_to(weight, dtype=x.dtype, device=x.device)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux3(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        image_model=None,
        final_layer: bool = True,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        """#### Initialize the Flux3 class.

        #### Args:
            - `image_model` (optional): The image model.
            - `final_layer` (bool, optional): Whether to include the final layer. Defaults to True.
            - `dtype` (optional): The data type.
            - `device` (optional): The device.
            - `operations` (optional): The operations module.
            - `**kwargs`: Additional keyword arguments.
        """
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = operations.Linear(
            self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.vector_in = MLPEmbedder(
            params.vec_in_dim,
            self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = operations.Linear(
            params.context_in_dim, self.hidden_size, dtype=dtype, device=device
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(
                self.hidden_size,
                1,
                self.out_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
    ) -> torch.Tensor:
        """#### Original forward pass for the Flux3 class.

        #### Args:
            - `img` (torch.Tensor): The image tensor.
            - `img_ids` (torch.Tensor): The image IDs tensor.
            - `txt` (torch.Tensor): The text tensor.
            - `txt_ids` (torch.Tensor): The text IDs tensor.
            - `timesteps` (torch.Tensor): The timesteps tensor.
            - `y` (torch.Tensor): The vector tensor.
            - `guidance` (torch.Tensor, optional): The guidance tensor. Defaults to None.
            - `control` (optional): The control tensor. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(sampling_util.timestep_embedding_flux(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                sampling_util.timestep_embedding_flux(guidance, 256).to(img.dtype)
            )

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, y: torch.Tensor, guidance: torch.Tensor, control=None, **kwargs) -> torch.Tensor:
        """#### Forward pass for the Flux3 class.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `timestep` (torch.Tensor): The timestep tensor.
            - `context` (torch.Tensor): The context tensor.
            - `y` (torch.Tensor): The vector tensor.
            - `guidance` (torch.Tensor): The guidance tensor.
            - `control` (optional): The control tensor. Defaults to None.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        bs, c, h, w = x.shape
        patch_size = 2
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(
            x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
        )

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[
                :, None
            ]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[
                None, :
            ]
        )
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control
        )
        return rearrange(
            out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2
        )[:, :, :h, :w]


class Flux2(ModelBase.BaseModel):
    def __init__(self, model_config: dict, model_type=sampling.ModelType.FLUX, device=None):
        """#### Initialize the Flux2 class.

        #### Args:
            - `model_config` (dict): The model configuration.
            - `model_type` (sampling.ModelType, optional): The model type. Defaults to sampling.ModelType.FLUX.
            - `device` (optional): The device.
        """
        super().__init__(model_config, model_type, device=device, unet_model=Flux3, flux=True)

    def encode_adm(self, **kwargs) -> torch.Tensor:
        """#### Encode the ADM.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: The encoded ADM tensor.
        """
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs) -> dict:
        """#### Get extra conditions.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `dict`: The extra conditions.
        """
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = cond.CONDRegular(cross_attn)
        out["guidance"] = cond.CONDRegular(torch.FloatTensor([kwargs.get("guidance", 3.5)]))
        return out


class Flux(ModelBase.BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {}

    unet_extra_config = {}
    latent_format = Latent.Flux1

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict: dict, prefix: str = "", device=None) -> Flux2:
        """#### Get the model.

        #### Args:
            - `state_dict` (dict): The state dictionary.
            - `prefix` (str, optional): The prefix. Defaults to "".
            - `device` (optional): The device.

        #### Returns:
            - `Flux2`: The Flux2 model.
        """
        out = Flux2(self, device=device)
        return out


models = [Flux]
