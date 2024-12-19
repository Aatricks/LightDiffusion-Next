import logging
from typing import Tuple, Union
import numpy as np
import torch
from modules.Model import ModelPatcher
import torch.nn as nn

from modules.Attention import Attention
from modules.AutoEncoders import ResBlock
from modules.Device import Device
from modules.cond import cast

class DiagonalGaussianDistribution(object):
    """#### Represents a diagonal Gaussian distribution parameterized by mean and log-variance.
    
    #### Attributes:
        - `parameters` (torch.Tensor): The concatenated mean and log-variance of the distribution.
        - `mean` (torch.Tensor): The mean of the distribution.
        - `logvar` (torch.Tensor): The log-variance of the distribution, clamped between -30.0 and 20.0.
        - `std` (torch.Tensor): The standard deviation of the distribution, computed as exp(0.5 * logvar).
        - `var` (torch.Tensor): The variance of the distribution, computed as exp(logvar).
        - `deterministic` (bool): If True, the distribution is deterministic.
        
    #### Methods:
        - `sample() -> torch.Tensor`:
            Samples from the distribution using the reparameterization trick.
        - `kl(other: DiagonalGaussianDistribution = None) -> torch.Tensor`:
            Computes the Kullback-Leibler divergence between this distribution and a standard normal distribution.
            If `other` is provided, computes the KL divergence between this distribution and `other`.
    """
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        """#### Samples from the distribution using the reparameterization trick.
        
        #### Returns:
            - `torch.Tensor`: A sample from the distribution.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        """#### Computes the Kullback-Leibler divergence between this distribution and a standard normal distribution.
        
        If `other` is provided, computes the KL divergence between this distribution and `other`.
        
        #### Args:
            - `other` (DiagonalGaussianDistribution, optional): Another distribution to compute the KL divergence with.
        #### Returns:
            - `torch.Tensor`: The KL divergence.
        """
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )

class DiagonalGaussianRegularizer(torch.nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AutoencodingEngine(nn.Module):
    def __init__(self, encoder, decoder, regularizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularization = regularizer
        self.post_quant_conv = cast.disable_weight_init.Conv2d(4, 4, 1)
        self.quant_conv = cast.disable_weight_init.Conv2d(8, 8, 1)

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **decoder_kwargs)
        return dec

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        z = self.quant_conv(z)
        z, reg_log = self.regularization(z)
        return z


ops = cast.disable_weight_init

if Device.xformers_enabled_vae():
    import xformers
    import xformers.ops


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = ops.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResBlock.ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock.ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = Attention.make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResBlock.ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Attention.Normalize(block_in)
        self.conv_out = ops.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        conv_out_op=ops.Conv2d,
        resnet_op=ResBlock.ResnetBlock,
        attn_op=Attention.AttnBlock,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = ops.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resnet_op(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Attention.Normalize(block_in)
        self.conv_out = conv_out_op(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        return h

class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None):
        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * Device.dtype_size(
            dtype
        )  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * Device.dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        if config is None:
            config = {
                "encoder": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "decoder": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "regularizer": {"sample": True},
            }
            self.first_stage_model = AutoencodingEngine(
                Encoder(**config["encoder"]),
                Decoder(**config["decoder"]),
                DiagonalGaussianRegularizer(**config["regularizer"]),
            )
        self.first_stage_model = self.first_stage_model.eval()

        self.first_stage_model.load_state_dict(sd, strict=False)

        if device is None:
            device = Device.vae_device()
        self.device = device
        offload_device = Device.vae_offload_device()
        if dtype is None:
            dtype = Device.vae_dtype()
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = Device.intermediate_device()

        self.patcher = ModelPatcher.ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        y = (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        return pixels

    def decode(self, samples_in):
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        Device.load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = Device.get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        pixel_samples = torch.empty(
            (
                samples_in.shape[0],
                3,
                round(samples_in.shape[2] * self.upscale_ratio),
                round(samples_in.shape[3] * self.upscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, samples_in.shape[0], batch_number):
            samples = (
                samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
            )
            pixel_samples[x : x + batch_number] = self.process_output(
                self.first_stage_model.decode(samples).to(self.output_device).float()
            )
        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def encode(self, pixel_samples):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        Device.load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = Device.get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)
        samples = torch.empty(
            (
                pixel_samples.shape[0],
                self.latent_channels,
                round(pixel_samples.shape[2] // self.downscale_ratio),
                round(pixel_samples.shape[3] // self.downscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = (
                self.process_input(pixel_samples[x : x + batch_number])
                .to(self.vae_dtype)
                .to(self.device)
            )
            samples[x : x + batch_number] = (
                self.first_stage_model.encode(pixels_in).to(self.output_device).float()
            )

        return samples
    
class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


class VAEEncode:
    def encode(self, vae, pixels):
        t = vae.encode(pixels[:, :, :, :3])
        return ({"samples": t},)