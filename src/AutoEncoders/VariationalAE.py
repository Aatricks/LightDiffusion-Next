import logging
import math
import itertools
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from src.Model import ModelPatcher
import torch.nn as nn

from src.Attention import Attention
from src.AutoEncoders import ResBlock
from src.Device import Device
from src.Utilities import util
from src.cond import cast


class DiagonalGaussianDistribution(object):
    """Diagonal Gaussian distribution parameterized by mean and log-variance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        """Sample using reparameterization trick."""
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        """Compute KL divergence to standard normal."""
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )


class DiagonalGaussianRegularizer(torch.nn.Module):
    """Regularizer for diagonal Gaussian distributions."""

    def __init__(self, sample: bool = True):
        """Initialize regularizer."""
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass."""
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AutoencodingEngine(nn.Module):
    """Autoencoding engine."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, regularizer: nn.Module, flux: bool = False):
        """Initialize encoder, decoder, regularizer."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularization = regularizer
        if not flux:
            self.post_quant_conv = cast.disable_weight_init.Conv2d(4, 4, 1)
            self.quant_conv = cast.disable_weight_init.Conv2d(8, 8, 1)
        
    def get_last_layer(self):
        """Get last decoder layer."""
        return self.decoder.get_last_layer()
    
    def decode(self, z: torch.Tensor, flux:bool = False, **kwargs) -> torch.Tensor:
        """Decode latent tensor."""
        if flux:
            x = self.decoder(z, **kwargs)
            return x
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **kwargs)
        return dec


    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        flux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Encode input tensor."""
        z = self.encoder(x)
        if not flux:
            z = self.quant_conv(z)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

ops = cast.disable_weight_init

if Device.xformers_enabled_vae():
    pass


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Apply swish nonlinearity."""
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    """Upsample layer."""

    def __init__(self, in_channels: int, with_conv: bool):
        """Initialize upsample layer."""
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsample layer."""

    def __init__(self, in_channels: int, with_conv: bool):
        """Initialize downsample layer."""
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """VAE Encoder."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        **ignore_kwargs,
    ):
        """Initialize encoder."""
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
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def to(self, device=None, dtype=None):
        """Move encoder to device/dtype."""
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return super().to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.device != self._device or x.dtype != self._dtype:
            self.to(device=x.device, dtype=x.dtype)
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
    """VAE Decoder."""

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        conv_out_op: nn.Module = ops.Conv2d,
        resnet_op: nn.Module = ResBlock.ResnetBlock,
        attn_op: nn.Module = Attention.AttnBlock,
        **ignorekwargs,
    ):
        """Initialize decoder."""
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        (1,) + tuple(ch_mult)
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

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
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
    """Variational Autoencoder."""

    def __init__(
        self,
        sd: Optional[dict] = None,
        device: Optional[torch.device] = None,
        config: Optional[dict] = None,
        dtype: Optional[torch.dtype] = None,
        flux: Optional[bool] = False,
    ):
        """Initialize VAE."""
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
        self.output_channels = 3
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        self.working_dtypes = [torch.bfloat16, torch.float32]

        if config is None:
            if "decoder.conv_in.weight" in sd:
                # default SD1.x/SD2.x VAE parameters
                ddconfig = {
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
                }

                if (
                    "encoder.down.2.downsample.conv.weight" not in sd
                    and "decoder.up.3.upsample.conv.weight" not in sd
                ):  # Stable diffusion x4 upscaler VAE
                    ddconfig["ch_mult"] = [1, 2, 4]
                    self.downscale_ratio = 4
                    self.upscale_ratio = 4

                self.latent_channels = ddconfig["z_channels"] = sd[
                    "decoder.conv_in.weight"
                ].shape[1]
                # Initialize model
                self.first_stage_model = AutoencodingEngine(
                    Encoder(**ddconfig),
                    Decoder(**ddconfig), 
                    DiagonalGaussianRegularizer(),
                    flux=flux
                )
            else:
                logging.warning("WARNING: No VAE weights detected, VAE not initalized.")
                self.first_stage_model = None
                return
        
        self.first_stage_model = self.first_stage_model.eval()

        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logging.warning("Missing VAE keys {}".format(m))

        if len(u) > 0:
            logging.debug("Leftover VAE keys {}".format(u))

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
        logging.debug(
            "VAE load device: {}, offload device: {}, dtype: {}".format(
                self.device, offload_device, self.vae_dtype
            )
        )


    def vae_encode_crop_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """#### Crop the input pixels to be compatible with the VAE.

        #### Args:
            - `pixels` (torch.Tensor): The input pixel tensor.

        #### Returns:
            - `torch.Tensor`: The cropped pixel tensor.
        """
        (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        return pixels

    def decode(self, samples_in: torch.Tensor, flux:bool = False) -> torch.Tensor:
        """#### Decode the latent samples to pixel samples.

        #### Args:
            - `samples_in` (torch.Tensor): The input latent samples.

        #### Returns:
            - `torch.Tensor`: The decoded pixel samples.
        """
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        
        # Check if tiling is needed (heuristically if memory_used > 80% of free VRAM)
        free_memory = Device.get_free_memory(self.device)
        if memory_used > free_memory * 0.8:
            logging.info("VRAM constraint detected, using tiled VAE decoding")
            return self.decode_tiled(samples_in, flux=flux)

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
                self.first_stage_model.decode(samples, flux=flux).to(self.output_device).float()
            )
        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode_tiled(
        self,
        samples: torch.Tensor,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
        flux: bool = False,
    ) -> torch.Tensor:
        """Decode latents in tiles to save VRAM."""
        output_device = self.output_device
        
        def decode_fn(s):
            return self.first_stage_model.decode(
                s.to(self.device).to(self.vae_dtype), flux=flux
            ).float()

        # Load model once
        Device.load_models_gpu([self.patcher])
        
        # Utilize utility for tiled processing
        decoded = util.tiled_scale(
            samples,
            decode_fn,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            upscale_amount=self.upscale_ratio,
            out_channels=3,
            output_device=output_device,
        )
        
        return self.process_output(decoded).movedim(1, -1)

    def encode(self, pixel_samples: torch.Tensor, flux:bool = False) -> torch.Tensor:
        """#### Encode the pixel samples to latent samples.

        #### Args:
            - `pixel_samples` (torch.Tensor): The input pixel samples.

        #### Returns:
            - `torch.Tensor`: The encoded latent samples.
        """
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        
        # Check if tiling is needed for encoding
        free_memory = Device.get_free_memory(self.device)
        if memory_used > free_memory * 0.8:
            logging.info("VRAM constraint detected, using tiled VAE encoding")
            return self.encode_tiled(pixel_samples, flux=flux)

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
                self.first_stage_model.encode(pixels_in, flux=flux).to(self.output_device).float()
            )

        return samples

    def encode_tiled(
        self,
        pixel_samples: torch.Tensor,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: int = 64,
        flux: bool = False,
    ) -> torch.Tensor:
        """Encode pixels in tiles to save VRAM."""
        output_device = self.output_device
        
        def encode_fn(s):
            s_in = self.process_input(s).to(self.device).to(self.vae_dtype)
            return self.first_stage_model.encode(s_in, flux=flux).float()

        # Load model once
        Device.load_models_gpu([self.patcher])
        
        # Utilize utility for tiled processing
        encoded = util.tiled_scale(
            pixel_samples,
            encode_fn,
            tile_x=tile_x,
            tile_y=tile_y,
            overlap=overlap,
            upscale_amount=1.0 / self.downscale_ratio,
            out_channels=self.latent_channels,
            output_device=output_device,
        )
        
        return encoded

    def get_sd(self):
        """#### Get the state dictionary.
        
        #### Returns:
            - `dict`: The state dictionary.
        """
        return self.first_stage_model.state_dict()


class VAEDecode:
    """#### Class for decoding VAE samples."""

    def decode(self, vae: VAE, samples: dict, flux:bool = False) -> Tuple[torch.Tensor]:
        """#### Decode the VAE samples.

        #### Args:
            - `vae` (VAE): The VAE instance.
            - `samples` (dict): The samples dictionary.

        #### Returns:
            - `Tuple[torch.Tensor]`: The decoded samples.
        """
        return (vae.decode(samples["samples"], flux=flux),)


class VAEEncode:
    """#### Class for encoding VAE samples."""

    def encode(self, vae: VAE, pixels: torch.Tensor, flux:bool = False) -> Tuple[dict]:
        """#### Encode the VAE samples.

        #### Args:
            - `vae` (VAE): The VAE instance.
            - `pixels` (torch.Tensor): The input pixel tensor.

        #### Returns:
            - `Tuple[dict]`: The encoded samples dictionary.
        """
        t = vae.encode(pixels[:, :, :, :3], flux=flux)
        return ({"samples": t},)


class VAELoader:
    """#### Class for loading VAEs."""
    # TODO: scale factor?
    def load_vae(self, vae_name):
        """#### Load the VAE.
        
        #### Args:
            - `vae_name`: The name of the VAE.
        
        #### Returns:
            - `Tuple[VAE]`: The VAE instance.
        """
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = "./include/vae/" + vae_name
            sd = util.load_torch_file(vae_path)
        vae = VAE(sd=sd)
        return (vae,)


