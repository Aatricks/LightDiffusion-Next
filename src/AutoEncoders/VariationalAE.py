"""Variational Autoencoder components."""
import logging
import numpy as np
import torch
import torch.nn as nn
from src.Model import ModelPatcher
from src.Attention import Attention
from src.AutoEncoders import ResBlock
from src.Device import Device
from src.Utilities import util
from src.cond import cast

ops = cast.disable_weight_init


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution."""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)

    def kl(self):
        return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])


class DiagonalGaussianRegularizer(nn.Module):
    """Regularizer for diagonal Gaussian distributions."""
    def __init__(self, sample=True):
        super().__init__()
        self.sample = sample

    def forward(self, z):
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample() if self.sample else posterior.mode()
        kl_loss = torch.sum(posterior.kl()) / posterior.kl().shape[0]
        return z, {"kl_loss": kl_loss}


class AutoencodingEngine(nn.Module):
    """Autoencoding engine."""
    def __init__(self, encoder, decoder, regularizer, flux=False, z_channels=4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularization = regularizer
        if not flux:
            # z_channels for post_quant_conv, z_channels*2 for quant_conv (double_z)
            self.post_quant_conv = ops.Conv2d(z_channels, z_channels, 1)
            self.quant_conv = ops.Conv2d(z_channels * 2, z_channels * 2, 1)

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def decode(self, z, flux=False, **kwargs):
        return self.decoder(z, **kwargs) if flux else self.decoder(self.post_quant_conv(z), **kwargs)

    def encode(self, x, return_reg_log=False, unregularized=False, flux=False):
        z = self.encoder(x) if flux else self.quant_conv(self.encoder(x))
        if unregularized:
            return z, {}
        z, reg_log = self.regularization(z)
        return (z, reg_log) if return_reg_log else z


def nonlinearity(x):
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    """Upsample layer."""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.conv = ops.Conv2d(in_channels, in_channels, 3, 1, 1) if with_conv else None

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.conv else x


class Downsample(nn.Module):
    """Downsample layer."""
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.conv = ops.Conv2d(in_channels, in_channels, 3, 2, 0) if with_conv else None

    def forward(self, x):
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x) if self.conv else x


class Encoder(nn.Module):
    """VAE Encoder."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions,
                 dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels,
                 double_z=True, use_linear_attn=False, attn_type="vanilla", **ignore_kwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = ops.Conv2d(in_channels, ch, 3, 1, 1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResBlock.ResnetBlock(in_channels=block_in, out_channels=block_out,
                                                   temb_channels=0, dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block, down.attn = block, nn.ModuleList()
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResBlock.ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = Attention.make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResBlock.ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.norm_out = Attention.Normalize(block_in)
        self.conv_out = ops.Conv2d(block_in, 2 * z_channels if double_z else z_channels, 3, 1, 1)
        self._device, self._dtype = torch.device("cpu"), torch.float32

    def to(self, device=None, dtype=None):
        if device: self._device = device
        if dtype: self._dtype = dtype
        return super().to(device=device, dtype=dtype)

    def forward(self, x):
        if x.device != self._device or x.dtype != self._dtype:
            self.to(device=x.device, dtype=x.dtype)
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class Decoder(nn.Module):
    """VAE Decoder."""
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks, attn_resolutions,
                 dropout=0.0, resamp_with_conv=True, in_channels, resolution, z_channels,
                 give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 conv_out_op=ops.Conv2d, resnet_op=ResBlock.ResnetBlock, attn_op=Attention.AttnBlock, **ignorekwargs):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[-1]
        self.conv_in = ops.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(resnet_op(in_channels=block_in, out_channels=block_out, temb_channels=0, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block, up.attn = block, nn.ModuleList()
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up)

        self.norm_out = Attention.Normalize(block_in)
        self.conv_out = conv_out_op(block_in, out_ch, 3, 1, 1)

    def forward(self, z, **kwargs):
        h = self.conv_in(z)
        h = self.mid.block_1(h, None, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, None, **kwargs)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class VAE:
    """Variational Autoencoder."""
    def __init__(self, sd=None, device=None, config=None, dtype=None, flux=False):
        self.memory_used_encode = lambda shape, dtype: 1767 * shape[2] * shape[3] * Device.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: 2178 * shape[2] * shape[3] * 64 * Device.dtype_size(dtype)
        self.downscale_ratio = self.upscale_ratio = 8
        self.latent_channels, self.output_channels = 4, 3
        self.process_input = lambda img: img * 2.0 - 1.0
        self.process_output = lambda img: torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.flux = flux

        if config is None and sd and "decoder.conv_in.weight" in sd:
            ddconfig = {"double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3,
                        "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
                        "attn_resolutions": [], "dropout": 0.0}
            if "encoder.down.2.downsample.conv.weight" not in sd:
                ddconfig["ch_mult"] = [1, 2, 4]
                self.downscale_ratio = self.upscale_ratio = 4
            self.latent_channels = ddconfig["z_channels"] = sd["decoder.conv_in.weight"].shape[1]
            self.first_stage_model = AutoencodingEngine(
                Encoder(**ddconfig), Decoder(**ddconfig), DiagonalGaussianRegularizer(), 
                flux=flux, z_channels=self.latent_channels)
        else:
            logging.warning("No VAE weights detected")
            self.first_stage_model = None
            return

        self.first_stage_model.eval()
        m, u = self.first_stage_model.load_state_dict(sd, strict=False)
        if m: logging.warning(f"Missing VAE keys {m}")
        if u: logging.debug(f"Leftover VAE keys {u}")

        self.device = device or Device.vae_device()
        self.vae_dtype = dtype or Device.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = Device.intermediate_device()
        self.patcher = ModelPatcher.ModelPatcher(self.first_stage_model, self.device, Device.vae_offload_device())

    def decode(self, samples_in, flux=None):
        if flux is None:
            flux = self.flux
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        if memory_used > Device.get_free_memory(self.device) * 0.8:
            return self.decode_tiled(samples_in, flux=flux)
        
        Device.load_models_gpu([self.patcher], memory_required=memory_used)
        batch = max(1, int(Device.get_free_memory(self.device) / memory_used))
        out = torch.empty((samples_in.shape[0], 3, samples_in.shape[2] * self.upscale_ratio,
                           samples_in.shape[3] * self.upscale_ratio), device=self.output_device)
        for i in range(0, samples_in.shape[0], batch):
            s = samples_in[i:i+batch].to(self.vae_dtype).to(self.device)
            out[i:i+batch] = self.process_output(self.first_stage_model.decode(s, flux=flux).to(self.output_device).float())
        return out.movedim(1, -1)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16, flux=None):
        if flux is None:
            flux = self.flux
        Device.load_models_gpu([self.patcher])
        decode_fn = lambda s: self.first_stage_model.decode(s.to(self.device).to(self.vae_dtype), flux=flux).float()
        return self.process_output(util.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap,
                                                     self.upscale_ratio, 3, self.output_device)).movedim(1, -1)

    def encode(self, pixel_samples, flux=None):
        if flux is None:
            flux = self.flux
        pixel_samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        if memory_used > Device.get_free_memory(self.device) * 0.8:
            return self.encode_tiled(pixel_samples, flux=flux)
        
        Device.load_models_gpu([self.patcher], memory_required=memory_used)
        batch = max(1, int(Device.get_free_memory(self.device) / memory_used))
        out = torch.empty((pixel_samples.shape[0], self.latent_channels,
                           pixel_samples.shape[2] // self.downscale_ratio,
                           pixel_samples.shape[3] // self.downscale_ratio), device=self.output_device)
        for i in range(0, pixel_samples.shape[0], batch):
            p = self.process_input(pixel_samples[i:i+batch]).to(self.vae_dtype).to(self.device)
            out[i:i+batch] = self.first_stage_model.encode(p, flux=flux).to(self.output_device).float()
        return out

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64, flux=None):
        if flux is None:
            flux = self.flux
        Device.load_models_gpu([self.patcher])
        encode_fn = lambda s: self.first_stage_model.encode(self.process_input(s).to(self.device).to(self.vae_dtype), flux=flux).float()
        return util.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap,
                                1.0 / self.downscale_ratio, self.latent_channels, self.output_device)

    def get_sd(self):
        return self.first_stage_model.state_dict()


class VAEDecode:
    def decode(self, vae, samples, flux=False):
        return (vae.decode(samples["samples"], flux=flux),)


class VAEEncode:
    def encode(self, vae, pixels, flux=False):
        return ({"samples": vae.encode(pixels[:, :, :, :3], flux=flux)},)


class VAELoader:
    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            sd = util.load_torch_file(f"./include/vae/{vae_name}")
        return (VAE(sd=sd),)
