"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""

# TODO: Check if multiprocessing is possible for this module
from PIL import Image
import numpy as np
import torch
from src.Utilities import util
import torch.nn as nn

from src.cond import cast
from src.user import app_instance


def conv(n_in: int, n_out: int, **kwargs) -> cast.disable_weight_init.Conv2d:
    """#### Create a convolutional layer.

    #### Args:
        - `n_in` (int): The number of input channels.
        - `n_out` (int): The number of output channels.

    #### Returns:
        - `torch.nn.Module`: The convolutional layer.
    """
    return cast.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """#### Class representing a clamping layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the clamping layer.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The clamped tensor.
        """
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    """#### Class representing a block layer."""

    def __init__(self, n_in: int, n_out: int):
        """#### Initialize the block layer.

        #### Args:
            - `n_in` (int): The number of input channels.
            - `n_out` (int): The number of output channels.

        #### Returns:
            - `Block`: The block layer.
        """
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = (
            cast.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False)
            if n_in != n_out
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder2(latent_channels: int = 4) -> nn.Sequential:
    """#### Create an encoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The encoder.
    """
    return nn.Sequential(
        conv(3, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, 64, stride=2, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        conv(64, latent_channels),
    )


def Decoder2(latent_channels: int = 4) -> nn.Sequential:
    """#### Create a decoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The decoder.
    """
    return nn.Sequential(
        Clamp(),
        conv(latent_channels, 64),
        nn.ReLU(),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        Block(64, 64),
        Block(64, 64),
        nn.Upsample(scale_factor=2),
        conv(64, 64, bias=False),
        Block(64, 64),
        conv(64, 3),
    )


class TAESD(nn.Module):
    """#### Class representing a Tiny AutoEncoder for Stable Diffusion.

    #### Attributes:
        - `latent_magnitude` (float): Magnitude of the latent space.
        - `latent_shift` (float): Shift value for the latent space.
        - `vae_shift` (torch.nn.Parameter): Shift parameter for the VAE.
        - `vae_scale` (torch.nn.Parameter): Scale parameter for the VAE.
        - `taesd_encoder` (Encoder2): Encoder network for the TAESD.
        - `taesd_decoder` (Decoder2): Decoder network for the TAESD.

    #### Args:
        - `encoder_path` (str, optional): Path to the encoder model file. Defaults to None.
        - `decoder_path` (str, optional): Path to the decoder model file. Defaults to "./include/vae_approx/taesd_decoder.safetensors".
        - `latent_channels` (int, optional): Number of channels in the latent space. Defaults to 4.

    #### Methods:
        - `scale_latents(x)`:
            Scales raw latents to the range [0, 1].
        - `unscale_latents(x)`:
            Unscales latents from the range [0, 1] to raw latents.
        - `decode(x)`:
            Decodes the given latent representation to the original space.
        - `encode(x)`:
            Encodes the given input to the latent space.
    """

    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(
        self,
        encoder_path: str = None,
        decoder_path: str = None,
        latent_channels: int = 4,
    ):
        """#### Initialize the TAESD model.

        #### Args:
            - `encoder_path` (str, optional): Path to the encoder model file. Defaults to None.
            - `decoder_path` (str, optional): Path to the decoder model file. Defaults to "./include/vae_approx/taesd_decoder.safetensors".
            - `latent_channels` (int, optional): Number of channels in the latent space. Defaults to 4.
        """
        super().__init__()
        # Use torch factories to keep trace graph free from torch.tensor constructors.
        self.vae_shift = torch.nn.Parameter(torch.zeros(1))
        self.vae_scale = torch.nn.Parameter(torch.ones(1))
        self.taesd_encoder = Encoder2(latent_channels)
        self.taesd_decoder = Decoder2(latent_channels)
        if decoder_path is None:
            if latent_channels == 16:
                decoder_path = "./include/vae_approx/diffusion_pytorch_model.safetensors"
            else:
                decoder_path = "./include/vae_approx/taesd_decoder.safetensors"

        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(
                util.load_torch_file(encoder_path, safe_load=True)
            )
        if decoder_path is not None:
            sd = util.load_torch_file(decoder_path, safe_load=True)
            
            # Load top-level shift/scale parameters if they exist in the checkpoint
            if "vae_shift" in sd:
                self.vae_shift.data.copy_(sd["vae_shift"])
            if "vae_scale" in sd:
                self.vae_scale.data.copy_(sd["vae_scale"])
                
            # Fix for Flux taef1 checkpoint structure
            if any(k.startswith("decoder.layers.") for k in sd.keys()):
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("decoder.layers."):
                        # k is like "decoder.layers.0.weight"
                        parts = k.split(".")
                        # parts = ["decoder", "layers", "0", "weight"]
                        try:
                            idx = int(parts[2])
                            new_idx = idx + 1
                            new_key = f"{new_idx}." + ".".join(parts[3:])
                            new_sd[new_key] = v
                        except ValueError:
                            pass
                self.taesd_decoder.load_state_dict(new_sd)
            else:
                # Filter out top-level parameters before loading into decoder
                decoder_sd = {k: v for k, v in sd.items() if k not in ["vae_shift", "vae_scale"]}
                self.taesd_decoder.load_state_dict(decoder_sd)

    @staticmethod
    def scale_latents(x: torch.Tensor) -> torch.Tensor:
        """#### Scales raw latents to the range [0, 1].

        #### Args:
            - `x` (torch.Tensor): The raw latents.

        #### Returns:
            - `torch.Tensor`: The scaled latents.
        """
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x: torch.Tensor) -> torch.Tensor:
        """#### Unscales latents from the range [0, 1] to raw latents.

        #### Args:
            - `x` (torch.Tensor): The scaled latents.

        #### Returns:
            - `torch.Tensor`: The raw latents.
        """
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """#### Decodes the given latent representation to the original space.

        #### Args:
            - `x` (torch.Tensor): The latent representation.

        #### Returns:
            - `torch.Tensor`: The decoded representation.
        """
        device = next(self.taesd_decoder.parameters()).device
        x = x.to(device)
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """#### Encodes the given input to the latent space.

        #### Args:
            - `x` (torch.Tensor): The input.

        #### Returns:
            - `torch.Tensor`: The latent representation.
        """
        device = next(self.taesd_encoder.parameters()).device
        x = x.to(device)
        x_sample = (x + 1) / 2
        latent = self.taesd_encoder(x_sample)
        latent = latent / self.vae_scale + self.vae_shift
        return latent
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift


from src.Device.ModelCache import get_model_cache


def decode_latents_to_images(x: torch.Tensor, flux: bool = False) -> list[Image.Image]:
    """Decode latents to PIL images using TAESD or approximation.
    
    Includes robustness checks for NaN/Inf and out-of-range values that
    can otherwise cause black preview images.
    """
    if x is None:
        return []
        
    # Robustness: Handle NaNs and extreme values that cause black images
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    
    latent_channels = x.shape[1]
    cache = get_model_cache()
    
    images = []
    decoded_batch = None
    
    # If we have TAESD model for these channels, use it (4 for SD, 16 for Flux1)
    if latent_channels in (4, 16):
        taesd_instance = cache.get_taesd(latent_channels, flux)
        if taesd_instance is None:
            taesd_instance = TAESD(latent_channels=latent_channels)
            # Use same dtype as latents for efficiency
            taesd_instance.to(x.device, dtype=x.dtype)
            cache.cache_taesd(latent_channels, flux, taesd_instance)
        elif next(taesd_instance.parameters()).device != x.device or next(taesd_instance.parameters()).dtype != x.dtype:
            taesd_instance.to(x.device, dtype=x.dtype)

        with torch.no_grad():
            # Optimization for large batches: only preview up to 4 images
            if x.shape[0] > 4:
                x = x[:4]
            
            # Robustness: Clamp latents to a reasonable range for the decoder
            # Standard TAESD expects latents roughly in [-5, 5] to [-10, 10] range
            x_clamped = torch.clamp(x, -12.0, 12.0)
            decoded_batch = taesd_instance.decode(x_clamped)

        # Normalize to [0, 1] range for both SD and Flux
        # Note: No channel swap needed - TAESD outputs RGB correctly for all models
        decoded_batch = decoded_batch.add(1.0).mul(0.5)
        # Apply sRGB transfer to approximate final display appearance before
        # converting to uint8. This helps previews better match final images.
        try:
            if getattr(app_instance.app, "preview_srgb", True):
                from src.Utilities import color as color_utils
                decoded_batch = color_utils.linear_to_srgb(decoded_batch)
        except Exception:
            # Non-fatal: fall back to simple clamp if anything goes wrong
            decoded_batch = decoded_batch.clamp(0, 1)
        finally:
            decoded_batch = decoded_batch.clamp(0, 1)
    
    # For Flux2 (32 channels), use RGB approximation since no TAESD exists for 32ch
    elif latent_channels == 32:
        try:
            from src.Utilities import Latent
            lf = Latent.Flux2()
            factors = torch.tensor(lf.latent_rgb_factors, device=x.device, dtype=x.dtype)
            bias = torch.tensor(lf.latent_rgb_factors_bias, device=x.device, dtype=x.dtype)
            
            # Simple linear preview: [B, 32, H, W] @ [32, 3] -> [B, 3, H, W]
            with torch.no_grad():
                if x.shape[0] > 4:
                    x = x[:4]
                
                # Permute for matmul: [B, H, W, 32] @ [32, 3]
                x_perm = x.permute(0, 2, 3, 1)
                decoded_batch = torch.matmul(x_perm, factors) + bias
                # Back to [B, 3, H, W]
                decoded_batch = decoded_batch.permute(0, 3, 1, 2).clamp(0, 1)
        except Exception:
            return []
    
    else:
        return [] # Unsupported channels

    if decoded_batch is not None:
        # Optimization: Use non_blocking=True for CPU transfer to avoid GPU stall
        # Final safety: ensure no NaNs survived to this point (uint8 cast of NaN is 0)
        decoded_batch = torch.nan_to_num(decoded_batch, nan=0.0)
        decoded_np = (decoded_batch.mul(255.0).to("cpu", dtype=torch.uint8, non_blocking=True).numpy())
        
        # Use simple transpose and PIL conversion
        # decoded_np is [B, C, H, W] -> [B, H, W, C]
        decoded_np = np.transpose(decoded_np, (0, 2, 3, 1))
        
        for i in range(decoded_np.shape[0]):
            img = Image.fromarray(decoded_np[i], mode='RGB')
            # Reduce preview size for faster base64 conversion and lower bandwidth
            if img.width > 512 or img.height > 512:
                # Use higher-quality resampling for downsampling (LANCZOS)
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            images.append(img)
            
    return images


def taesd_preview(x: torch.Tensor, flux: bool = False, step: int = 0, total_steps: int = 0):
    """#### Preview the batched latent tensors as images.

    #### Args:
        - `x` (torch.Tensor): Input latent tensor with shape [B,C,H,W]
        - `flux` (bool, optional): Whether using flux model (for channel ordering). Defaults to False.
        - `step` (int): Current step number.
        - `total_steps` (int): Total number of steps.
    """
    if app_instance.app.previewer_var.get() is True:
        try:
            images = decode_latents_to_images(x, flux)
            if images:
                app_instance.app.update_image(images, step=step, total_steps=total_steps)
        except Exception:
            pass

