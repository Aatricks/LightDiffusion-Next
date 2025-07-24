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
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.taesd_encoder = Encoder2(latent_channels)
        self.taesd_decoder = Decoder2(latent_channels)
        decoder_path = (
            "./include/vae_approx/taesd_decoder.safetensors"
            if decoder_path is None
            else decoder_path
        )
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(
                util.load_torch_file(encoder_path, safe_load=True)
            )
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(
                util.load_torch_file(decoder_path, safe_load=True)
            )

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


def taesd_preview(x: torch.Tensor, flux: bool = False):
    """#### Preview the batched latent tensors as images.

    #### Args:
        - `x` (torch.Tensor): Input latent tensor with shape [B,C,H,W]
        - `flux` (bool, optional): Whether using flux model (for channel ordering). Defaults to False.
    """
    if app_instance.app.previewer_var.get() is True:
        taesd_instance = TAESD()

        # Handle channel dimension
        if x.shape[1] != 4:
            desired_channels = 4
            current_channels = x.shape[1]

            if current_channels > desired_channels:
                x = x[:, :desired_channels, :, :]
            else:
                padding = torch.zeros(x.shape[0], desired_channels - current_channels,
                                   x.shape[2], x.shape[3], device=x.device)
                x = torch.cat([x, padding], dim=1)

        # Process entire batch at once
        decoded_batch = taesd_instance.decode(x)

        images = []

        # Convert each image in batch
        for decoded in decoded_batch:
            # Handle channel dimension
            if decoded.shape[0] == 1:
                decoded = decoded.repeat(3, 1, 1)

            # Apply different normalization for flux vs standard mode
            if flux:
                # For flux: Assume BGR ordering and different normalization
                decoded = decoded[[2,1,0], :, :] # BGR -> RGB
                # Adjust normalization for flux model range
                decoded = decoded.clamp(-1, 1)
                decoded = (decoded + 1.0) * 0.5 # Scale from [-1,1] to [0,1]
            else:
                # Standard normalization
                decoded = (decoded + 1.0) / 2.0

            # Convert to numpy and uint8
            image_np = (decoded.cpu().detach().numpy() * 255.0)
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

            # Create PIL Image
            img = Image.fromarray(image_np, mode='RGB')
            images.append(img)

        # Update display with all images
        app_instance.app.update_image(images)
    else:
        pass
