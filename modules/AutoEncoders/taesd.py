"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
# TODO: Check if multiprocessing is possible for this module
from PIL import Image
import numpy as np
import torch
from modules import util
import torch.nn as nn

from modules.cond import cast, cond
from modules.user import app_instance

def conv(n_in, n_out, **kwargs):
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
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    """#### Class representing a block layer."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = cast.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder2(latent_channels=4):
    """#### Create an encoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The encoder.
    """
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def Decoder2(latent_channels=4):
    """#### Create a decoder.

    #### Args:
        - `latent_channels` (int, optional): The number of latent channels. Defaults to 4.

    #### Returns:
        - `torch.nn.Module`: The decoder.
    """
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
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
        - `decoder_path` (str, optional): Path to the decoder model file. Defaults to "./_internal/vae_approx/taesd_decoder.safetensors".
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

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        super().__init__()
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.taesd_encoder = Encoder2(latent_channels)
        self.taesd_decoder = Decoder2(latent_channels)
        decoder_path = "./_internal/vae_approx/taesd_decoder.safetensors" if decoder_path is None else decoder_path
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(util.load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(util.load_torch_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x):
        """#### Scales raw latents to the range [0, 1].
        
        #### Args:
            - `x` (torch.Tensor): The raw latents.
            
        #### Returns:
            - `torch.Tensor`: The scaled latents.
        """
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """#### Unscales latents from the range [0, 1] to raw latents.
        
        #### Args:
            - `x` (torch.Tensor): The scaled latents.
            
        #### Returns:
            - `torch.Tensor`: The raw latents.
        """
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
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

    def encode(self, x):
        """#### Encodes the given input to the latent space.
        
        #### Args:
            - `x` (torch.Tensor): The input.
            
        #### Returns:
            - `torch.Tensor`: The latent representation.
        """
        device = next(self.taesd_encoder.parameters()).device
        x = x.to(device) 
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift

def taesd_preview(x):
    """#### Preview the input latent as an image.
    
    Uses the TAESD model to decode the latent and updates the image in the App.
    
    #### Args:
        - `x` (torch.Tensor): The input latent.
    """
    if app_instance.app.previewer_checkbox.get() == True:
        taesd_instance = TAESD()
        for image in taesd_instance.decode(x[0].unsqueeze(0))[0]:
            i = 255.0 * image.cpu().detach().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")
        app_instance.app.update_image(img)
    else:
        pass
