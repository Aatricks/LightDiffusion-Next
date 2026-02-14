import torch
import pytest

pytestmark = pytest.mark.slow
from src.AutoDetailer.ADetailer import _compute_detailer_resize, to_latent_image
from src.NeuralNetwork.flux2.model import Flux2
from src.AutoEncoders.VariationalAE import VAEEncode


class DummyVAE:
    def __init__(self, downscale_ratio=8, latent_channels=4):
        self.downscale_ratio = downscale_ratio
        self.latent_channels = latent_channels

    def encode(self, pixels, flux=False):
        batch = pixels.shape[0]
        latent_h = int(pixels.shape[1]) // self.downscale_ratio
        latent_w = int(pixels.shape[2]) // self.downscale_ratio
        return torch.zeros((batch, self.latent_channels, latent_h, latent_w))


@pytest.mark.slow
def test_compute_and_latent_transformer_options_consistency():
    w, h = 1537, 1567  # edge sizes that require rounding
    guide_size = 512
    max_size = 2048
    upscale, new_w, new_h, force_inpaint = _compute_detailer_resize(w, h, guide_size, max_size)
    # Convert to a fake upscaled image (H, W in pixels order expected by to_latent_image)
    # to_latent_image expects [1, H, W, 3]
    upscaled = torch.zeros((1, new_h, new_w, 3), dtype=torch.float32)
    vae = DummyVAE()
    latent = to_latent_image(upscaled, vae)
    # to_latent_image wraps output in dict with key 'samples'
    samples = latent["samples"] if isinstance(latent, dict) else latent
    # sampling.sample uses latent.shape[2] * 8 to set transformer img_h
    computed_img_h = samples.shape[2] * vae.downscale_ratio
    computed_img_w = samples.shape[3] * vae.downscale_ratio
    assert computed_img_h == new_h
    assert computed_img_w == new_w


@pytest.mark.slow
def test_flux2_forward_resolves_transformer_mismatch():
    model = Flux2()
    # Create an input image with token grid 12x13
    img = torch.randn(1, model.in_channels, 12, 13)
    txt = torch.randn(1, 4, model.params.context_in_dim)
    timesteps = torch.tensor([0.5])
    y = torch.randn(1, model.params.vec_in_dim)

    # Provide transformer_options that are inconsistent (8x8 tokens)
    transformer_options = {"img_h": 8 * 16, "img_w": 8 * 16}

    out = model.forward(img, txt, timesteps, y, transformer_options=transformer_options)
    assert out is not None
    # Output spatial dims should be present and reasonable
    assert out.ndim == 4
