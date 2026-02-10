import torch
from unittest.mock import patch, MagicMock

from src.Core.Context import Context
from src.Core.AbstractModel import ModelCapabilities
from src.Processors.HiresFix import HiresFix


from src.Core.AbstractModel import AbstractModel


class FakeModel(AbstractModel):
    def __init__(self):
        super().__init__(model_path=None)
        self._capabilities = ModelCapabilities()
        self._capabilities.supports_hires_fix = True
        self._capabilities.requires_size_conditioning = False
        self.model = object()

    def _create_capabilities(self):
        return self._capabilities

    def load(self, model_path: str = None):
        self._loaded = True
        return self

    def get_model_object(self, name):
        # Provide a latent format object with downscale factor for tests
        class LF:
            downscale_factor = 8
        return LF()

    def encode_prompt(self, prompt, negative_prompt, clip_skip: int = -2):
        # Return a dummy encoded conditioning
        cond = [[torch.zeros(1, 77, 768), {}]]
        return cond, cond

    def generate(self, ctx, positive, negative, latent_image=None, start_step=None, last_step=None, disable_noise=False, callback=None):
        return {"samples": torch.randn(1, 4, 64, 64)}

    def decode(self, latents):
        batch, channels, lh, lw = latents.shape
        return torch.randn(batch, 3, lh * 8, lw * 8)


def _make_latents():
    return {"samples": torch.zeros(1, 4, 64, 64)}


def test_uses_hires_ctx_defaults():
    ctx = Context(prompt="test")
    ctx.generation.width = 512
    ctx.generation.height = 512
    ctx.sampling.steps = 40
    ctx.sampling.cfg = 2.5
    ctx.sampling.sampler = "foo"
    ctx.sampling.scheduler = "bar"

    model = FakeModel()
    latents = _make_latents()

    with patch("src.Utilities.upscale.LatentUpscale") as mock_upscale:
        mock_upscale.return_value.upscale.return_value = ({"samples": torch.zeros(1, 4, 128, 128)},)
        with patch("src.sample.sampling.KSampler") as mock_ksampler:
            mock_ksampler.return_value.sample.return_value = ({"samples": torch.zeros(1, 4, 64, 64)},)

            HiresFix.apply(latents, ctx, model, positive=[[torch.zeros(1, 77, 768), {}]], negative=[[torch.zeros(1, 77, 768), {}]])

            # Verify sampler was called
            assert mock_ksampler.return_value.sample.called
            call_kwargs = mock_ksampler.return_value.sample.call_args.kwargs

            # Steps should reflect hires_ctx.steps -> max(10, int(40 * 0.5)) == 20
            assert call_kwargs["steps"] == 20
            # CFG should be the hires_ctx default (8.0) for non-flux models
            assert call_kwargs["cfg"] == 8.0
            # Sampler name should be taken from hires_ctx (unchanged here)
            assert call_kwargs["sampler_name"] == ctx.sampling.sampler


def test_injects_size_conditioning_for_sdxl():
    ctx = Context(prompt="sdxl test")
    ctx.generation.width = 512
    ctx.generation.height = 512

    model = FakeModel()
    # mark model as requiring size conditioning (SDXL-like)
    model.capabilities.requires_size_conditioning = True

    # Provide encoded conditioning structures (list of [tensor, meta]) that should be updated
    positive = [[torch.zeros(1, 77, 2048), {}]]
    negative = [[torch.zeros(1, 77, 2048), {}]]

    latents = _make_latents()

    with patch("src.Utilities.upscale.LatentUpscale") as mock_upscale:
        mock_upscale.return_value.upscale.return_value = ({"samples": torch.zeros(1, 4, 128, 128)},)
        with patch("src.sample.sampling.KSampler") as mock_ksampler:
            mock_ksampler.return_value.sample.return_value = ({"samples": torch.zeros(1, 4, 64, 64)},)

            HiresFix.apply(latents, ctx, model, positive=positive, negative=negative)

            assert mock_ksampler.return_value.sample.called
            call_kwargs = mock_ksampler.return_value.sample.call_args.kwargs
            called_positive = call_kwargs.get("positive")

            # The metadata dict in the conditioning should now contain width/height matching hires ctx (512*2 = 1024)
            assert isinstance(called_positive, list)
            meta = called_positive[0][1]
            assert meta.get("width") == 1024
            assert meta.get("height") == 1024
            assert meta.get("target_width") == 1024
            assert meta.get("target_height") == 1024


def test_respects_cfg_for_flux():
    ctx = Context(prompt="flux test")
    ctx.generation.width = 512
    ctx.generation.height = 512
    ctx.sampling.cfg = 1.2

    model = FakeModel()
    model.capabilities.is_flux = True

    latents = _make_latents()

    with patch("src.Utilities.upscale.LatentUpscale") as mock_upscale:
        mock_upscale.return_value.upscale.return_value = ({"samples": torch.zeros(1, 16, 128, 128)},)
        with patch("src.sample.sampling.KSampler") as mock_ksampler:
            mock_ksampler.return_value.sample.return_value = ({"samples": torch.zeros(1, 16, 64, 64)},)

            HiresFix.apply(latents, ctx, model, positive=[[torch.zeros(1, 77, 4096), {}]], negative=[[torch.zeros(1, 77, 4096), {}]])

            assert mock_ksampler.return_value.sample.called
            call_kwargs = mock_ksampler.return_value.sample.call_args.kwargs

            # For Flux models, HiresFix should honor the original ctx.sampling.cfg
            assert call_kwargs["cfg"] == 1.2


def test_reencodes_raw_prompt_for_sdxl():
    ctx = Context(prompt="raw prompt")
    ctx.generation.width = 512
    ctx.generation.height = 512

    model = FakeModel()
    model.capabilities.requires_size_conditioning = True

    latents = _make_latents()

    with patch("src.Utilities.upscale.LatentUpscale") as mock_upscale:
        mock_upscale.return_value.upscale.return_value = ({"samples": torch.zeros(1, 4, 128, 128)},)
        with patch("src.sample.sampling.KSampler") as mock_ksampler:
            mock_ksampler.return_value.sample.return_value = ({"samples": torch.zeros(1, 4, 64, 64)},)

            # Provide raw text prompts (should trigger re-encoding)
            HiresFix.apply(latents, ctx, model, positive="a prompt", negative="")

            assert mock_ksampler.return_value.sample.called
            call_kwargs = mock_ksampler.return_value.sample.call_args.kwargs
            called_positive = call_kwargs.get("positive")

            # Should be encoded and updated to hires size
            assert isinstance(called_positive, list)
            meta = called_positive[0][1]
            assert meta.get("width") == 1024
            assert meta.get("height") == 1024
