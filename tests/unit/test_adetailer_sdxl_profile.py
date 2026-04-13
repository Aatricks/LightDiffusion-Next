from types import SimpleNamespace

from src.Core.Context import Context
from src.Processors.Adetailer import Adetailer


class DummyModel:
    def __init__(self, uses_dual_clip=False, is_flux=False, is_flux2=False):
        self.capabilities = SimpleNamespace(
            uses_dual_clip=uses_dual_clip,
            is_flux=is_flux,
            is_flux2=is_flux2,
        )


def test_runtime_profile_uses_fast_sdxl_settings():
    ctx = Context.from_kwargs(prompt="test", w=1024, h=1024, scheduler="simple")

    profile = Adetailer._runtime_profile(ctx, DummyModel(uses_dual_clip=True))

    assert profile["is_sdxl"] is True
    assert profile["guide_size"] == 512
    assert profile["max_size"] == 768
    assert profile["steps"] == 8
    assert profile["scheduler"] == "simple"
    assert profile["body_crop_factor"] == 1.4
    assert profile["face_crop_factor"] == 1.6


def test_runtime_profile_keeps_flux_overrides():
    ctx = Context.from_kwargs(prompt="test", w=1024, h=1024, scheduler="simple")

    flux2_profile = Adetailer._runtime_profile(ctx, DummyModel(is_flux=True, is_flux2=True))
    flux_profile = Adetailer._runtime_profile(ctx, DummyModel(is_flux=True, is_flux2=False))

    assert flux2_profile["steps"] == 6
    assert flux2_profile["cfg"] == 1.0
    assert flux_profile["steps"] == 20
    assert flux_profile["cfg"] == 1.0
