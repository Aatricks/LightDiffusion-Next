import torch
from unittest.mock import patch

from src.Processors.AutoHDRProcessor import AutoHDRProcessor


def _make_image(batch=1, h=64, w=64, c=3):
    # Image expected shape for the processor (B, H, W, C)
    return torch.rand(batch, h, w, c)


def test_autohdr_fallback_on_profile_error(monkeypatch):
    img = _make_image()

    # Simulate ImageCms.profileToProfile raising an error
    import src.AutoHDR.ahdr as ahdr_module

    def fake_profile_to_profile(*args, **kwargs):
        raise OSError("cannot build transform")

    monkeypatch.setattr('PIL.ImageCms.profileToProfile', fake_profile_to_profile)

    # Should not raise and should return a tensor of same batch shape
    out = AutoHDRProcessor.apply(img, ctx=None)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == img.shape[0]
