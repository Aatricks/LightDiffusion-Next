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


def test_autohdr_detects_lcms_missing_on_import(monkeypatch, caplog):
    """When ImageCms.profileToProfile is unavailable at import time, module sets _HAVE_LCMS=False and logs a warning."""
    import importlib
    from PIL import ImageCms as ImageCmsPIL
    import src.AutoHDR.ahdr as ahdr_module

    orig = ImageCmsPIL.profileToProfile
    try:
        monkeypatch.setattr(ImageCmsPIL, 'profileToProfile', lambda *a, **k: (_ for _ in ()).throw(OSError('cannot build transform')))
        caplog.set_level('WARNING')
        importlib.reload(ahdr_module)

        assert ahdr_module._HAVE_LCMS is False
        assert any('LCMS profile transform not available' in rec.getMessage() for rec in caplog.records)
    finally:
        # Restore clean module state for other tests
        monkeypatch.setattr(ImageCmsPIL, 'profileToProfile', orig)
        importlib.reload(ahdr_module)
