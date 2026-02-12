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


def test_autohdr_handles_rgba_image(monkeypatch):
    """AutoHDR should accept RGBA images and preserve the alpha channel."""
    img = _make_image(batch=1, h=64, w=64, c=4)

    out = AutoHDRProcessor.apply(img, ctx=None)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == img.shape[0]
    # Ensure output still has 4 channels
    assert out.shape[-1] == 4


def test_autohdr_rgba_fallback_on_profile_error(monkeypatch):
    """When ImageCms.profileToProfile raises for RGBA input, AutoHDR should fallback without raising."""
    img = _make_image(batch=1, h=64, w=64, c=4)

    import src.AutoHDR.ahdr as ahdr_module

    def fake_profile_to_profile(*args, **kwargs):
        raise OSError("cannot build transform")

    monkeypatch.setattr('PIL.ImageCms.profileToProfile', fake_profile_to_profile)

    out = AutoHDRProcessor.apply(img, ctx=None)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == img.shape[0]
    assert out.shape[-1] == 4


def test_autohdr_disables_lcms_on_runtime_failure(monkeypatch):
    """When ImageCms.profileToProfile fails at runtime, AutoHDR should set _HAVE_LCMS=False."""
    import src.AutoHDR.ahdr as ahdr_module

    # Simulate LCMS being available at import time
    ahdr_module._HAVE_LCMS = True

    # But make profileToProfile raise during runtime
    def fake_profile_to_profile(*args, **kwargs):
        raise OSError("cannot build transform")

    monkeypatch.setattr('PIL.ImageCms.profileToProfile', fake_profile_to_profile)

    img = _make_image()
    out = AutoHDRProcessor.apply(img, ctx=None)
    assert isinstance(out, torch.Tensor)
    assert ahdr_module._HAVE_LCMS is False
