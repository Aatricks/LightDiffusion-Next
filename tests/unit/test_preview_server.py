import types

from server import _apply_preview_fidelity_to_app, _restore_preview_settings
from src.user import app_instance as _ai


def test_apply_and_restore_preview_fidelity_high():
    # Remember current values to ensure we restore them at the end
    orig_format = getattr(_ai.app, "preview_format", "WEBP")
    orig_quality = getattr(_ai.app, "preview_quality", 90)

    # Apply a high-fidelity request (preview enabled)
    req = types.SimpleNamespace(preview_fidelity="high", enable_preview=True)
    prev = _apply_preview_fidelity_to_app(req)

    # High fidelity should set PNG + max quality
    assert getattr(_ai.app, "preview_format") in ("PNG", "png")
    assert getattr(_ai.app, "preview_quality") == 100

    # Restore previous settings
    _restore_preview_settings(prev)
    assert getattr(_ai.app, "preview_format") == orig_format
    assert getattr(_ai.app, "preview_quality") == orig_quality


def test_apply_and_restore_preview_fidelity_low():
    orig_format = getattr(_ai.app, "preview_format", "WEBP")
    orig_quality = getattr(_ai.app, "preview_quality", 90)

    req = types.SimpleNamespace(preview_fidelity="low", enable_preview=True)
    prev = _apply_preview_fidelity_to_app(req)

    assert getattr(_ai.app, "preview_format") in ("WEBP", "webp")
    assert getattr(_ai.app, "preview_quality") == 70

    _restore_preview_settings(prev)
    assert getattr(_ai.app, "preview_format") == orig_format
    assert getattr(_ai.app, "preview_quality") == orig_quality


def test_apply_preview_fidelity_balanced_defaults():
    orig_format = getattr(_ai.app, "preview_format", "WEBP")
    orig_quality = getattr(_ai.app, "preview_quality", 90)

    req = types.SimpleNamespace(preview_fidelity=None, enable_preview=True)
    prev = _apply_preview_fidelity_to_app(req)

    assert getattr(_ai.app, "preview_format") in ("WEBP", "webp")
    assert getattr(_ai.app, "preview_quality") == 90

    _restore_preview_settings(prev)
    assert getattr(_ai.app, "preview_format") == orig_format
    assert getattr(_ai.app, "preview_quality") == orig_quality
