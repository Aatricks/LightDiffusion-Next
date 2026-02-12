"""Unit tests for feature-trigger flags and context mapping.

Moved from top-level `tests/test_feature_triggers.py` and cleaned up to
avoid duplication with `tests/unit/test_model_detection.py`.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def test_hires_fix_is_called():
    """HiresFix flag should be enabled when requested."""
    from src.Core import Context
    from src.Processors import HiresFix

    ctx = Context.from_kwargs(prompt="test prompt", w=512, h=512, hires_fix=True)
    assert HiresFix.is_enabled(ctx) is True


def test_hires_fix_not_called_when_disabled():
    from src.Core import Context
    from src.Processors import HiresFix

    ctx = Context.from_kwargs(prompt="test prompt", w=512, h=512, hires_fix=False)
    assert HiresFix.is_enabled(ctx) is False


def test_adetailer_is_called():
    from src.Core import Context
    from src.Processors import Adetailer

    ctx = Context.from_kwargs(prompt="test prompt", w=512, h=512, adetailer=True)
    assert Adetailer.is_enabled(ctx) is True


def test_adetailer_not_called_when_disabled():
    from src.Core import Context
    from src.Processors import Adetailer

    ctx = Context.from_kwargs(prompt="test prompt", w=512, h=512, adetailer=False)
    assert Adetailer.is_enabled(ctx) is False


def test_img2img_is_called():
    from src.Core import Context
    from src.Processors import Img2Img

    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        img2img=True,
        img2img_image="test.png",
    )
    assert Img2Img.is_enabled(ctx) is True


def test_context_from_kwargs_mappings():
    """Verify Context.from_kwargs maps legacy kwargs to the new Context."""
    from src.Core import Context

    ctx = Context.from_kwargs(
        prompt="a beautiful landscape",
        w=768,
        h=512,
        number=2,
        batch=1,
        scheduler="karras",
        sampler="euler",
        steps=30,
        hires_fix=True,
        adetailer=True,
        stable_fast=True,
        deepcache_enabled=True,
        multiscale_preset="performance",
    )

    assert ctx.generation.width == 768
    assert ctx.generation.height == 512
    assert ctx.generation.number == 2
    assert ctx.generation.stable_fast is True
    assert ctx.sampling.steps == 30
    assert ctx.sampling.sampler == "euler"
    assert ctx.sampling.scheduler == "karras"
    assert ctx.sampling.deepcache_enabled is True
    assert ctx.features.hires_fix is True
    assert ctx.features.adetailer is True


def test_hires_context_creation():
    from src.Core import Context

    ctx = Context.from_kwargs(prompt="test", w=512, h=512, steps=20)
    hires_ctx = ctx.with_hires_settings(scale=2.0)

    assert hires_ctx.generation.width == 1024
    assert hires_ctx.generation.height == 1024
    assert hires_ctx.sampling.steps == 10
    assert hires_ctx.sampling.denoise == 0.45
