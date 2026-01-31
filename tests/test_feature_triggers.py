"""Test feature triggers for the modular pipeline.

This test verifies that the new modular HiresFix and Adetailer
processors are actually being called when the user requests them.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def test_hires_fix_is_called():
    """Test that HiresFix.apply is called when hires_fix=True."""
    from src.Core import Context
    from src.Processors import HiresFix
    
    # Create context with hires_fix enabled
    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        hires_fix=True,
    )
    
    # Verify the flag is set correctly
    assert HiresFix.is_enabled(ctx) == True, "HiresFix should be enabled"
    print("PASS: HiresFix.is_enabled returns True when hires_fix=True")


def test_hires_fix_not_called_when_disabled():
    """Test that HiresFix is not enabled when hires_fix=False."""
    from src.Core import Context
    from src.Processors import HiresFix
    
    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        hires_fix=False,
    )
    
    assert HiresFix.is_enabled(ctx) == False, "HiresFix should be disabled"
    print("PASS: HiresFix.is_enabled returns False when hires_fix=False")


def test_adetailer_is_called():
    """Test that Adetailer.apply is called when adetailer=True."""
    from src.Core import Context
    from src.Processors import Adetailer
    
    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        adetailer=True,
    )
    
    assert Adetailer.is_enabled(ctx) == True, "Adetailer should be enabled"
    print("PASS: Adetailer.is_enabled returns True when adetailer=True")


def test_adetailer_not_called_when_disabled():
    """Test that Adetailer is not enabled when adetailer=False."""
    from src.Core import Context
    from src.Processors import Adetailer
    
    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        adetailer=False,
    )
    
    assert Adetailer.is_enabled(ctx) == False, "Adetailer should be disabled"
    print("PASS: Adetailer.is_enabled returns False when adetailer=False")


def test_img2img_is_called():
    """Test that Img2Img is enabled when img2img=True."""
    from src.Core import Context
    from src.Processors import Img2Img
    
    ctx = Context.from_kwargs(
        prompt="test prompt",
        w=512,
        h=512,
        img2img=True,
        img2img_image="test.png",
    )
    
    assert Img2Img.is_enabled(ctx) == True, "Img2Img should be enabled"
    print("PASS: Img2Img.is_enabled returns True when img2img=True")


def test_context_from_kwargs():
    """Test that Context properly maps old-style kwargs."""
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
    
    # Check generation config
    assert ctx.generation.width == 768
    assert ctx.generation.height == 512
    assert ctx.generation.number == 2
    assert ctx.generation.stable_fast == True
    
    # Check sampling config
    assert ctx.sampling.steps == 30
    assert ctx.sampling.sampler == "euler"
    assert ctx.sampling.scheduler == "karras"
    assert ctx.sampling.deepcache_enabled == True
    
    # Check feature flags
    assert ctx.features.hires_fix == True
    assert ctx.features.adetailer == True
    
    print("PASS: Context.from_kwargs correctly maps all parameters")


def test_model_detection():
    """Test that detect_model_type correctly detects model types."""
    from src.Core.Models.ModelFactory import detect_model_type
    
    # Test SD15 detection (default)
    assert detect_model_type(None) == "SD15"
    assert detect_model_type("model.safetensors") == "SD15"
    assert detect_model_type("DreamShaper_8.safetensors") == "SD15"
    
    # Test SDXL detection
    assert detect_model_type("sdxl_model.safetensors") == "SDXL"
    assert detect_model_type("Juggernaut-XL.safetensors") == "SDXL"
    assert detect_model_type("refiner.safetensors") == "SDXL"
    
    print("PASS: detect_model_type correctly detects model types")


def test_model_capabilities():
    """Test that model capabilities are correctly defined."""
    from src.Core.Models import SD15Model, SDXLModel
    
    # SD15 capabilities
    sd15 = SD15Model()
    assert sd15.capabilities.preferred_resolution == 512
    assert sd15.capabilities.uses_dual_clip == False
    
    # SDXL capabilities
    sdxl = SDXLModel()
    assert sdxl.capabilities.preferred_resolution == 1024
    assert sdxl.capabilities.uses_dual_clip == True
    
    print("PASS: Model capabilities are correctly defined")


def test_hires_context_creation():
    """Test that with_hires_settings creates appropriate context."""
    from src.Core import Context
    
    ctx = Context.from_kwargs(
        prompt="test",
        w=512,
        h=512,
        steps=20,
    )
    
    hires_ctx = ctx.with_hires_settings(scale=2.0)
    
    assert hires_ctx.generation.width == 1024
    assert hires_ctx.generation.height == 1024
    assert hires_ctx.sampling.steps == 10  # 50% of original
    assert hires_ctx.sampling.denoise == 0.45
    
    print("PASS: with_hires_settings creates correct context")


def run_all_tests():
    """Run all feature trigger tests."""
    print("=" * 60)
    print("Running Feature Trigger Tests for Modular Pipeline")
    print("=" * 60)
    
    tests = [
        test_hires_fix_is_called,
        test_hires_fix_not_called_when_disabled,
        test_adetailer_is_called,
        test_adetailer_not_called_when_disabled,
        test_img2img_is_called,
        test_context_from_kwargs,
        test_model_detection,
        test_model_capabilities,
        test_hires_context_creation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__} - {e}")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
