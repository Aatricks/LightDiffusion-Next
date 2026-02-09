"""Test img2img denoising strength functionality.

Tests the new img2img diffusion mode with configurable denoising strength.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="module")
def dummy_image_path():
    """Create a dummy test image."""
    from PIL import Image
    
    path = project_root / "tests" / "test_img2img_input.png"
    if not path.exists():
        img = Image.new('RGB', (512, 512), color='blue')
        # Add some variation
        for x in range(256):
            for y in range(256):
                img.putpixel((x, y), (x, y, 128))
        img.save(path)
    return str(path)


@pytest.mark.slow
def test_img2img_denoise_low(dummy_image_path):
    """Test img2img with low denoise (0.3) - should preserve most of input."""
    from src.user.pipeline import pipeline
    
    result = pipeline(
        prompt="a beautiful landscape",
        w=512,
        h=512,
        number=1,
        batch=1,
        img2img=True,
        img2img_image=dummy_image_path,
        img2img_denoise=0.3,
        steps=10,
    )
    assert result is not None


@pytest.mark.slow
def test_img2img_denoise_medium(dummy_image_path):
    """Test img2img with medium denoise (0.75) - balanced modification."""
    from src.user.pipeline import pipeline
    
    result = pipeline(
        prompt="a beautiful landscape with mountains",
        w=512,
        h=512,
        number=1,
        batch=1,
        img2img=True,
        img2img_image=dummy_image_path,
        img2img_denoise=0.75,
        steps=15,
    )
    assert result is not None


@pytest.mark.slow
def test_img2img_denoise_high(dummy_image_path):
    """Test img2img with high denoise (1.0) - nearly full generation."""
    from src.user.pipeline import pipeline
    
    result = pipeline(
        prompt="a vibrant sunset over ocean waves",
        w=512,
        h=512,
        number=1,
        batch=1,
        img2img=True,
        img2img_image=dummy_image_path,
        img2img_denoise=1.0,
        steps=20,
    )
    assert result is not None


@pytest.mark.slow
def test_img2img_context_denoise():
    """Test that Context correctly handles img2img_denoise parameter."""
    from src.Core.Context import Context, FeatureFlags
    
    # Test default value
    ff = FeatureFlags()
    assert ff.img2img_denoise == 0.75
    
    # Test custom value
    ff2 = FeatureFlags(img2img_denoise=0.5)
    assert ff2.img2img_denoise == 0.5
    
    # Test Context.from_kwargs
    ctx = Context.from_kwargs(
        prompt="test",
        img2img=True,
        img2img_denoise=0.6,
    )
    assert ctx.features.img2img_denoise == 0.6


def test_img2img_context_defaults():
    """Quick test for Context img2img_denoise defaults (no GPU needed)."""
    from src.Core.Context import Context, FeatureFlags
    
    ff = FeatureFlags()
    assert hasattr(ff, 'img2img_denoise')
    assert ff.img2img_denoise == 0.75
    
    ctx = Context.from_kwargs(prompt="test")
    assert ctx.features.img2img_denoise == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
