"""E2E img2img denoise tests (parametrized).

This file was moved from the top-level `tests/` and the three similar
cases were combined into a single parametrized test for clarity.
"""

import os
import sys
from pathlib import Path
import pytest

# Keep project root on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def dummy_image_path():
    """Create a small dummy image used by img2img tests."""
    from PIL import Image

    path = project_root / "tests" / "test_img2img_input.png"
    if not path.exists():
        img = Image.new('RGB', (512, 512), color='blue')
        for x in range(256):
            for y in range(256):
                img.putpixel((x, y), (x, y, 128))
        img.save(path)
    return str(path)


@pytest.mark.slow
@pytest.mark.parametrize("denoise,steps,prompt", [
    (0.3, 10, "a beautiful landscape"),
    (0.75, 15, "a beautiful landscape with mountains"),
    (1.0, 20, "a vibrant sunset over ocean waves"),
])
def test_img2img_denoise_parametrized(dummy_image_path, denoise, steps, prompt):
    from src.user.pipeline import pipeline

    result = pipeline(
        prompt=prompt,
        w=512,
        h=512,
        number=1,
        batch=1,
        img2img=True,
        img2img_image=dummy_image_path,
        img2img_denoise=denoise,
        steps=steps,
    )

    assert isinstance(result, dict)
    assert 'original_prompt' in result or 'used_prompt' in result


@pytest.mark.slow
def test_img2img_context_defaults():
    from src.Core.Context import Context, FeatureFlags

    ff = FeatureFlags()
    assert hasattr(ff, 'img2img_denoise')
    assert ff.img2img_denoise == 0.75

    ctx = Context.from_kwargs(prompt="test")
    assert ctx.features.img2img_denoise == 0.75
