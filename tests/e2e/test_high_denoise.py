"""
E2E test for very high img2img denoise with assertions.
Moved from top-level `tests/test_high_denoise.py` and updated to assert
on pipeline return value.
"""

import os
import sys
import pytest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

pytestmark = pytest.mark.slow

from PIL import Image
from src.user.pipeline import pipeline


def create_red_image():
    img = Image.new('RGB', (512, 512), color='red')
    path = project_root / 'tests' / 'test_solid_red.png'
    img.save(path)
    return str(path)


@pytest.mark.slow
def test_high_denoise_asserts_result():
    img_path = create_red_image()

    result = pipeline(
        prompt="a solid blue background, pure blue color, no red",
        negative_prompt="red, crimson, scarlet, maroon",
        w=512,
        h=512,
        number=1,
        batch=1,
        steps=20,
        img2img=True,
        img2img_image=img_path,
        img2img_denoise=0.95,
        hires_fix=False,
        adetailer=False,
        enable_multiscale=False,
    )

    # Basic sanity assertions
    assert isinstance(result, dict)
    assert 'original_prompt' in result or 'used_prompt' in result
    # pipeline should not return an empty dict for a valid request
    assert result, "Expected non-empty pipeline result for high-denoise img2img"
