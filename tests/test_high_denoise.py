"""
Test img2img with HIGH DENOISE to verify complete color transformation.
Using denoise=0.95 should almost completely regenerate the image.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from PIL import Image
from src.user.pipeline import pipeline


def create_red_image():
    """Create a simple red test image."""
    img = Image.new('RGB', (512, 512), color='red')
    path = project_root / "tests" / "test_solid_red.png"
    img.save(path)
    print(f"Created solid red image at {path}")
    return str(path)


def test_high_denoise():
    """Test with denoise=0.95 - should almost fully regenerate to blue."""
    img_path = create_red_image()
    
    print("\n" + "="*60)
    print("HIGH DENOISE TEST (0.95) - Should generate BLUE image")
    print("="*60)
    
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
        img2img_denoise=0.95,  # Very high - almost full generation
        hires_fix=False,
        adetailer=False,
        enable_multiscale=False,
    )
    
    print("Test complete. Check output/Img2Img for latest image.")
    return result


if __name__ == "__main__":
    test_high_denoise()
