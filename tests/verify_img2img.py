"""
Real end-to-end img2img test with actual models.
Tests SD1.5, SDXL, and Flux2 img2img with denoising strength.
"""

import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from PIL import Image
from src.user.pipeline import pipeline


def create_test_image():
    """Create a simple red test image."""
    img = Image.new('RGB', (512, 512), color='red')
    # Add a simple "car shape" with darker red
    for x in range(150, 362):
        for y in range(200, 312):
            img.putpixel((x, y), (180, 0, 0))
    path = project_root / "tests" / "test_red_car.png"
    img.save(path)
    print(f"Created test image at {path}")
    return str(path)


def test_sd15_img2img(img_path):
    """Test SD1.5 img2img - change red car to blue."""
    print("\n" + "="*60)
    print("SD1.5 Img2Img Test: Change red to blue")
    print("="*60)
    
    try:
        result = pipeline(
            prompt="a blue car, photorealistic, high quality",
            negative_prompt="red, red car",
            w=512,
            h=512,
            number=1,
            batch=1,
            steps=15,
            img2img=True,
            img2img_image=img_path,
            img2img_denoise=0.65,  # Balanced change
            realistic_model=False,  # SD1.5
            hires_fix=False,
            adetailer=False,
            enable_multiscale=False,
        )
        print("SD1.5 img2img completed successfully!")
        return True
    except Exception as e:
        print(f"SD1.5 img2img FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sdxl_img2img(img_path):
    """Test SDXL img2img - change red car to green."""
    print("\n" + "="*60)
    print("SDXL Img2Img Test: Change red to green")
    print("="*60)
    
    # Check if SDXL model exists
    sdxl_path = project_root / "include" / "checkpoints" / "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    if not sdxl_path.exists():
        print(f"SDXL model not found at {sdxl_path}, skipping...")
        return None
    
    try:
        result = pipeline(
            prompt="a green car, photorealistic, high quality, 4k",
            negative_prompt="red, red car",
            w=512,
            h=512,
            number=1,
            batch=1,
            steps=15,
            img2img=True,
            img2img_image=img_path,
            img2img_denoise=0.6,
            realistic_model=True,  # SDXL
            hires_fix=False,
            adetailer=False,
            enable_multiscale=False,
        )
        print("SDXL img2img completed successfully!")
        return True
    except Exception as e:
        print(f"SDXL img2img FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flux2_img2img(img_path):
    """Test Flux2 Klein img2img - change red car to yellow."""
    print("\n" + "="*60)
    print("Flux2 Klein Img2Img Test: Change red to yellow")
    print("="*60)
    
    # Check if Flux model exists
    flux_path = project_root / "include" / "diffusion_model" / "flux-2-klein-4b.safetensors"
    if not flux_path.exists():
        print(f"Flux2 model not found at {flux_path}, skipping...")
        return None
    
    try:
        result = pipeline(
            prompt="a yellow car, photorealistic, high quality",
            w=512,
            h=512,
            number=1,
            batch=1,
            steps=4,  # Flux uses fewer steps (distilled)
            img2img=True,
            img2img_image=img_path,
            img2img_denoise=0.7,
            model_path="__FLUX2_KLEIN__",
            hires_fix=False,
            adetailer=False,
            enable_multiscale=False,
        )
        print("Flux2 img2img completed successfully!")
        return True
    except Exception as e:
        print(f"Flux2 img2img FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_denoise_variations(img_path):
    """Test different denoise levels to verify effect."""
    print("\n" + "="*60)
    print("Testing denoise strength variations (0.3, 0.6, 0.9)")
    print("="*60)
    
    results = {}
    for denoise in [0.3, 0.6, 0.9]:
        try:
            print(f"\nTesting denoise={denoise}...")
            result = pipeline(
                prompt="a blue car, photorealistic",
                w=512,
                h=512,
                number=1,
                steps=10,
                img2img=True,
                img2img_image=img_path,
                img2img_denoise=denoise,
                hires_fix=False,
                adetailer=False,
                enable_multiscale=False,
            )
            results[denoise] = True
            print(f"  denoise={denoise} completed successfully")
        except Exception as e:
            results[denoise] = False
            print(f"  denoise={denoise} FAILED: {e}")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("IMG2IMG END-TO-END VERIFICATION")
    print("="*60)
    
    # Create test image
    img_path = create_test_image()
    
    # Run tests
    sd15_result = test_sd15_img2img(img_path)
    sdxl_result = test_sdxl_img2img(img_path)
    flux_result = test_flux2_img2img(img_path)
    denoise_results = test_denoise_variations(img_path)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"SD1.5 img2img: {'PASS' if sd15_result else 'FAIL' if sd15_result is False else 'SKIPPED'}")
    print(f"SDXL img2img:  {'PASS' if sdxl_result else 'FAIL' if sdxl_result is False else 'SKIPPED'}")
    print(f"Flux2 img2img: {'PASS' if flux_result else 'FAIL' if flux_result is False else 'SKIPPED'}")
    print(f"Denoise 0.3:   {'PASS' if denoise_results.get(0.3) else 'FAIL'}")
    print(f"Denoise 0.6:   {'PASS' if denoise_results.get(0.6) else 'FAIL'}")
    print(f"Denoise 0.9:   {'PASS' if denoise_results.get(0.9) else 'FAIL'}")
    print("="*60)
