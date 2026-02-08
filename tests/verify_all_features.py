
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
    """Create a simple red test image with a white shape for edge detection."""
    img = Image.new('RGB', (512, 512), color='red')
    # Add a white rectangle
    for x in range(100, 412):
        for y in range(100, 412):
            img.putpixel((x, y), (255, 255, 255))
    path = project_root / "tests" / "test_controlnet_input.png"
    img.save(path)
    print(f"Created test image at {path}")
    return str(path)


def test_controlnet_sd15(img_path):
    """Test SD1.5 ControlNet."""
    print("\n" + "="*60)
    print("SD1.5 ControlNet Test: Canny edges")
    print("="*60)
    
    try:
        result = pipeline(
            prompt="a futuristic neon building, high quality",
            w=512,
            h=512,
            number=1,
            steps=20,
            controlnet_model="control_v11p_sd15_canny.safetensors",
            controlnet_type="canny",
            controlnet_strength=1.0,
            img2img_image=img_path,
            realistic_model=False,
        )
        print("SD1.5 ControlNet completed successfully!")
        return True
    except Exception as e:
        print(f"SD1.5 ControlNet FAILED: {e}")
        return False


def test_controlnet_sdxl(img_path):
    """Test SDXL ControlNet."""
    print("\n" + "="*60)
    print("SDXL ControlNet Test: Canny edges")
    print("="*60)
    
    sdxl_path = project_root / "include" / "checkpoints" / "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
    if not sdxl_path.exists():
        print(f"SDXL model not found at {sdxl_path}, skipping...")
        return None

    try:
        result = pipeline(
            prompt="a futuristic neon building, high quality, cinematic",
            w=768,
            h=768,
            number=1,
            steps=20,
            controlnet_model="controlnet-canny-sdxl.fp16.safetensors",
            controlnet_type="canny",
            controlnet_strength=1.0,
            img2img_image=img_path,
            realistic_model=True,
        )
        print("SDXL ControlNet completed successfully!")
        return True
    except Exception as e:
        print(f"SDXL ControlNet FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controlnet_flux(img_path):
    """Test Flux ControlNet."""
    print("\n" + "="*60)
    print("Flux ControlNet Test: Canny edges")
    print("="*60)
    
    flux_path = project_root / "include" / "diffusion_model" / "flux-2-klein-4b.safetensors"
    if not flux_path.exists():
        print(f"Flux model not found at {flux_path}, skipping...")
        return None

    try:
        result = pipeline(
            prompt="a futuristic neon building, high quality",
            w=512,
            h=512,
            number=1,
            steps=4,
            controlnet_model="flux-canny-controlnet-v3.safetensors",
            controlnet_type="canny",
            controlnet_strength=1.0,
            img2img_image=img_path,
            model_path="__FLUX2_KLEIN__",
        )
        print("Flux ControlNet completed successfully!")
        return True
    except Exception as e:
        print(f"Flux ControlNet FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hires_fix():
    """Test HiresFix."""
    print("\n" + "="*60)
    print("HiresFix Test")
    print("="*60)
    
    try:
        result = pipeline(
            prompt="a beautiful landscape, 4k, detailed",
            w=512,
            h=512,
            number=1,
            steps=15,
            hires_fix=True,
        )
        print("HiresFix completed successfully!")
        return True
    except Exception as e:
        print(f"HiresFix FAILED: {e}")
        return False


def test_adetailer():
    """Test Adetailer with a portrait prompt."""
    print("\n" + "="*60)
    print("ADetailer Test")
    print("="*60)
    
    try:
        result = pipeline(
            prompt="a portrait of a beautiful woman, highly detailed eyes and face",
            w=512,
            h=512,
            number=1,
            steps=20,
            adetailer=True,
        )
        print("ADetailer completed successfully!")
        return True
    except Exception as e:
        print(f"ADetailer FAILED: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE FEATURE VERIFICATION")
    print("="*60)
    
    img_path = create_test_image()
    
    cn_sd15 = test_controlnet_sd15(img_path)
    cn_sdxl = test_controlnet_sdxl(img_path)
    cn_flux = test_controlnet_flux(img_path)
    hf = test_hires_fix()
    ad = test_adetailer()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"SD1.5 ControlNet: {'PASS' if cn_sd15 else 'FAIL'}")
    print(f"SDXL ControlNet:  {'PASS' if cn_sdxl else 'FAIL' if cn_sdxl is False else 'SKIPPED'}")
    print(f"Flux ControlNet:  {'PASS' if cn_flux else 'FAIL' if cn_flux is False else 'SKIPPED'}")
    print(f"HiresFix:        {'PASS' if hf else 'FAIL'}")
    print(f"ADetailer:       {'PASS' if ad else 'FAIL'}")
    print("="*60)
