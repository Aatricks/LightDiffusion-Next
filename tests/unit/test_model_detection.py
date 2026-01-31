"""
Unit tests for model detection functionality.

Tests the detect_model_type function in src/Core/Models/ModelFactory.py
with various filename patterns and edge cases.

Note: GGUF/FLUX support has been removed. GGUF files now raise ValueError.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.Core.Models.ModelFactory import detect_model_type, list_available_models


class TestDetectModelType:
    """Test suite for detect_model_type function."""
    
    # =========================================================================
    # SD1.5 Detection Tests
    # =========================================================================
    
    def test_detect_sd15_from_generic_safetensors(self):
        """SD1.5 should be detected for generic .safetensors files."""
        result = detect_model_type("model.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_from_pt_file(self):
        """SD1.5 should be detected for .pt files without SDXL marker."""
        result = detect_model_type("dreamshaper_8.pt")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_from_pth_file(self):
        """SD1.5 should be detected for .pth files without SDXL marker."""
        result = detect_model_type("anime_model.pth")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_from_dreamshaper(self):
        """DreamShaper models should be detected as SD1.5."""
        result = detect_model_type("DreamShaper_8_pruned.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_from_meina(self):
        """Meina models should be detected as SD1.5."""
        result = detect_model_type("Meina V10 - baked VAE.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_from_realistic_vision(self):
        """Realistic Vision models should be detected as SD1.5."""
        result = detect_model_type("realisticVisionV60.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_with_absolute_path(self):
        """Detection should work with absolute paths."""
        # Windows-style path
        result = detect_model_type("C:\\Models\\checkpoints\\my_model.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
        
        # Unix-style path
        result = detect_model_type("/home/user/models/my_model.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    def test_detect_sd15_with_relative_path(self):
        """Detection should work with relative paths."""
        result = detect_model_type("./include/checkpoints/model.safetensors")
        assert result == "SD15", f"Expected SD15, got {result}"
    
    # =========================================================================
    # SDXL Detection Tests
    # =========================================================================
    
    def test_detect_sdxl_from_filename_marker(self):
        """SDXL should be detected from 'sdxl' in filename."""
        result = detect_model_type("juggernaut_sdxl_v9.safetensors")
        assert result == "SDXL", f"Expected SDXL, got {result}"
    
    def test_detect_sdxl_case_insensitive(self):
        """SDXL detection should be case-insensitive."""
        test_cases = [
            "SDXL_model.safetensors",
            "Sdxl_model.safetensors", 
            "model_SDXL.safetensors",
            "mySdXlModel.safetensors",
        ]
        for filename in test_cases:
            result = detect_model_type(filename)
            assert result == "SDXL", f"Expected SDXL for {filename}, got {result}"
    
    def test_detect_sdxl_from_refiner(self):
        """SDXL should be detected from 'refiner' in filename."""
        result = detect_model_type("sd_xl_refiner_1.0.safetensors")
        assert result == "SDXL", f"Expected SDXL, got {result}"
    
    def test_detect_sdxl_from_hassaku(self):
        """SDXL should be detected from 'hassaku' in filename."""
        result = detect_model_type("hassakuXL_v13.safetensors")
        assert result == "SDXL", f"Expected SDXL, got {result}"
    
    def test_detect_sdxl_juggernaut(self):
        """Juggernaut XL models should be detected as SDXL due to 'juggernaut' indicator."""
        result = detect_model_type("Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors")
        assert result == "SDXL", f"Expected SDXL (juggernaut indicator), got {result}"
    
    def test_detect_sdxl_with_path(self):
        """SDXL detection works with full paths if basename contains marker."""
        # Note: Detection is on os.path.basename(lp), not full path
        result = detect_model_type("/models/checkpoints/sdxl_base_model.safetensors")
        assert result == "SDXL", f"Expected SDXL, got {result}"
        
        # Path with sdxl in directory but not in filename defaults to SD15
        result_nomarker = detect_model_type("/models/sdxl/base_model.safetensors")
        assert result_nomarker == "SD15", f"Expected SD15 (marker not in basename), got {result_nomarker}"
    
    # =========================================================================
    # GGUF Files - No Longer Supported (Must Raise ValueError)
    # =========================================================================
    
    def test_gguf_files_raise_value_error(self):
        """GGUF files should raise ValueError as they're no longer supported."""
        with pytest.raises(ValueError, match="GGUF files not supported"):
            detect_model_type("flux1-dev-Q8_0.gguf")
    
    def test_gguf_any_filename_raises_error(self):
        """Any .gguf file should raise ValueError."""
        test_cases = [
            "my_flux_model.gguf",
            "FLUX_model.gguf",
            "Flux_model.gguf",
            "model_FLUX.gguf",
            "random_model.gguf",
            "/models/flux/flux1-dev.gguf",
        ]
        for filename in test_cases:
            with pytest.raises(ValueError, match="GGUF files not supported"):
                detect_model_type(filename)
    
    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================
    
    def test_detect_none_input(self):
        """None input should return SD15 (default)."""
        result = detect_model_type(None)
        assert result == "SD15", f"Expected SD15 for None input, got {result}"
    
    def test_detect_empty_string(self):
        """Empty string should return SD15 (default)."""
        result = detect_model_type("")
        assert result == "SD15", f"Expected SD15 for empty string, got {result}"
    
    def test_detect_unknown_extension(self):
        """Unknown extensions should default to SD15."""
        result = detect_model_type("model.bin")
        assert result == "SD15", f"Expected SD15 for .bin file, got {result}"
    
    def test_detect_no_extension(self):
        """Files without extension should default to SD15."""
        result = detect_model_type("model_file")
        assert result == "SD15", f"Expected SD15 for no extension, got {result}"
    
    def test_detect_preserves_original_path(self):
        """Detection should not modify the input path."""
        original_path = "path/to/model.safetensors"
        detect_model_type(original_path)
        assert original_path == "path/to/model.safetensors"


class TestListAvailableModels:
    """Test suite for list_available_models function."""
    
    def test_list_returns_list(self):
        """list_available_models should return a list."""
        result = list_available_models()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
    
    def test_list_with_mapping_returns_tuples(self):
        """list_available_models(return_mapping=True) should return list of tuples."""
        result = list_available_models(return_mapping=True)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        # If non-empty, check tuple format
        if result:
            assert all(
                isinstance(item, tuple) and len(item) == 2 
                for item in result
            ), "Each item should be a (display_name, full_path) tuple"
    
    def test_list_filters_valid_extensions(self):
        """Only valid model extensions should be returned (no .gguf)."""
        valid_extensions = (".safetensors", ".pt", ".pth")
        result = list_available_models(return_mapping=True)
        
        for display_name, full_path in result:
            ext = os.path.splitext(display_name.lower())[1]
            assert ext in valid_extensions, (
                f"Invalid extension {ext} in {display_name}"
            )
    
    def test_list_returns_basenames_by_default(self):
        """Default return should be basenames only."""
        result = list_available_models(return_mapping=False)
        
        for name in result:
            # Should not contain path separators
            assert "/" not in name and "\\" not in name, (
                f"Expected basename, got path: {name}"
            )


class TestModelDetectionIntegration:
    """Integration tests for model detection with real file patterns."""
    
    @pytest.mark.parametrize("filename,expected", [
        # SD1.5 models
        ("DreamShaper_8_pruned.safetensors", "SD15"),
        ("v1-5-pruned.safetensors", "SD15"),
        ("anythingV5.safetensors", "SD15"),
        ("deliberate_v3.safetensors", "SD15"),
        ("realisticVision.safetensors", "SD15"),
        
        # SDXL models (contain 'sdxl', 'refiner', 'hassaku', 'juggernaut', or 'xl')
        ("sd_xl_base_1.0.safetensors", "SDXL"),  # contains 'xl'
        ("Juggernaut-XL_v9.safetensors", "SDXL"),  # contains 'juggernaut' and 'xl'
        ("sdxl_vae.safetensors", "SDXL"),
        ("hassakuXLv13.safetensors", "SDXL"),
        ("SDXL_refiner_1.0.safetensors", "SDXL"),
    ])
    def test_detection_matrix(self, filename, expected):
        """Test detection across a matrix of common model filenames."""
        result = detect_model_type(filename)
        assert result == expected, f"Expected {expected} for {filename}, got {result}"
    
    @pytest.mark.parametrize("filename", [
        "flux1-dev-Q8_0.gguf",
        "flux-schnell.gguf",
        "any_model.gguf",
    ])
    def test_gguf_files_raise_error(self, filename):
        """All GGUF files should raise ValueError."""
        with pytest.raises(ValueError, match="GGUF files not supported"):
            detect_model_type(filename)
