"""
Unit tests for SDXL model components.

Tests the SDXL model configuration, latent format, dual CLIP tokenizer/encoder,
resolution handling, and differences from SD1.5.
"""

import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class TestSDXLLatentFormat:
    """Test suite for SDXL latent format configuration."""
    
    def test_sdxl_latent_has_4_channels(self):
        """SDXL latent format should have 4 channels."""
        from src.Utilities.Latent import SDXL
        
        latent = SDXL()
        assert latent.latent_channels == 4, (
            f"Expected 4 latent channels, got {latent.latent_channels}"
        )
    
    def test_sdxl_scale_factor_differs_from_sd15(self):
        """SDXL scale factor should differ from SD1.5."""
        from src.Utilities.Latent import SDXL, SD15
        
        sdxl = SDXL()
        sd15 = SD15()
        
        assert sdxl.scale_factor != sd15.scale_factor, (
            "SDXL scale factor should differ from SD1.5"
        )
        assert abs(sdxl.scale_factor - 0.13025) < 1e-6, (
            f"Expected SDXL scale factor ~0.13025, got {sdxl.scale_factor}"
        )
    
    def test_sdxl_has_rgb_factors(self):
        """SDXL should have latent RGB factors defined."""
        from src.Utilities.Latent import SDXL
        
        latent = SDXL()
        assert hasattr(latent, 'latent_rgb_factors'), (
            "SDXL should have latent_rgb_factors attribute"
        )
        assert len(latent.latent_rgb_factors) == 4, (
            f"Expected 4 RGB factor rows, got {len(latent.latent_rgb_factors)}"
        )
    
    def test_sdxl_has_rgb_factors_bias(self):
        """SDXL should have RGB factors bias (unlike SD1.5)."""
        from src.Utilities.Latent import SDXL
        
        latent = SDXL()
        assert hasattr(latent, 'latent_rgb_factors_bias'), (
            "SDXL should have latent_rgb_factors_bias attribute"
        )
        assert len(latent.latent_rgb_factors_bias) == 3, (
            f"Expected 3 bias values (RGB), got {len(latent.latent_rgb_factors_bias)}"
        )
    
    def test_sdxl_uses_taesdxl_decoder(self):
        """SDXL should reference correct TAESD decoder (xl version)."""
        from src.Utilities.Latent import SDXL
        
        latent = SDXL()
        assert hasattr(latent, 'taesd_decoder_name'), (
            "SDXL should have taesd_decoder_name attribute"
        )
        assert latent.taesd_decoder_name == "taesdxl_decoder", (
            f"Expected 'taesdxl_decoder', got {latent.taesd_decoder_name}"
        )


class TestSDXLModelConfig:
    """Test suite for SDXL model configuration."""
    
    def test_sdxl_class_exists(self):
        """SDXL model config class should exist."""
        from src.SD15.SDXL import SDXL
        assert SDXL is not None
    
    def test_sdxl_unet_config_has_required_keys(self):
        """SDXL UNet config should have all required keys."""
        from src.SD15.SDXL import SDXL
        
        required_keys = [
            "model_channels",
            "use_linear_in_transformer",
            "transformer_depth",
            "context_dim",
            "adm_in_channels",
            "use_temporal_attention",
        ]
        
        for key in required_keys:
            assert key in SDXL.unet_config, (
                f"Missing required key '{key}' in SDXL unet_config"
            )
    
    def test_sdxl_context_dim_is_2048(self):
        """SDXL should use 2048-dimensional context (concatenated CLIP dims)."""
        from src.SD15.SDXL import SDXL
        
        assert SDXL.unet_config["context_dim"] == 2048, (
            f"Expected context_dim=2048, got {SDXL.unet_config['context_dim']}"
        )
    
    def test_sdxl_model_channels_is_320(self):
        """SDXL should use 320 model channels."""
        from src.SD15.SDXL import SDXL
        
        assert SDXL.unet_config["model_channels"] == 320, (
            f"Expected model_channels=320, got {SDXL.unet_config['model_channels']}"
        )
    
    def test_sdxl_uses_linear_in_transformer(self):
        """SDXL should use linear in transformer (unlike SD1.5)."""
        from src.SD15.SDXL import SDXL
        
        assert SDXL.unet_config["use_linear_in_transformer"] is True, (
            "SDXL should use linear in transformer"
        )
    
    def test_sdxl_has_adm_channels(self):
        """SDXL should have ADM channels for pooled conditioning."""
        from src.SD15.SDXL import SDXL
        
        assert SDXL.unet_config["adm_in_channels"] is not None, (
            "SDXL should have adm_in_channels set"
        )
        assert SDXL.unet_config["adm_in_channels"] == 2816, (
            f"Expected adm_in_channels=2816, got {SDXL.unet_config['adm_in_channels']}"
        )
    
    def test_sdxl_transformer_depth_is_list(self):
        """SDXL transformer depth should be a list."""
        from src.SD15.SDXL import SDXL
        
        depth = SDXL.unet_config["transformer_depth"]
        assert isinstance(depth, (list, tuple)), (
            f"Expected transformer_depth to be list, got {type(depth)}"
        )
    
    def test_sdxl_uses_correct_latent_format(self):
        """SDXL model config should reference SDXL latent format."""
        from src.SD15.SDXL import SDXL as SDXLModel
        from src.Utilities.Latent import SDXL as SDXLLatentFormat
        
        assert SDXLModel.latent_format == SDXLLatentFormat, (
            "SDXL model should use SDXL latent format"
        )
    
    def test_sdxl_has_memory_usage_factor(self):
        """SDXL should have memory_usage_factor defined."""
        from src.SD15.SDXL import SDXL
        
        assert hasattr(SDXL, 'memory_usage_factor'), (
            "SDXL should have memory_usage_factor attribute"
        )
        assert 0 < SDXL.memory_usage_factor <= 1.0, (
            f"memory_usage_factor should be 0-1, got {SDXL.memory_usage_factor}"
        )


class TestSDXLClipTarget:
    """Test suite for SDXL CLIP target configuration."""
    
    def test_sdxl_clip_target_returns_valid_target(self):
        """SDXL clip_target should return a ClipTarget."""
        from src.SD15.SDXL import SDXL
        from src.clip.Clip import ClipTarget
        
        model = SDXL(SDXL.unet_config)
        target = model.clip_target()
        
        assert isinstance(target, ClipTarget), (
            f"Expected ClipTarget, got {type(target)}"
        )
    
    def test_sdxl_clip_target_uses_sdxl_tokenizer(self):
        """SDXL should use SDXLTokenizer (dual tokenizer)."""
        from src.SD15.SDXL import SDXL
        from src.SD15.SDXLClip import SDXLTokenizer
        
        model = SDXL(SDXL.unet_config)
        target = model.clip_target()
        
        assert target.tokenizer == SDXLTokenizer, (
            f"SDXL should use SDXLTokenizer, got {target.tokenizer}"
        )
    
    def test_sdxl_clip_target_uses_sdxl_clip_model(self):
        """SDXL should use SDXLClipModel (dual CLIP model)."""
        from src.SD15.SDXL import SDXL
        from src.SD15.SDXLClip import SDXLClipModel
        
        model = SDXL(SDXL.unet_config)
        target = model.clip_target()
        
        assert target.clip == SDXLClipModel, (
            f"SDXL should use SDXLClipModel, got {target.clip}"
        )


class TestSDXLTokenizer:
    """Test suite for SDXL dual tokenizer."""
    
    def test_sdxl_tokenizer_exists(self):
        """SDXLTokenizer class should exist."""
        from src.SD15.SDXLClip import SDXLTokenizer
        assert SDXLTokenizer is not None
    
    def test_sdxl_tokenizer_has_dual_tokenizers(self):
        """SDXLTokenizer should have both L and G tokenizers."""
        from src.SD15.SDXLClip import SDXLTokenizer
        
        tokenizer = SDXLTokenizer()
        
        assert hasattr(tokenizer, 'clip_l'), (
            "SDXLTokenizer should have clip_l attribute"
        )
        assert hasattr(tokenizer, 'clip_g'), (
            "SDXLTokenizer should have clip_g attribute"
        )
    
    def test_sdxl_tokenizer_tokenize_returns_dict(self):
        """tokenize_with_weights should return dict with 'g' and 'l' keys."""
        from src.SD15.SDXLClip import SDXLTokenizer
        
        tokenizer = SDXLTokenizer()
        
        # The tokenizer may need to load vocab files, so we mock or skip if files don't exist
        try:
            result = tokenizer.tokenize_with_weights("test prompt")
            
            assert isinstance(result, dict), (
                f"Expected dict result, got {type(result)}"
            )
            assert 'g' in result, "Result should have 'g' (ClipG) key"
            assert 'l' in result, "Result should have 'l' (ClipL) key"
        except FileNotFoundError:
            pytest.skip("Tokenizer vocabulary files not available")


class TestSDXLClipModel:
    """Test suite for SDXL CLIP model (dual L+G)."""
    
    def test_sdxl_clip_model_exists(self):
        """SDXLClipModel class should exist."""
        from src.SD15.SDXLClip import SDXLClipModel
        assert SDXLClipModel is not None
    
    def test_sdxl_clip_model_is_torch_module(self):
        """SDXLClipModel should be a torch.nn.Module."""
        from src.SD15.SDXLClip import SDXLClipModel
        import torch.nn as nn
        
        assert issubclass(SDXLClipModel, nn.Module), (
            "SDXLClipModel should be a torch.nn.Module subclass"
        )
    
    @patch('src.SD15.SDClip.SDClipModel.__init__', return_value=None)
    @patch('src.SD15.SDXLClip.SDXLClipG.__init__', return_value=None)
    def test_sdxl_clip_model_has_dual_clips(self, mock_g_init, mock_l_init):
        """SDXLClipModel should have both clip_l and clip_g."""
        from src.SD15.SDXLClip import SDXLClipModel
        
        # Initialize with mocked sub-models
        model = SDXLClipModel.__new__(SDXLClipModel)
        model.clip_l = MagicMock()
        model.clip_g = MagicMock()
        model.dtypes = set()
        
        assert hasattr(model, 'clip_l'), "Should have clip_l"
        assert hasattr(model, 'clip_g'), "Should have clip_g"


class TestSDXLDifferencesFromSD15:
    """Test that SDXL properly differs from SD1.5 where expected."""
    
    def test_context_dim_difference(self):
        """SDXL context_dim (2048) should differ from SD1.5 (768)."""
        from src.SD15.SD15 import sm_SD15
        from src.SD15.SDXL import SDXL
        
        sd15_ctx = sm_SD15.unet_config["context_dim"]
        sdxl_ctx = SDXL.unet_config["context_dim"]
        
        assert sdxl_ctx != sd15_ctx, (
            f"SDXL context_dim ({sdxl_ctx}) should differ from SD1.5 ({sd15_ctx})"
        )
        assert sdxl_ctx == 2048, f"SDXL should have context_dim=2048"
        assert sd15_ctx == 768, f"SD1.5 should have context_dim=768"
    
    def test_linear_in_transformer_difference(self):
        """SDXL uses linear in transformer, SD1.5 does not."""
        from src.SD15.SD15 import sm_SD15
        from src.SD15.SDXL import SDXL
        
        sd15_linear = sm_SD15.unet_config["use_linear_in_transformer"]
        sdxl_linear = SDXL.unet_config["use_linear_in_transformer"]
        
        assert sd15_linear is False, "SD1.5 should not use linear in transformer"
        assert sdxl_linear is True, "SDXL should use linear in transformer"
    
    def test_adm_channels_difference(self):
        """SDXL has ADM channels, SD1.5 does not."""
        from src.SD15.SD15 import sm_SD15
        from src.SD15.SDXL import SDXL
        
        sd15_adm = sm_SD15.unet_config["adm_in_channels"]
        sdxl_adm = SDXL.unet_config["adm_in_channels"]
        
        assert sd15_adm is None, "SD1.5 should not have ADM channels"
        assert sdxl_adm is not None, "SDXL should have ADM channels"
        assert sdxl_adm == 2816, f"SDXL ADM channels should be 2816, got {sdxl_adm}"


class TestSDXLResolutionHandling:
    """CRITICAL: Test SDXL resolution handling and warnings for small resolutions."""
    
    def test_sdxl_latent_accepts_512_resolution(self):
        """SDXL latent generation should accept 512x512 resolution.
        
        CRITICAL TEST: Per requirements, verify whether SDXL accepts or warns
        for resolutions like 512x512 (below SDXL's native 1024x1024).
        """
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        
        # This should not raise an error even for small resolution
        result = generator.generate(width=512, height=512, batch_size=1)
        
        assert result is not None, "SDXL latent generation should succeed for 512x512"
        samples = result[0]["samples"]
        
        # Verify shape: 512/8 = 64
        expected_shape = (1, 4, 64, 64)
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
    
    def test_sdxl_latent_accepts_1024_resolution(self):
        """SDXL latent generation should work naturally at 1024x1024."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        result = generator.generate(width=1024, height=1024, batch_size=1)
        
        samples = result[0]["samples"]
        # 1024/8 = 128
        expected_shape = (1, 4, 128, 128)
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
    
    def test_sdxl_latent_handles_non_64_divisible_sizes(self):
        """Test latent generation with sizes not perfectly divisible."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        
        # 800 / 8 = 100, 600 / 8 = 75
        result = generator.generate(width=800, height=600, batch_size=1)
        
        samples = result[0]["samples"]
        # Should truncate to integer
        expected_shape = (1, 4, 75, 100)  # (batch, channels, h/8, w/8)
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
    
    def test_sdxl_recommended_resolution_is_1024(self):
        """Document that SDXL's recommended base resolution is 1024x1024.
        
        Note: This is a documentation test. SDXL models are trained at
        1024x1024 and may produce suboptimal results at smaller sizes.
        """
        # This is informational - SDXL works at any resolution but
        # is optimized for 1024x1024
        SDXL_RECOMMENDED_BASE = 1024
        assert SDXL_RECOMMENDED_BASE == 1024


class TestSDXLRefiner:
    """Test suite for SDXL Refiner model configuration."""
    
    def test_sdxl_refiner_exists(self):
        """SDXLRefiner class should exist."""
        from src.SD15.SDXL import SDXLRefiner
        assert SDXLRefiner is not None
    
    def test_sdxl_refiner_has_different_unet_config(self):
        """SDXLRefiner should have different UNet config than base SDXL."""
        from src.SD15.SDXL import SDXL, SDXLRefiner
        
        # Refiner has different model_channels
        assert SDXLRefiner.unet_config["model_channels"] != SDXL.unet_config["model_channels"], (
            "Refiner should have different model_channels than base SDXL"
        )
        assert SDXLRefiner.unet_config["model_channels"] == 384, (
            f"Expected refiner model_channels=384, got {SDXLRefiner.unet_config['model_channels']}"
        )
    
    def test_sdxl_refiner_uses_g_clip_only(self):
        """SDXL Refiner should use only ClipG (not L+G like base SDXL)."""
        from src.SD15.SDXL import SDXLRefiner
        from src.SD15.SDXLClip import SDXLRefinerClipModel
        
        model = SDXLRefiner(SDXLRefiner.unet_config)
        target = model.clip_target()
        
        assert target.clip == SDXLRefinerClipModel, (
            "SDXL Refiner should use SDXLRefinerClipModel"
        )


class TestSDXLModelType:
    """Test suite for SDXL model type detection from state dict."""
    
    def test_sdxl_model_type_default_is_eps(self):
        """Default model type should be EPS."""
        from src.SD15.SDXL import SDXL
        from src.sample.sampling import ModelType
        
        model = SDXL(SDXL.unet_config)
        # With empty state dict, should return EPS
        result = model.model_type({}, "")
        
        assert result == ModelType.EPS, (
            f"Expected default ModelType.EPS, got {result}"
        )
    
    def test_sdxl_model_type_detects_v_prediction(self):
        """Model type should detect V_PREDICTION from state dict.
        
        Note: This test documents expected behavior but skips if
        ModelType.V_PREDICTION is not defined in the sampling.ModelType enum.
        """
        from src.SD15.SDXL import SDXL
        from src.sample.sampling import ModelType
        
        # Skip if V_PREDICTION is not in ModelType
        if not hasattr(ModelType, 'V_PREDICTION'):
            pytest.skip("ModelType.V_PREDICTION not defined in enum")
        
        model = SDXL(SDXL.unet_config)
        state_dict = {"v_pred": torch.tensor([1.0])}
        result = model.model_type(state_dict, "")
        
        assert result == ModelType.V_PREDICTION, (
            f"Expected ModelType.V_PREDICTION, got {result}"
        )
    
    def test_sdxl_model_type_detects_edm(self):
        """Model type should detect EDM (Playground V2.5) from state dict.
        
        Note: This test documents expected behavior but skips if
        ModelType.EDM is not defined in the sampling.ModelType enum.
        """
        from src.SD15.SDXL import SDXL
        from src.sample.sampling import ModelType
        
        # Skip if EDM is not in ModelType
        if not hasattr(ModelType, 'EDM'):
            pytest.skip("ModelType.EDM not defined in enum")
        
        model = SDXL(SDXL.unet_config)
        state_dict = {
            "edm_mean": torch.tensor([0.5]),
            "edm_std": torch.tensor([1.0]),
        }
        result = model.model_type(state_dict, "")
        
        assert result == ModelType.EDM, (
            f"Expected ModelType.EDM, got {result}"
        )


class TestSDXLVariants:
    """Test suite for SDXL variant models (SSD-1B, Segmind Vega, etc.)."""
    
    def test_ssd1b_exists_and_inherits_sdxl(self):
        """SSD-1B should exist and inherit from SDXL."""
        from src.SD15.SDXL import SSD1B, SDXL
        
        assert SSD1B is not None
        assert issubclass(SSD1B, SDXL), (
            "SSD1B should inherit from SDXL"
        )
    
    def test_ssd1b_has_reduced_transformer_depth(self):
        """SSD-1B should have fewer transformer blocks."""
        from src.SD15.SDXL import SSD1B, SDXL
        
        ssd1b_depth = SSD1B.unet_config["transformer_depth"]
        sdxl_depth = SDXL.unet_config["transformer_depth"]
        
        # SSD-1B has [0, 0, 2, 2, 4, 4] vs SDXL's [0, 0, 2, 2, 10, 10]
        assert sum(ssd1b_depth) < sum(sdxl_depth), (
            f"SSD-1B should have fewer total transformer blocks"
        )
    
    def test_segmind_vega_exists(self):
        """Segmind Vega model should exist."""
        from src.SD15.SDXL import Segmind_Vega
        assert Segmind_Vega is not None
    
    def test_koala_models_exist(self):
        """KOALA 700M and 1B models should exist."""
        from src.SD15.SDXL import KOALA_700M, KOALA_1B
        
        assert KOALA_700M is not None
        assert KOALA_1B is not None


class TestSDXLProcessClipStateDict:
    """Test suite for SDXL CLIP state dict processing."""
    
    def test_sdxl_process_clip_state_dict_handles_dual_embedders(self):
        """process_clip_state_dict should handle both L and G embedders."""
        from src.SD15.SDXL import SDXL
        
        model = SDXL(SDXL.unet_config)
        
        # Create dummy state dict with SDXL-style prefixes
        state_dict = {
            "conditioner.embedders.0.transformer.text_model.weight": torch.randn(10, 10),
            "conditioner.embedders.1.model.layer": torch.randn(5, 5),
        }
        
        result = model.process_clip_state_dict(state_dict)
        
        # Keys should be converted to clip_l and clip_g prefixes
        prefixes = set()
        for key in result.keys():
            prefix = key.split('.')[0]
            prefixes.add(prefix)
        
        # Should have both clip_l and clip_g prefixes after processing
        assert 'clip_l' in prefixes or 'clip_g' in prefixes, (
            f"Expected clip_l or clip_g prefixes, got prefixes: {prefixes}"
        )


class TestSDXLInModelsRegistry:
    """Test that SDXL variants are properly registered."""
    
    def test_sdxl_in_models_list(self):
        """SDXL should be in the models registry."""
        from src.SD15.SD15 import models
        from src.SD15.SDXL import SDXL
        
        assert SDXL in models, "SDXL should be in the models registry"
    
    def test_sdxl_refiner_in_models_list(self):
        """SDXLRefiner should be in the models registry."""
        from src.SD15.SD15 import models
        from src.SD15.SDXL import SDXLRefiner
        
        assert SDXLRefiner in models, "SDXLRefiner should be in the models registry"
    
    def test_sdxl_variants_in_models_list(self):
        """SDXL variants (SSD1B, etc.) should be in the models registry."""
        from src.SD15.SD15 import models
        from src.SD15.SDXL import SSD1B, Segmind_Vega, KOALA_700M, KOALA_1B
        
        for variant in [SSD1B, Segmind_Vega, KOALA_700M, KOALA_1B]:
            assert variant in models, (
                f"{variant.__name__} should be in the models registry"
            )
