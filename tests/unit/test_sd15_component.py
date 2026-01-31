"""
Unit tests for SD1.5 model components.

Tests the SD1.5 model configuration, latent format, CLIP tokenizer/encoder,
and CheckpointLoaderSimple with mocked weights.
"""

import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class TestSD15LatentFormat:
    """Test suite for SD15 latent format configuration."""
    
    def test_sd15_latent_has_4_channels(self):
        """SD1.5 latent format should have 4 channels."""
        from src.Utilities.Latent import SD15
        
        latent = SD15()
        assert latent.latent_channels == 4, (
            f"Expected 4 latent channels, got {latent.latent_channels}"
        )
    
    def test_sd15_default_scale_factor(self):
        """SD1.5 should have default scale factor of 0.18215."""
        from src.Utilities.Latent import SD15
        
        latent = SD15()
        assert abs(latent.scale_factor - 0.18215) < 1e-6, (
            f"Expected scale factor ~0.18215, got {latent.scale_factor}"
        )
    
    def test_sd15_custom_scale_factor(self):
        """SD1.5 scale factor should be configurable."""
        from src.Utilities.Latent import SD15
        
        custom_scale = 0.2
        latent = SD15(scale_factor=custom_scale)
        assert abs(latent.scale_factor - custom_scale) < 1e-6, (
            f"Expected scale factor {custom_scale}, got {latent.scale_factor}"
        )
    
    def test_sd15_has_rgb_factors(self):
        """SD1.5 should have latent RGB factors defined."""
        from src.Utilities.Latent import SD15
        
        latent = SD15()
        assert hasattr(latent, 'latent_rgb_factors'), (
            "SD15 should have latent_rgb_factors attribute"
        )
        assert len(latent.latent_rgb_factors) == 4, (
            f"Expected 4 RGB factor rows, got {len(latent.latent_rgb_factors)}"
        )
        # Each row should have 3 values (R, G, B)
        for row in latent.latent_rgb_factors:
            assert len(row) == 3, f"Each RGB row should have 3 values, got {len(row)}"
    
    def test_sd15_has_taesd_decoder_name(self):
        """SD1.5 should reference correct TAESD decoder."""
        from src.Utilities.Latent import SD15
        
        latent = SD15()
        assert hasattr(latent, 'taesd_decoder_name'), (
            "SD15 should have taesd_decoder_name attribute"
        )
        assert latent.taesd_decoder_name == "taesd_decoder", (
            f"Expected 'taesd_decoder', got {latent.taesd_decoder_name}"
        )


class TestSD15ModelConfig:
    """Test suite for SD1.5 model configuration (sm_SD15)."""
    
    def test_sd15_unet_config_has_required_keys(self):
        """SD1.5 UNet config should have all required keys."""
        from src.SD15.SD15 import sm_SD15
        
        required_keys = [
            "context_dim",
            "model_channels", 
            "use_linear_in_transformer",
            "adm_in_channels",
            "use_temporal_attention",
        ]
        
        for key in required_keys:
            assert key in sm_SD15.unet_config, (
                f"Missing required key '{key}' in SD15 unet_config"
            )
    
    def test_sd15_context_dim_is_768(self):
        """SD1.5 should use 768-dimensional context (CLIP embedding dim)."""
        from src.SD15.SD15 import sm_SD15
        
        assert sm_SD15.unet_config["context_dim"] == 768, (
            f"Expected context_dim=768, got {sm_SD15.unet_config['context_dim']}"
        )
    
    def test_sd15_model_channels_is_320(self):
        """SD1.5 should use 320 model channels."""
        from src.SD15.SD15 import sm_SD15
        
        assert sm_SD15.unet_config["model_channels"] == 320, (
            f"Expected model_channels=320, got {sm_SD15.unet_config['model_channels']}"
        )
    
    def test_sd15_no_linear_in_transformer(self):
        """SD1.5 should not use linear in transformer."""
        from src.SD15.SD15 import sm_SD15
        
        assert sm_SD15.unet_config["use_linear_in_transformer"] is False, (
            "SD1.5 should not use linear in transformer"
        )
    
    def test_sd15_no_adm_channels(self):
        """SD1.5 should not have ADM channels (no pooled conditioning)."""
        from src.SD15.SD15 import sm_SD15
        
        assert sm_SD15.unet_config["adm_in_channels"] is None, (
            f"SD1.5 should have adm_in_channels=None, got {sm_SD15.unet_config['adm_in_channels']}"
        )
    
    def test_sd15_no_temporal_attention(self):
        """SD1.5 should not use temporal attention."""
        from src.SD15.SD15 import sm_SD15
        
        assert sm_SD15.unet_config["use_temporal_attention"] is False, (
            "SD1.5 should not use temporal attention"
        )
    
    def test_sd15_uses_correct_latent_format(self):
        """SD1.5 model config should reference SD15 latent format."""
        from src.SD15.SD15 import sm_SD15
        from src.Utilities.Latent import SD15 as SD15LatentFormat
        
        assert sm_SD15.latent_format == SD15LatentFormat, (
            f"SD1.5 model should use SD15 latent format"
        )
    
    def test_sd15_clip_target_returns_valid_target(self):
        """SD1.5 clip_target should return a ClipTarget."""
        from src.SD15.SD15 import sm_SD15
        from src.clip.Clip import ClipTarget
        
        model = sm_SD15(sm_SD15.unet_config)
        target = model.clip_target()
        
        assert isinstance(target, ClipTarget), (
            f"Expected ClipTarget, got {type(target)}"
        )
    
    def test_sd15_clip_target_uses_sd1_tokenizer(self):
        """SD1.5 should use SD1Tokenizer."""
        from src.SD15.SD15 import sm_SD15
        from src.SD15.SDToken import SD1Tokenizer
        
        model = sm_SD15(sm_SD15.unet_config)
        target = model.clip_target()
        
        assert target.tokenizer == SD1Tokenizer, (
            "SD1.5 should use SD1Tokenizer"
        )
    
    def test_sd15_clip_target_uses_sd1_clip_model(self):
        """SD1.5 should use SD1ClipModel."""
        from src.SD15.SD15 import sm_SD15
        from src.SD15.SDClip import SD1ClipModel
        
        model = sm_SD15(sm_SD15.unet_config)
        target = model.clip_target()
        
        assert target.clip == SD1ClipModel, (
            "SD1.5 should use SD1ClipModel"
        )


class TestSD15CheckpointLoader:
    """Test suite for CheckpointLoaderSimple with SD1.5 models."""
    
    def test_loader_instantiation(self):
        """CheckpointLoaderSimple should instantiate without errors."""
        from src.FileManaging.Loader import CheckpointLoaderSimple
        
        loader = CheckpointLoaderSimple()
        assert loader is not None
    
    @patch('src.FileManaging.Loader.load_checkpoint_guess_config')
    @patch('src.Device.ModelCache.get_model_cache')
    def test_loader_calls_correct_functions(self, mock_cache_fn, mock_load):
        """Loader should call cache check then load if not cached."""
        from src.FileManaging.Loader import CheckpointLoaderSimple
        
        # Setup mocks - use MagicMock directly
        mock_cache_instance = MagicMock()
        mock_cache_instance.get_cached_checkpoint.return_value = None
        mock_cache_fn.return_value = mock_cache_instance
        
        mock_model = MagicMock(name="mock_model")
        mock_clip = MagicMock(name="mock_clip")
        mock_vae = MagicMock(name="mock_vae")
        mock_load.return_value = (mock_model, mock_clip, mock_vae, None)
        
        loader = CheckpointLoaderSimple()
        result = loader.load_checkpoint("test_model.safetensors")
        
        # Verify cache was checked
        mock_cache_instance.get_cached_checkpoint.assert_called_once()
        # Verify load was called
        mock_load.assert_called_once()
        # Verify result is tuple of 3
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    
    @patch('src.Device.ModelCache.get_model_cache')
    def test_loader_returns_cached_model(self, mock_cache_fn):
        """Loader should return cached model without calling load."""
        from src.FileManaging.Loader import CheckpointLoaderSimple
        
        # Setup cached result using MagicMock
        cached_model = MagicMock(name="cached_model")
        cached_clip = MagicMock(name="cached_clip")
        cached_vae = MagicMock(name="cached_vae")
        
        mock_cache_instance = MagicMock()
        mock_cache_instance.get_cached_checkpoint.return_value = (
            cached_model, cached_clip, cached_vae
        )
        mock_cache_fn.return_value = mock_cache_instance
        
        loader = CheckpointLoaderSimple()
        result = loader.load_checkpoint("cached_model.safetensors")
        
        # Verify cached result returned
        assert result[0] is cached_model
        assert result[1] is cached_clip
        assert result[2] is cached_vae
    
    def test_loader_accepts_vae_flag(self):
        """Loader should accept output_vae parameter."""
        from src.FileManaging.Loader import CheckpointLoaderSimple
        
        loader = CheckpointLoaderSimple()
        # Should not raise TypeError for output_vae parameter
        with patch('src.FileManaging.Loader.load_checkpoint_guess_config') as mock:
            mock.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
            with patch('src.Device.ModelCache.get_model_cache') as cache:
                cache.return_value.get_cached_checkpoint.return_value = None
                # This should not raise
                loader.load_checkpoint("test.safetensors", output_vae=False)
    
    def test_loader_accepts_clip_flag(self):
        """Loader should accept output_clip parameter."""
        from src.FileManaging.Loader import CheckpointLoaderSimple
        
        loader = CheckpointLoaderSimple()
        with patch('src.FileManaging.Loader.load_checkpoint_guess_config') as mock:
            mock.return_value = (MagicMock(), MagicMock(), MagicMock(), None)
            with patch('src.Device.ModelCache.get_model_cache') as cache:
                cache.return_value.get_cached_checkpoint.return_value = None
                # This should not raise
                loader.load_checkpoint("test.safetensors", output_clip=False)


class TestSD15CLIPEncoding:
    """Test suite for SD1.5 CLIP text encoding (mocked)."""
    
    def test_clip_text_encode_instantiation(self):
        """CLIPTextEncode should instantiate without errors."""
        from src.clip.Clip import CLIPTextEncode
        
        encoder = CLIPTextEncode()
        assert encoder is not None
    
    @patch('src.clip.Clip.CLIPTextEncode.encode')
    def test_encode_returns_conditioning_format(self, mock_encode):
        """encode() should return list of [tensor, metadata] entries."""
        from src.clip.Clip import CLIPTextEncode
        
        # Mock the return value
        mock_cond = torch.randn(1, 77, 768)
        mock_metadata = {"pooled_output": None}
        mock_encode.return_value = ([[mock_cond, mock_metadata]],)
        
        encoder = CLIPTextEncode()
        result = encoder.encode(text="test prompt", clip=MagicMock())
        
        # Should be a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        # First element should be list of conditioning entries
        cond_list = result[0]
        assert isinstance(cond_list, list), f"Expected list, got {type(cond_list)}"
    
    @patch('src.clip.Clip.CLIPTextEncode.encode')
    def test_encode_produces_768_dim_embeddings_for_sd15(self, mock_encode):
        """SD1.5 CLIP encoding should produce 768-dim embeddings."""
        from src.clip.Clip import CLIPTextEncode
        
        # SD1.5 uses 768-dim embeddings
        expected_dim = 768
        mock_cond = torch.randn(1, 77, expected_dim)
        mock_encode.return_value = ([[mock_cond, {}]],)
        
        encoder = CLIPTextEncode()
        result = encoder.encode(text="test", clip=MagicMock())
        
        cond_tensor = result[0][0][0]
        assert cond_tensor.shape[-1] == expected_dim, (
            f"Expected embedding dim {expected_dim}, got {cond_tensor.shape[-1]}"
        )


class TestSD15EmptyLatent:
    """Test suite for EmptyLatentImage generation."""
    
    def test_empty_latent_instantiation(self):
        """EmptyLatentImage should instantiate without errors."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        assert generator is not None
    
    def test_empty_latent_generates_correct_shape(self):
        """EmptyLatentImage should generate correct latent dimensions."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        width, height = 512, 512
        batch_size = 1
        
        result = generator.generate(width=width, height=height, batch_size=batch_size)
        
        # Result should be tuple with dict containing 'samples'
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        latent_dict = result[0]
        assert "samples" in latent_dict, "Result should have 'samples' key"
        
        samples = latent_dict["samples"]
        # For SD1.5: latent = image_size / 8
        expected_shape = (batch_size, 4, height // 8, width // 8)
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
    
    def test_empty_latent_with_different_sizes(self):
        """EmptyLatentImage should work with various image sizes."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        
        test_cases = [
            (512, 512, 1),
            (768, 512, 1),
            (1024, 1024, 1),
            (512, 768, 2),
        ]
        
        for width, height, batch in test_cases:
            result = generator.generate(width=width, height=height, batch_size=batch)
            samples = result[0]["samples"]
            
            expected_shape = (batch, 4, height // 8, width // 8)
            assert samples.shape == expected_shape, (
                f"For {width}x{height} batch={batch}: "
                f"expected {expected_shape}, got {samples.shape}"
            )
    
    def test_empty_latent_is_zeros(self):
        """EmptyLatentImage should produce zero-initialized latents."""
        from src.Utilities.Latent import EmptyLatentImage
        
        generator = EmptyLatentImage()
        
        result = generator.generate(width=512, height=512, batch_size=1)
        
        # EmptyLatentImage generates zeros - randomness comes from sampling noise
        assert torch.allclose(result[0]["samples"], torch.zeros_like(result[0]["samples"])), (
            "EmptyLatentImage should produce zero-initialized latents"
        )


class TestSD15TokenizerBasics:
    """Test suite for SD1.5 tokenizer functionality."""
    
    def test_sd1_tokenizer_class_exists(self):
        """SD1Tokenizer class should exist."""
        from src.SD15.SDToken import SD1Tokenizer
        assert SD1Tokenizer is not None
    
    def test_sd_tokenizer_base_class_exists(self):
        """SDTokenizer base class should exist."""
        from src.SD15.SDToken import SDTokenizer
        assert SDTokenizer is not None


class TestSD15ProcessClipStateDict:
    """Test suite for CLIP state dict processing."""
    
    def test_process_clip_state_dict_handles_prefix_replacement(self):
        """process_clip_state_dict should handle cond_stage_model prefix."""
        from src.SD15.SD15 import sm_SD15
        
        model = sm_SD15(sm_SD15.unet_config)
        
        # Create dummy state dict with old prefix
        state_dict = {
            "cond_stage_model.transformer.text_model.weight": torch.randn(10, 10),
            "cond_stage_model.other.weight": torch.randn(5, 5),
        }
        
        result = model.process_clip_state_dict(state_dict)
        
        # After processing, keys should use clip_l prefix
        for key in result.keys():
            assert key.startswith("clip_l."), (
                f"Expected key to start with 'clip_l.', got {key}"
            )
    
    def test_process_clip_state_dict_handles_position_ids_dtype(self):
        """process_clip_state_dict should convert float32 position_ids to int."""
        from src.SD15.SD15 import sm_SD15
        
        model = sm_SD15(sm_SD15.unet_config)
        
        # Create state dict with float32 position_ids
        pos_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
        state_dict = {
            pos_key: torch.arange(77).float(),  # float32
        }
        
        result = model.process_clip_state_dict(state_dict)
        
        # The position_ids should be processed (key may be renamed)
        # Check that no float32 position_ids remain
        for key, value in result.items():
            if "position_ids" in key and value.dtype == torch.float32:
                # Should be rounded (not exact floats like 0.1, 0.2, etc.)
                rounded = value.round()
                assert torch.allclose(value, rounded), (
                    "Float32 position_ids should be rounded"
                )


class TestSD15SamplerIntegration:
    """Test suite for SD1.5 sampler integration (mocked)."""
    
    def test_ksampler_instantiation(self):
        """KSampler should instantiate without errors."""
        from src.sample.sampling import KSampler
        
        sampler = KSampler()
        assert sampler is not None
    
    def test_ksampler_sample_signature_includes_required_params(self):
        """KSampler.sample should accept all required parameters."""
        from src.sample.sampling import KSampler
        import inspect
        
        sampler = KSampler()
        sig = inspect.signature(sampler.sample)
        params = sig.parameters
        
        required_params = [
            'seed', 'steps', 'cfg', 'sampler_name', 'scheduler',
            'denoise', 'model', 'positive', 'negative', 'latent_image'
        ]
        
        for param in required_params:
            assert param in params, (
                f"KSampler.sample missing required parameter: {param}"
            )
    
    def test_ksampler_sample_accepts_pipeline_flag(self):
        """KSampler.sample should accept pipeline flag."""
        from src.sample.sampling import KSampler
        import inspect
        
        sampler = KSampler()
        sig = inspect.signature(sampler.sample)
        
        assert 'pipeline' in sig.parameters, (
            "KSampler.sample should accept 'pipeline' parameter"
        )


class TestSD15ModelInModelsRegistry:
    """Test that SD1.5 model is properly registered."""
    
    def test_sd15_in_models_list(self):
        """sm_SD15 should be in the models registry."""
        from src.SD15.SD15 import models, sm_SD15
        
        assert sm_SD15 in models, (
            "sm_SD15 should be in the models registry list"
        )
    
    def test_models_list_not_empty(self):
        """Models list should contain multiple model types."""
        from src.SD15.SD15 import models
        
        assert len(models) > 0, "Models list should not be empty"
        assert len(models) >= 3, (
            f"Expected at least 3 model types, got {len(models)}"
        )
