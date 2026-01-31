"""
Shared pytest fixtures and mock utilities for LightDiffusion test suite.

This module provides:
- Mock checkpoint loaders that don't load real model weights
- Mock model patchers and CLIP models
- Common utilities for testing without GPU dependencies
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Tuple, Dict, Any, Optional, List

import pytest
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Mock Model Classes
# =============================================================================

class MockModelPatcher:
    """Mock model patcher that simulates model loading without actual weights.
    
    This provides the same interface as the real ModelPatcher but doesn't
    load any actual model data, making tests fast and memory-efficient.
    """
    
    def __init__(self, model_name: str = "mock_model", model_type: str = "SD15"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = MagicMock()
        self.model.diffusion_model = MagicMock()
        self.patches = {}
        self.object_patches = {}
        self.weight_inplace_update = False
        self.load_device = torch.device("cpu")
        self.current_device = torch.device("cpu")
        self.model_options = {}
    
    def clone(self):
        """Return a clone of this patcher."""
        cloned = MockModelPatcher(self.model_name, self.model_type)
        cloned.patches = self.patches.copy()
        cloned.object_patches = self.object_patches.copy()
        return cloned
    
    def add_patches(self, patches: Dict, strength: float = 1.0):
        """Add patches (LoRA weights, etc.)."""
        self.patches.update(patches)
    
    def get_model_object(self, name: str):
        """Get a model object by name."""
        return getattr(self.model, name, MagicMock())
    
    def set_model_option(self, key: str, value: Any):
        """Set a model option."""
        self.model_options[key] = value
    
    def apply_tome(self, ratio: float = 0.5, max_downsample: int = 1) -> bool:
        """Mock ToMe application."""
        return True
    
    def remove_tome(self):
        """Mock ToMe removal."""
        pass


class MockCLIP:
    """Mock CLIP model for testing text encoding without loading real weights."""
    
    def __init__(self, clip_type: str = "SD15"):
        self.clip_type = clip_type
        self.cond_stage_model = MagicMock()
        self.tokenizer = MagicMock()
        self.layer_idx = -2
        self.patcher = MagicMock()
    
    def encode(self, text: str) -> Tuple[torch.Tensor, Dict]:
        """Mock encode that returns fake embeddings."""
        # Return fake conditioning tensor (batch, seq_len, embed_dim)
        if self.clip_type == "SDXL":
            embed_dim = 2048  # SDXL uses concatenated L+G (768+1280)
        else:
            embed_dim = 768  # SD1.5
        
        cond = torch.randn(1, 77, embed_dim)
        pooled = torch.randn(1, embed_dim) if self.clip_type == "SDXL" else None
        return cond, {"pooled_output": pooled}
    
    def tokenize(self, text: str) -> Dict:
        """Mock tokenize."""
        return {"input_ids": torch.randint(0, 49407, (1, 77))}
    
    def clone(self):
        """Clone the CLIP model."""
        return MockCLIP(self.clip_type)


class MockVAE:
    """Mock VAE for testing encode/decode without real model weights."""
    
    def __init__(self, latent_channels: int = 4):
        self.latent_channels = latent_channels
        self.first_stage_model = MagicMock()
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        batch, channels, height, width = images.shape
        latent_h = height // 8
        latent_w = width // 8
        return torch.randn(batch, self.latent_channels, latent_h, latent_w)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        batch, channels, latent_h, latent_w = latents.shape
        height = latent_h * 8
        width = latent_w * 8
        return torch.randn(batch, 3, height, width)


class MockCheckpointResult:
    """Container for mock checkpoint loading results."""
    
    def __init__(self, model_type: str = "SD15"):
        self.model_patcher = MockModelPatcher("mock_checkpoint", model_type)
        self.clip = MockCLIP(model_type)
        self.vae = MockVAE()
    
    def as_tuple(self) -> Tuple:
        """Return as tuple matching CheckpointLoaderSimple output."""
        return (self.model_patcher, self.clip, self.vae)


# =============================================================================
# Mock Loader Classes
# =============================================================================

class MockCheckpointLoaderSimple:
    """Mock checkpoint loader that doesn't load real model files.
    
    Use this when you want to test code that calls CheckpointLoaderSimple
    without actually loading 6GB model files.
    """
    
    def __init__(self):
        self.loaded_checkpoints = []
    
    def load_checkpoint(
        self, 
        ckpt_name: str, 
        output_vae: bool = True, 
        output_clip: bool = True
    ) -> Tuple:
        """Load a mock checkpoint.
        
        Args:
            ckpt_name: Path/name of checkpoint (used to detect model type)
            output_vae: Whether to return VAE
            output_clip: Whether to return CLIP
            
        Returns:
            Tuple of (model_patcher, clip, vae)
        """
        self.loaded_checkpoints.append(ckpt_name)
        
        # Detect model type from filename
        ckpt_lower = ckpt_name.lower()
        if "sdxl" in ckpt_lower or "xl" in ckpt_lower:
            model_type = "SDXL"
        elif "flux" in ckpt_lower:
            model_type = "FLUX"
        else:
            model_type = "SD15"
        
        result = MockCheckpointResult(model_type)
        return result.as_tuple()


class MockUnetLoaderGGUF:
    """Mock GGUF UNet loader for Flux models."""
    
    def __init__(self):
        self.loaded_models = []
    
    def load_unet(
        self,
        unet_name: str,
        dequant_dtype: Optional[str] = None,
        patch_dtype: Optional[str] = None
    ) -> Tuple:
        """Load a mock GGUF UNet."""
        self.loaded_models.append(unet_name)
        return (MockModelPatcher(unet_name, "FLUX"),)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_checkpoint_loader():
    """Provide a MockCheckpointLoaderSimple instance."""
    return MockCheckpointLoaderSimple()


@pytest.fixture
def mock_model_patcher():
    """Provide a MockModelPatcher instance."""
    return MockModelPatcher()


@pytest.fixture
def mock_sd15_checkpoint():
    """Provide mock SD1.5 checkpoint result."""
    return MockCheckpointResult("SD15")


@pytest.fixture
def mock_sdxl_checkpoint():
    """Provide mock SDXL checkpoint result."""
    return MockCheckpointResult("SDXL")


@pytest.fixture
def mock_flux_checkpoint():
    """Provide mock Flux checkpoint result."""
    return MockCheckpointResult("FLUX")


@pytest.fixture
def mock_clip_sd15():
    """Provide mock SD1.5 CLIP model."""
    return MockCLIP("SD15")


@pytest.fixture
def mock_clip_sdxl():
    """Provide mock SDXL CLIP model."""
    return MockCLIP("SDXL")


@pytest.fixture
def mock_vae():
    """Provide mock VAE model."""
    return MockVAE()


@pytest.fixture
def sample_latent_4ch():
    """Provide sample 4-channel latent tensor (SD1.5/SDXL)."""
    return {"samples": torch.randn(1, 4, 64, 64)}


@pytest.fixture
def sample_latent_16ch():
    """Provide sample 16-channel latent tensor (Flux)."""
    return {"samples": torch.randn(1, 16, 64, 64)}


@pytest.fixture
def sample_image_tensor():
    """Provide sample image tensor (B, H, W, C) normalized 0-1."""
    return torch.rand(1, 512, 512, 3)


@pytest.fixture
def patch_checkpoint_loader():
    """Context manager to patch CheckpointLoaderSimple globally."""
    with patch(
        "src.FileManaging.Loader.CheckpointLoaderSimple",
        MockCheckpointLoaderSimple
    ) as mock:
        yield mock


@pytest.fixture
def patch_model_loader():
    """Patch load_model_for_pipeline to return mock results."""
    def mock_load(model_path=None, flux_dequant_dtype=None, flux_patch_dtype=None):
        if model_path and "flux" in model_path.lower():
            return ("FLUX", (MockModelPatcher(model_path, "FLUX"),))
        elif model_path and "sdxl" in model_path.lower():
            return ("SDXL", MockCheckpointResult("SDXL").as_tuple())
        else:
            return ("SD15", MockCheckpointResult("SD15").as_tuple())
    
    with patch(
        "src.user.model_loader.load_model_for_pipeline",
        side_effect=mock_load
    ) as mock:
        yield mock


@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary mock model file path."""
    model_file = tmp_path / "test_model.safetensors"
    model_file.touch()  # Create empty file
    return str(model_file)


# =============================================================================
# Utility Functions
# =============================================================================

def create_mock_conditioning(
    batch_size: int = 1,
    seq_len: int = 77,
    embed_dim: int = 768,
    model_type: str = "SD15"
) -> List:
    """Create mock conditioning entries matching pipeline format.
    
    Args:
        batch_size: Number of conditions
        seq_len: Sequence length
        embed_dim: Embedding dimension
        model_type: "SD15" (768), "SDXL" (2048), or "FLUX"
        
    Returns:
        List of [tensor, metadata_dict] entries
    """
    if model_type == "SDXL":
        embed_dim = 2048
    elif model_type == "FLUX":
        embed_dim = 4096
    
    entries = []
    for i in range(batch_size):
        cond_tensor = torch.randn(1, seq_len, embed_dim)
        meta = {"batch_index": [i]}
        if model_type == "SDXL":
            meta["pooled_output"] = torch.randn(1, 1280)
        entries.append([cond_tensor, meta])
    return entries


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple):
    """Assert tensor has expected shape with informative error message."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_valid_latent(latent_dict: Dict, expected_channels: int = 4):
    """Assert latent dictionary is valid."""
    assert "samples" in latent_dict, "Latent dict must have 'samples' key"
    samples = latent_dict["samples"]
    assert samples.ndim == 4, f"Latent must be 4D, got {samples.ndim}D"
    assert samples.shape[1] == expected_channels, (
        f"Expected {expected_channels} channels, got {samples.shape[1]}"
    )


# =============================================================================
# Test Path Helpers
# =============================================================================

def get_test_data_path(relative_path: str) -> Path:
    """Get absolute path to test data file."""
    return project_root / "tests" / "data" / relative_path


def get_checkpoint_path(model_name: str) -> str:
    """Get path to checkpoint (returns mock path for testing)."""
    return str(project_root / "include" / "checkpoints" / model_name)


# Create test data directory if needed
(project_root / "tests" / "data").mkdir(parents=True, exist_ok=True)
