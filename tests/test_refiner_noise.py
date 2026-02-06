"""Verification script for SDXL refiner noise fix.

This script verifies that the Pipeline correctly sets disable_noise=True
when calling the refiner model's generate method.
"""

import sys
import unittest
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.Core import Context, Pipeline
from src.Core.AbstractModel import AbstractModel

@pytest.mark.unit
@pytest.mark.slow
class TestRefinerNoiseFix(unittest.TestCase):
    
    @patch('src.Core.Pipeline.create_model')
    @patch('src.FileManaging.ImageSaver.SaveImage')
    def test_refiner_disables_noise(self, mock_saver, mock_create_model):
        """Verify that refiner pass calls generate with disable_noise=True."""
        
        # 1. Setup mocks
        mock_base = MagicMock(spec=AbstractModel)
        mock_base.is_loaded = True
        mock_base.model_path = "base.safetensors"
        mock_base.capabilities.supports_lora = False
        mock_base.generate.return_value = {"samples": "base_latents"}
        mock_base.encode_prompt.return_value = ("pos", "neg")
        mock_base.decode.return_value = ["base_image"]
        
        mock_refiner = MagicMock(spec=AbstractModel)
        mock_refiner.is_loaded = True
        mock_refiner.model_path = "refiner.safetensors"
        mock_refiner.capabilities.supports_lora = False
        mock_refiner.generate.return_value = {"samples": "refined_latents"}
        mock_refiner.encode_prompt.return_value = ("ref_pos", "ref_neg")
        mock_refiner.decode.return_value = ["refined_image"]
        
        # model_factory returns base first, then refiner
        mock_create_model.side_effect = [mock_base, mock_refiner]
        
        # 2. Create context with refiner
        ctx = Context.from_kwargs(
            prompt="test",
            w=1024,
            h=1024,
            steps=20,
            refiner_model_path="refiner.safetensors",
            refiner_switch_step=16
        )
        
        # 3. Run pipeline
        pipeline = Pipeline(model_factory=mock_create_model)
        with patch('src.Core.Pipeline.logger'): # Silence logs
            pipeline.run(ctx)
        
        # 4. Verify base generate call: disable_noise should be default (False)
        # Find the call to base model's generate
        base_call = mock_base.generate.call_args_list[0]
        self.assertEqual(base_call.kwargs.get('last_step'), 16)
        # disable_noise is not passed by run() for base, so it should be missing or default
        self.assertIsNone(base_call.kwargs.get('disable_noise'))
        
        # 5. Verify refiner generate call: disable_noise MUST be True
        refiner_call = mock_refiner.generate.call_args_list[0]
        self.assertEqual(refiner_call.kwargs.get('start_step'), 16)
        self.assertTrue(refiner_call.kwargs.get('disable_noise'), "Refiner must be called with disable_noise=True")
        
        print("SUCCESS: Refiner correctly called with disable_noise=True")

if __name__ == "__main__":
    unittest.main()
