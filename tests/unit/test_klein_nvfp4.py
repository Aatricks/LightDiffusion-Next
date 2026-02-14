import torch
import unittest
from src.clip.KleinEncoder import Qwen3_4BModel, KleinCLIP, KleinTokenizer
from src.Model.ModelPatcher import ModelPatcher
from src.Utilities.Quantization import dequantize_nvfp4

class TestKleinNVFP4(unittest.TestCase):
    def test_text_encoder_nvfp4_quantization(self):
        # Create a small version of the model for testing
        from src.cond.cast import manual_cast as ops
        device = torch.device("cpu")
        dtype = torch.bfloat16
        
        # Use our custom Linear that supports comfy_cast_weights
        model = torch.nn.Sequential(
            ops.Linear(128, 128, bias=False, dtype=dtype, device=device)
        )
        
        # Fill with some identifiable data
        with torch.no_grad():
            model[0].weight.copy_(torch.randn(128, 128, dtype=dtype))
            orig_weight = model[0].weight.clone()

        # Wrap in ModelPatcher and quantize to nvfp4
        patcher = ModelPatcher(model, device, device)
        patcher.weight_only_quantize("nvfp4")
        
        # Verify quantization attributes
        self.assertTrue(patcher.model[0].comfy_cast_weights)
        self.assertEqual(patcher.model[0].quant_format, "nvfp4")
        self.assertTrue(hasattr(patcher.model[0], "weight_scale"))
        self.assertTrue(hasattr(patcher.model[0], "weight_scale_2"))
        
        # Verify weight is now packed uint8
        self.assertEqual(patcher.model[0].weight.dtype, torch.uint8)
        self.assertEqual(patcher.model[0].weight.shape, (128, 64)) # Packed 4-bit
        
        # Test forward pass (triggers dequantization)
        input_tensor = torch.randn(1, 128, dtype=dtype, device=device)
        output = patcher.model(input_tensor)
        
        self.assertEqual(output.shape, (1, 128))
        self.assertEqual(output.dtype, dtype)
        self.assertFalse(torch.isnan(output).any())
        
        print("NVFP4 Text Encoder layer test passed!")

if __name__ == "__main__":
    unittest.main()
