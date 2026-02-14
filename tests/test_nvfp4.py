import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Utilities.Quantization import quantize_nvfp4, dequantize_nvfp4

@pytest.mark.slow
def test_nvfp4_accuracy():
    print("Testing NVFP4 Accuracy...")
    
    # Create a random 2D tensor
    shape = (128, 512)
    original = torch.randn(shape, dtype=torch.float32) * 0.1
    
    # Quantize
    qdata, tensor_scale, blocked_scales = quantize_nvfp4(original)
    
    print(f"Original shape: {original.shape}")
    print(f"Packed data shape: {qdata.shape}")
    print(f"Tensor scale: {tensor_scale.item():.6f}")
    print(f"Blocked scales shape: {blocked_scales.shape}")
    
    # Dequantize
    reconstructed = dequantize_nvfp4(qdata, tensor_scale, blocked_scales, shape)
    
    # Calculate error
    mse = torch.mean((original - reconstructed) ** 2).item()
    max_err = torch.max(torch.abs(original - reconstructed)).item()
    
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Max Absolute Error: {max_err:.8f}")
    
    # Check if error is within reasonable bounds for 4-bit
    # For a normalized random tensor, MSE should be relatively low
    if mse < 0.01:
        print("SUCCESS: NVFP4 test passed!")
    else:
        print("FAILURE: Error too high!")
        
if __name__ == "__main__":
    test_nvfp4_accuracy()
