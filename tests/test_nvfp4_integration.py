import torch
import pytest
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.Model.ModelPatcher import ModelPatcher
from src.cond.cast import disable_weight_init

logging.basicConfig(level=logging.INFO)

@pytest.mark.slow
def test_nvfp4_integration():
    print("Testing NVFP4 Integration...")
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = disable_weight_init.Linear(512, 128, bias=True)
            self.linear.weight.data.normal_(0, 0.1)
            self.linear.bias.data.zero_()
            
        def forward(self, x):
            return self.linear(x)
            
    model = SimpleModel()
    
    # Reference output (FP32)
    input_data = torch.randn(1, 512)
    reference_output = model(input_data)
    
    # Patch and Quantize to NVFP4
    load_device = torch.device("cpu")
    offload_device = torch.device("cpu")
    patcher = ModelPatcher(model, load_device, offload_device)
    
    print("Quantizing to NVFP4...")
    patcher.weight_only_quantize("nvfp4")
    
    # Run forward pass with quantized weights
    print("Running forward pass with NVFP4 weights...")
    quantized_output = model(input_data)
    
    # Calculate error
    mse = torch.mean((reference_output - quantized_output) ** 2).item()
    print(f"Integration MSE: {mse:.8f}")
    
    if mse < 0.2:
        print("SUCCESS: NVFP4 integration test passed!")
    else:
        print("FAILURE: Integration error too high!")

if __name__ == "__main__":
    test_nvfp4_integration()
