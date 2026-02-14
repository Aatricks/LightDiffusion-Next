
import torch
import pytest
from unittest.mock import MagicMock
from src.NeuralNetwork.transformer import BasicTransformerBlock
from src.Model.ModelPatcher import ModelPatcher

def test_tome_forward_signature():
    # Create the block
    block = BasicTransformerBlock(dim=64, n_heads=1, d_head=64)
    
    # Setup a mock model with the necessary structure for tomesd
    class MockDiffusionModel(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.transformer_block = block
            self.dtype = torch.float32
    
    class MockModel(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.diffusion_model = MockDiffusionModel(block)
    
    mock_model = MockModel(block)
    
    # Use ModelPatcher to apply tome
    # ModelPatcher expects the model to have certain attributes
    patcher = ModelPatcher(mock_model, torch.device("cpu"), torch.device("cpu"))
    
    try:
        import tomesd
    except ImportError:
        pytest.skip("tomesd not installed")

    success = patcher.apply_tome(ratio=0.5)
    assert success, "Failed to apply ToMe"
    
    # The block's class should now be ToMeBlock
    assert block.__class__.__name__ == "ToMeBlock"
    
    # Now try to call the block
    x = torch.randn(1, 16, 64)
    transformer_options = {"some_option": True}
    
    # This should NOT raise TypeError: ToMeBlock._forward() takes from 2 to 3 positional arguments but 4 were given
    # Even if it fails later due to mock issues, the TypeError should be gone.
    try:
        # We need to mock compute_merge or ensure it has what it needs
        # tomesd.patch.hook_tome_model was called by apply_patch, which added _tome_info to diffusion_model
        # and ToMeBlock._forward uses self._tome_info (which is a reference to diffusion_model._tome_info)
        
        # Ensure _tome_info["size"] is set (normally set by hook)
        mock_model.diffusion_model._tome_info["size"] = (64, 64)
        
        block.forward(x, transformer_options=transformer_options)
    except TypeError as e:
        if "ToMeBlock._forward() takes" in str(e):
            pytest.fail(f"ToMe fix failed: {e}")
        # If it's another TypeError, it might be expected due to mocks
        raise e
    except Exception as e:
        # We don't care about other errors (like shape mismatches in mocks) 
        # as long as the signature mismatch is fixed.
        print(f"Caught expected non-TypeError: {e}")

if __name__ == "__main__":
    test_tome_forward_signature()
