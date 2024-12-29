import torch
import torch.nn as nn
from modules.cond import cast

class GEGLU(nn.Module):
    """#### Class representing the GEGLU activation function.
    
    GEGLU is a gated activation function that is a combination of GELU and ReLU,
    used to fire the neurons in the network.

    #### Args:
        - `dim_in` (int): The input dimension.
        - `dim_out` (int): The output dimension.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = cast.manual_cast.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the GEGLU activation function.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)