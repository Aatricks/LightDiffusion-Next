from modules.Device import Device
import torch

def cast_bias_weight(s: torch.nn.Module, input: torch.Tensor) -> tuple:
    """#### Cast the bias and weight of a module to match the input tensor.

    #### Args:
        - `s` (torch.nn.Module): The module.
        - `input` (torch.Tensor): The input tensor.

    #### Returns:
        - `tuple`: The cast weight and bias.
    """
    bias = None
    non_blocking = Device.device_supports_non_blocking(input.device)
    if s.bias is not None:
        bias = s.bias.to(
            device=input.device, dtype=input.dtype, non_blocking=non_blocking
        )
    weight = s.weight.to(
        device=input.device, dtype=input.dtype, non_blocking=non_blocking
    )
    return weight, bias


class CastWeightBiasOp:
    """#### Class representing a cast weight and bias operation."""
    comfy_cast_weights: bool = False
    weight_function: callable = None
    bias_function: callable = None


class disable_weight_init:
    """#### Class representing a module with disabled weight initialization."""
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        """#### Linear layer with disabled weight initialization."""
        def reset_parameters(self) -> None:
            """#### Reset the parameters of the linear layer."""
            return None

        def forward_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """#### Forward pass with comfy cast weights.

            #### Args:
                - `input` (torch.Tensor): The input tensor.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the linear layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.comfy_cast_weights:
                return self.forward_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        """#### Conv2d layer with disabled weight initialization."""
        def reset_parameters(self) -> None:
            """#### Reset the parameters of the Conv2d layer."""
            return None

        def forward_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """#### Forward pass with comfy cast weights.

            #### Args:
                - `input` (torch.Tensor): The input tensor.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the Conv2d layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.comfy_cast_weights:
                return self.forward_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        """#### GroupNorm layer with disabled weight initialization."""
        def reset_parameters(self) -> None:
            """#### Reset the parameters of the GroupNorm layer."""
            return None

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the GroupNorm layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        """#### LayerNorm layer with disabled weight initialization."""
        def reset_parameters(self) -> None:
            """#### Reset the parameters of the LayerNorm layer."""
            return None

        def forward_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """#### Forward pass with cast weights.

            #### Args:
                - `input` (torch.Tensor): The input tensor.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs) -> torch.Tensor:
            """#### Forward pass for the LayerNorm layer.

            #### Args:
                - `*args`: Variable length argument list.
                - `**kwargs`: Arbitrary keyword arguments.

            #### Returns:
                - `torch.Tensor`: The output tensor.
            """
            if self.comfy_cast_weights:
                return self.forward_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(cls, dims: int, *args, **kwargs) -> torch.nn.Conv2d:
        """#### Create a Conv2d layer with the specified dimensions.

        #### Args:
            - `dims` (int): The number of dimensions.
            - `*args`: Variable length argument list.
            - `**kwargs`: Arbitrary keyword arguments.

        #### Returns:
            - `torch.nn.Conv2d`: The Conv2d layer.
        """
        return cls.Conv2d(*args, **kwargs)


class manual_cast(disable_weight_init):
    """#### Class representing a module with manual casting."""
    class Linear(disable_weight_init.Linear):
        """#### Linear layer with manual casting."""
        comfy_cast_weights: bool = True

    class Conv2d(disable_weight_init.Conv2d):
        """#### Conv2d layer with manual casting."""
        comfy_cast_weights: bool = True

    class GroupNorm(disable_weight_init.GroupNorm):
        """#### GroupNorm layer with manual casting."""
        comfy_cast_weights: bool = True

    class LayerNorm(disable_weight_init.LayerNorm):
        """#### LayerNorm layer with manual casting."""
        comfy_cast_weights: bool = True