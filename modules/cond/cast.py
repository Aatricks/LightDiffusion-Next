from modules.Device import Device
import torch

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return cast_to(
        weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy
    )

def cast_bias_weight(s: torch.nn.Module, input: torch.Tensor= None, dtype:torch.dtype = None, device:torch.device = None, bias_dtype:torch.dtype = None) -> tuple:
    """#### Cast the bias and weight of a module to match the input tensor.

    #### Args:
        - `s` (torch.nn.Module): The module.
        - `input` (torch.Tensor): The input tensor.

    #### Returns:
        - `tuple`: The cast weight and bias.
    """
    if input is not None:
        if dtype is None:
            dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    bias = None
    non_blocking = Device.device_supports_non_blocking(device)
    if s.bias is not None:
        has_function = s.bias_function is not None
        bias = cast_to(
            s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_function
        )
        if has_function:
            bias = s.bias_function(bias)

    has_function = s.weight_function is not None
    weight = cast_to(
        s.weight, dtype, device, non_blocking=non_blocking, copy=has_function
    )
    if has_function:
        weight = s.weight_function(weight)
    return weight, bias

class CastWeightBiasOp:
    """#### Class representing a cast weight and bias operation."""

    comfy_cast_weights: bool = False
    weight_function: callable = None
    bias_function: callable = None


class disable_weight_init:
    """#### Class representing a module with disabled weight initialization."""

    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
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
        
    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None
        
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        
        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        """#### GroupNorm layer with disabled weight initialization."""

        def reset_parameters(self) -> None:
            """#### Reset the parameters of the GroupNorm layer."""
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
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
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
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

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims,
                self.dilation,
            )

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)
    
    @classmethod
    def conv_nd(s, dims: int, *args, **kwargs) -> torch.nn.Conv2d:
        """#### Create a Conv2d layer with the specified dimensions.

        #### Args:
            - `dims` (int): The number of dimensions.
            - `*args`: Variable length argument list.
            - `**kwargs`: Arbitrary keyword arguments.

        #### Returns:
            - `torch.nn.Conv2d`: The Conv2d layer.
        """
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    """#### Class representing a module with manual casting."""

    class Linear(disable_weight_init.Linear):
        """#### Linear layer with manual casting."""

        comfy_cast_weights: bool = True
    
    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        """#### Conv2d layer with manual casting."""

        comfy_cast_weights: bool = True
        
    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        """#### GroupNorm layer with manual casting."""

        comfy_cast_weights: bool = True

    class LayerNorm(disable_weight_init.LayerNorm):
        """#### LayerNorm layer with manual casting."""

        comfy_cast_weights: bool = True
    
    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True
