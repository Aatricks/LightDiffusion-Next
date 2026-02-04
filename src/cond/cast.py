"""Weight casting utilities for efficient model loading."""
from src.Device import Device
import torch


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    """Cast a weight tensor to specified dtype and device."""
    if device is None or weight.device == device:
        if not copy and (dtype is None or weight.dtype == dtype):
            return weight
        return weight.to(dtype=dtype, copy=copy)
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_to_input(weight, input, non_blocking=False, copy=True):
    """Cast weight tensor to match input tensor's dtype and device."""
    return cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
    """Cast module's bias and weight to match input tensor."""
    if input is not None:
        dtype = dtype or input.dtype
        bias_dtype = bias_dtype or dtype
        device = device or input.device

    non_blocking = Device.device_supports_non_blocking(device)
    bias = None
    if s.bias is not None:
        has_fn = s.bias_function is not None
        bias = cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, copy=has_fn)
        if has_fn:
            bias = s.bias_function(bias)

    has_fn = s.weight_function is not None
    weight = cast_to(s.weight, dtype, device, non_blocking=non_blocking, copy=has_fn)
    if has_fn:
        weight = s.weight_function(weight)
    return weight, bias


class CastWeightBiasOp:
    """Mixin for cast weight/bias operations."""
    comfy_cast_weights = False
    weight_function = None
    bias_function = None


class disable_weight_init:
    """Module wrappers with disabled weight initialization."""

    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        def forward(self, *args, **kwargs):
            return self.forward_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input) if self.weight is not None else (None, None)
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        def forward(self, *args, **kwargs):
            return self.forward_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input, output_size=None):
            output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self): return None
        def forward_comfy_cast_weights(self, input, output_size=None):
            output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        def forward(self, *args, **kwargs):
            return self.forward_comfy_cast_weights(*args, **kwargs) if self.comfy_cast_weights else super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None
        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                out_dtype = None
            weight, _ = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            kwargs.pop("out_dtype", None)
            return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        """Create Conv2d/Conv3d based on dimensions."""
        if dims == 2: return cls.Conv2d(*args, **kwargs)
        if dims == 3: return cls.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    """Module wrappers with manual casting enabled by default."""
    class Linear(disable_weight_init.Linear): comfy_cast_weights = True
    class Conv1d(disable_weight_init.Conv1d): comfy_cast_weights = True
    class Conv2d(disable_weight_init.Conv2d): comfy_cast_weights = True
    class Conv3d(disable_weight_init.Conv3d): comfy_cast_weights = True
    class GroupNorm(disable_weight_init.GroupNorm): comfy_cast_weights = True
    class LayerNorm(disable_weight_init.LayerNorm): comfy_cast_weights = True
    class ConvTranspose2d(disable_weight_init.ConvTranspose2d): comfy_cast_weights = True
    class ConvTranspose1d(disable_weight_init.ConvTranspose1d): comfy_cast_weights = True
    class Embedding(disable_weight_init.Embedding): comfy_cast_weights = True
