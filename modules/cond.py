import torch
import torch.nn as nn
from modules import Device, util


class CONDRegular:
    """#### Class representing a regular condition."""
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        """#### Copy the condition with a new condition.
        
        #### Args:
            - `cond` (torch.Tensor): The new condition.
        
        #### Returns:
            - `CONDRegular`: The copied condition.
        """
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        """#### Process the condition.
        
        #### Args:
            - `batch_size` (int): The batch size.
            - `device` (torch.device): The device.
            
        #### Returns:
            - `CONDRegular`: The processed condition.
        """
        return self._copy_with(util.repeat_to_batch_size(self.cond, batch_size).to(device))


class CONDCrossAttn(CONDRegular):
    """#### Class representing a cross-attention condition."""
    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = util.lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(
                    1, crossattn_max_len // c.shape[1], 1
                )  # padding with repeat doesn't change result, but avoids an error on tensor shape
            out.append(c)
        return torch.cat(out)
    
def get_models_from_cond(cond, model_type):
    models = []
    return models


def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out

def get_additional_models(conds, dtype):
    """loads additional _internal in conditioning"""
    cnets = []
    gligen = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory


def prepare_sampling(model, noise_shape, conds):
    device = model.load_device
    real_model = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    Device.load_models_gpu(
        [model] + models,
        model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]))
        + inference_memory,
    )
    real_model = model.model

    return real_model, conds, models


def cleanup_models(conds, models):
    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")


def cast_bias_weight(s, input):
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
    comfy_cast_weights = False
    weight_function = None
    bias_function = None


class disable_weight_init:
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

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
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
        def reset_parameters(self):
            return None

        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        return s.Conv2d(*args, **kwargs)


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True
        
# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = manual_cast.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)