from typing import Literal
import torch
import torch.nn as nn

ConvMode = Literal["CNA", "NAC", "CNAC"]


def act(act_type: str, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    """Get activation layer (LeakyReLU)."""
    return nn.LeakyReLU(neg_slope, inplace)


def get_valid_padding(kernel_size: int, dilation: int) -> int:
    """Calculate padding for 'same' convolution."""
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


def sequential(*args: nn.Module) -> nn.Sequential:
    """Flatten nested Sequential modules into one."""
    modules = []
    for m in args:
        if isinstance(m, nn.Sequential):
            modules.extend(m.children())
        elif isinstance(m, nn.Module):
            modules.append(m)
    return nn.Sequential(*modules)

def conv_block(in_nc: int, out_nc: int, kernel_size: int, stride: int = 1, dilation: int = 1,
               groups: int = 1, bias: bool = True, pad_type: str = "zero", norm_type=None,
               act_type: str | None = "relu", mode: ConvMode = "CNA", c2x2: bool = False) -> nn.Sequential:
    """Create Conv-Norm-Act block."""
    padding = get_valid_padding(kernel_size, dilation) if pad_type == "zero" else 0
    c = nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, dilation, groups, bias)
    a = act(act_type) if act_type else None
    return sequential(c, a) if mode in ("CNA", "CNAC") else sequential(c)


def upconv_block(in_nc: int, out_nc: int, upscale_factor: int = 2, kernel_size: int = 3,
                 stride: int = 1, bias: bool = True, pad_type: str = "zero", norm_type=None,
                 act_type: str = "relu", mode: str = "nearest", c2x2: bool = False) -> nn.Sequential:
    """Create Upsample + Conv block."""
    return sequential(
        nn.Upsample(scale_factor=upscale_factor, mode=mode),
        conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, pad_type=pad_type, act_type=act_type)
    )


class ShortcutBlock(nn.Module):
    """Residual block: x + submodule(x)."""
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)
