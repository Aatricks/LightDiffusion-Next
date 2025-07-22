from typing import Literal
import torch
import torch.nn as nn

ConvMode = Literal["CNA", "NAC", "CNAC"]

def act(act_type: str, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    """#### Get the activation layer.

    #### Args:
        - `act_type` (str): The type of activation.
        - `inplace` (bool, optional): Whether to perform the operation in-place. Defaults to True.
        - `neg_slope` (float, optional): The negative slope for LeakyReLU. Defaults to 0.2.
        - `n_prelu` (int, optional): The number of PReLU parameters. Defaults to 1.

    #### Returns:
        - `nn.Module`: The activation layer.
    """
    act_type = act_type.lower()
    layer = nn.LeakyReLU(neg_slope, inplace)
    return layer

def get_valid_padding(kernel_size: int, dilation: int) -> int:
    """#### Get the valid padding for a convolutional layer.

    #### Args:
        - `kernel_size` (int): The size of the kernel.
        - `dilation` (int): The dilation rate.

    #### Returns:
        - `int`: The valid padding.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def sequential(*args: nn.Module) -> nn.Sequential:
    """#### Create a sequential container.

    #### Args:
        - `*args` (nn.Module): The modules to include in the sequential container.

    #### Returns:
        - `nn.Sequential`: The sequential container.
    """
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    pad_type: str = "zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
    mode: ConvMode = "CNA",
    c2x2: bool = False,
) -> nn.Sequential:
    """#### Create a convolutional block.

    #### Args:
        - `in_nc` (int): The number of input channels.
        - `out_nc` (int): The number of output channels.
        - `kernel_size` (int): The size of the kernel.
        - `stride` (int, optional): The stride of the convolution. Defaults to 1.
        - `dilation` (int, optional): The dilation rate. Defaults to 1.
        - `groups` (int, optional): The number of groups. Defaults to 1.
        - `bias` (bool, optional): Whether to include a bias term. Defaults to True.
        - `pad_type` (str, optional): The type of padding. Defaults to "zero".
        - `norm_type` (str | None, optional): The type of normalization. Defaults to None.
        - `act_type` (str | None, optional): The type of activation. Defaults to "relu".
        - `mode` (ConvMode, optional): The mode of the convolution. Defaults to "CNA".
        - `c2x2` (bool, optional): Whether to use 2x2 convolutions. Defaults to False.

    #### Returns:
        - `nn.Sequential`: The convolutional block.
    """
    assert mode in ("CNA", "NAC", "CNAC"), "Wrong conv mode [{:s}]".format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    padding = padding if pad_type == "zero" else 0

    c = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    a = act(act_type) if act_type else None
    if mode in ("CNA", "CNAC"):
        return sequential(None, c, None, a)

def upconv_block(
    in_nc: int,
    out_nc: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    pad_type: str = "zero",
    norm_type: str | None = None,
    act_type: str = "relu",
    mode: str = "nearest",
    c2x2: bool = False,
) -> nn.Sequential:
    """#### Create an upsampling convolutional block.

    #### Args:
        - `in_nc` (int): The number of input channels.
        - `out_nc` (int): The number of output channels.
        - `upscale_factor` (int, optional): The upscale factor. Defaults to 2.
        - `kernel_size` (int, optional): The size of the kernel. Defaults to 3.
        - `stride` (int, optional): The stride of the convolution. Defaults to 1.
        - `bias` (bool, optional): Whether to include a bias term. Defaults to True.
        - `pad_type` (str, optional): The type of padding. Defaults to "zero".
        - `norm_type` (str | None, optional): The type of normalization. Defaults to None.
        - `act_type` (str, optional): The type of activation. Defaults to "relu".
        - `mode` (str, optional): The mode of upsampling. Defaults to "nearest".
        - `c2x2` (bool, optional): Whether to use 2x2 convolutions. Defaults to False.

    #### Returns:
        - `nn.Sequential`: The upsampling convolutional block.
    """
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        c2x2=c2x2,
    )
    return sequential(upsample, conv)

class ShortcutBlock(nn.Module):
    """#### Elementwise sum the output of a submodule to its input."""

    def __init__(self, submodule: nn.Module):
        """#### Initialize the ShortcutBlock.

        #### Args:
            - `submodule` (nn.Module): The submodule to apply.
        """
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        output = x + self.sub(x)
        return output
