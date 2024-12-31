from collections import OrderedDict
import functools
import math
import re
from typing import Union, Dict
import torch
import torch.nn as nn
from modules.UltimateSDUpscale import USDU_util


class RRDB(nn.Module):
    """#### Residual in Residual Dense Block (RRDB) class.

    #### Args:
        - `nf` (int): Number of filters.
        - `kernel_size` (int, optional): Kernel size. Defaults to 3.
        - `gc` (int, optional): Growth channel. Defaults to 32.
        - `stride` (int, optional): Stride. Defaults to 1.
        - `bias` (bool, optional): Whether to use bias. Defaults to True.
        - `pad_type` (str, optional): Padding type. Defaults to "zero".
        - `norm_type` (str, optional): Normalization type. Defaults to None.
        - `act_type` (str, optional): Activation type. Defaults to "leakyrelu".
        - `mode` (USDU_util.ConvMode, optional): Convolution mode. Defaults to "CNA".
        - `_convtype` (str, optional): Convolution type. Defaults to "Conv2D".
        - `_spectral_norm` (bool, optional): Whether to use spectral normalization. Defaults to False.
        - `plus` (bool, optional): Whether to use the plus variant. Defaults to False.
        - `c2x2` (bool, optional): Whether to use 2x2 convolution. Defaults to False.
    """

    def __init__(
        self,
        nf: int,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: str = "zero",
        norm_type: str = None,
        act_type: str = "leakyrelu",
        mode: USDU_util.ConvMode = "CNA",
        _convtype: str = "Conv2D",
        _spectral_norm: bool = False,
        plus: bool = False,
        c2x2: bool = False,
    ) -> None:
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            kernel_size,
            gc,
            stride,
            bias,
            pad_type,
            norm_type,
            act_type,
            mode,
            plus=plus,
            c2x2=c2x2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the RRDB.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """#### Residual Dense Block with 5 Convolutions (ResidualDenseBlock_5C) class.

    #### Args:
        - `nf` (int, optional): Number of filters. Defaults to 64.
        - `kernel_size` (int, optional): Kernel size. Defaults to 3.
        - `gc` (int, optional): Growth channel. Defaults to 32.
        - `stride` (int, optional): Stride. Defaults to 1.
        - `bias` (bool, optional): Whether to use bias. Defaults to True.
        - `pad_type` (str, optional): Padding type. Defaults to "zero".
        - `norm_type` (str, optional): Normalization type. Defaults to None.
        - `act_type` (str, optional): Activation type. Defaults to "leakyrelu".
        - `mode` (USDU_util.ConvMode, optional): Convolution mode. Defaults to "CNA".
        - `plus` (bool, optional): Whether to use the plus variant. Defaults to False.
        - `c2x2` (bool, optional): Whether to use 2x2 convolution. Defaults to False.
    """

    def __init__(
        self,
        nf: int = 64,
        kernel_size: int = 3,
        gc: int = 32,
        stride: int = 1,
        bias: bool = True,
        pad_type: str = "zero",
        norm_type: str = None,
        act_type: str = "leakyrelu",
        mode: USDU_util.ConvMode = "CNA",
        plus: bool = False,
        c2x2: bool = False,
    ) -> None:
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1x1 = None

        self.conv1 = USDU_util.conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv2 = USDU_util.conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv3 = USDU_util.conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        self.conv4 = USDU_util.conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            c2x2=c2x2,
        )
        last_act = None
        self.conv5 = USDU_util.conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            c2x2=c2x2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the ResidualDenseBlock_5C.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBNet(nn.Module):
    """#### Residual in Residual Dense Block Network (RRDBNet) class.

    #### Args:
        - `state_dict` (dict): State dictionary.
        - `norm` (str, optional): Normalization type. Defaults to None.
        - `act` (str, optional): Activation type. Defaults to "leakyrelu".
        - `upsampler` (str, optional): Upsampler type. Defaults to "upconv".
        - `mode` (USDU_util.ConvMode, optional): Convolution mode. Defaults to "CNA".
    """

    def __init__(
        self,
        state_dict: Dict[str, torch.Tensor],
        norm: str = None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: USDU_util.ConvMode = "CNA",
    ) -> None:
        super(RRDBNet, self).__init__()
        self.model_arch = "ESRGAN"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.state_map = {
            # currently supports old, new, and newer RRDBNet arch _internal
            # ESRGAN, BSRGAN/RealSR, Real-ESRGAN
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)",
            ),
        }
        self.num_blocks = self.get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state.keys())

        self.state = self.new_to_old_arch(self.state)

        self.key_arr = list(self.state.keys())

        self.in_nc: int = self.state[self.key_arr[0]].shape[1]
        self.out_nc: int = self.state[self.key_arr[-1]].shape[0]

        self.scale: int = self.get_scale()
        self.num_filters: int = self.state[self.key_arr[0]].shape[0]

        c2x2 = False

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        self.shuffle_factor = None

        upsample_block = {
            "upconv": USDU_util.upconv_block,
        }.get(self.upsampler)
        upsample_blocks = [
            upsample_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                act_type=self.act,
                c2x2=c2x2,
            )
            for _ in range(int(math.log(self.scale, 2)))
        ]

        self.model = USDU_util.sequential(
            # fea conv
            USDU_util.conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            USDU_util.ShortcutBlock(
                USDU_util.sequential(
                    # rrdb blocks
                    *[
                        RRDB(
                            nf=self.num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=self.norm,
                            act_type=self.act,
                            mode="CNA",
                            plus=self.plus,
                            c2x2=c2x2,
                        )
                        for _ in range(self.num_blocks)
                    ],
                    # lr conv
                    USDU_util.conv_block(
                        in_nc=self.num_filters,
                        out_nc=self.num_filters,
                        kernel_size=3,
                        norm_type=self.norm,
                        act_type=None,
                        mode=self.mode,
                        c2x2=c2x2,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            USDU_util.conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            USDU_util.conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """#### Convert new architecture state dictionary to old architecture.

        #### Args:
            - `state` (dict): State dictionary.

        #### Returns:
            - `dict`: Converted state dictionary.
        """
        # add nb to state keys
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[
                f"model.1.sub./NB/.{kind}"
            ]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k:
                            old_state[sub] = v
                else:
                    if new_key in state:
                        old_state[old_key] = state[new_key]

        # upconv layers
        max_upconv = 0
        for key in state.keys():
            match = re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key)
            if match is not None:
                _, key_num, key_type = match.groups()
                old_state[f"model.{int(key_num) * 3}.{key_type}"] = state[key]
                max_upconv = max(max_upconv, int(key_num) * 3)

        # final layers
        for key in state.keys():
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key in ("conv_last.weight",):
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key in ("conv_last.bias",):
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        # Sort by first numeric value of each layer
        def compare(item1: str, item2: str) -> int:
            parts1 = item1.split(".")
            parts2 = item2.split(".")
            int1 = int(parts1[1])
            int2 = int(parts2[1])
            return int1 - int2

        sorted_keys = sorted(old_state.keys(), key=functools.cmp_to_key(compare))

        # Rebuild the output dict in the right order
        out_dict = OrderedDict((k, old_state[k]) for k in sorted_keys)

        return out_dict

    def get_scale(self, min_part: int = 6) -> int:
        """#### Get the scale factor.

        #### Args:
            - `min_part` (int, optional): Minimum part. Defaults to 6.

        #### Returns:
            - `int`: Scale factor.
        """
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        """#### Get the number of blocks.

        #### Returns:
            - `int`: Number of blocks.
        """
        nbs = []
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",
        )
        for state_key in state_keys:
            for k in self.state:
                m = re.search(state_key, k)
                if m:
                    nbs.append(int(m.group(1)))
            if nbs:
                break
        return max(*nbs) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass of the RRDBNet.

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        return self.model(x)


PyTorchSRModels = (RRDBNet,)
PyTorchSRModel = Union[RRDBNet,]

PyTorchModels = (*PyTorchSRModels,)
PyTorchModel = Union[PyTorchSRModel]