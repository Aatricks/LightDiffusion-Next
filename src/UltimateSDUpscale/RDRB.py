from collections import OrderedDict
import functools
import math
import re
from typing import Union, Dict
import torch
import torch.nn as nn
from src.UltimateSDUpscale import USDU_util


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, nf: int, kernel_size: int = 3, gc: int = 32, stride: int = 1,
                 bias: bool = True, pad_type: str = "zero", norm_type: str = None,
                 act_type: str = "leakyrelu", mode: USDU_util.ConvMode = "CNA",
                 _convtype: str = "Conv2D", _spectral_norm: bool = False,
                 plus: bool = False, c2x2: bool = False) -> None:
        super().__init__()
        args = (nf, kernel_size, gc, stride, bias, pad_type, norm_type, act_type, mode)
        self.RDB1 = ResidualDenseBlock_5C(*args, plus=plus, c2x2=c2x2)
        self.RDB2 = ResidualDenseBlock_5C(*args, plus=plus, c2x2=c2x2)
        self.RDB3 = ResidualDenseBlock_5C(*args, plus=plus, c2x2=c2x2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.RDB3(self.RDB2(self.RDB1(x))) * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """Residual Dense Block with 5 Convolutions."""

    def __init__(self, nf: int = 64, kernel_size: int = 3, gc: int = 32, stride: int = 1,
                 bias: bool = True, pad_type: str = "zero", norm_type: str = None,
                 act_type: str = "leakyrelu", mode: USDU_util.ConvMode = "CNA",
                 plus: bool = False, c2x2: bool = False) -> None:
        super().__init__()
        self.conv1x1 = None
        cb = lambda inc, outc, act=act_type: USDU_util.conv_block(
            inc, outc, kernel_size, stride, bias=bias, pad_type=pad_type,
            norm_type=norm_type, act_type=act, mode=mode, c2x2=c2x2)
        self.conv1 = cb(nf, gc)
        self.conv2 = cb(nf + gc, gc)
        self.conv3 = cb(nf + 2 * gc, gc)
        self.conv4 = cb(nf + 3 * gc, gc)
        self.conv5 = cb(nf + 4 * gc, nf, act=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN/Real-ESRGAN upscaling network."""

    def __init__(self, state_dict: Dict[str, torch.Tensor], norm: str = None,
                 act: str = "leakyrelu", upsampler: str = "upconv",
                 mode: USDU_util.ConvMode = "CNA") -> None:
        super().__init__()
        self.model_arch, self.sub_type = "ESRGAN", "SR"
        self.state, self.norm, self.act, self.upsampler, self.mode = state_dict, norm, act, upsampler, mode
        self.state_map = {
            "model.0.weight": ("conv_first.weight",),
            "model.0.bias": ("conv_first.bias",),
            "model.1.sub./NB/.weight": ("trunk_conv.weight", "conv_body.weight"),
            "model.1.sub./NB/.bias": ("trunk_conv.bias", "conv_body.bias"),
            r"model.1.sub.\1.RDB\2.conv\3.0.\4": (
                r"RRDB_trunk\.(\d+)\.RDB(\d)\.conv(\d+)\.(weight|bias)",
                r"body\.(\d+)\.rdb(\d)\.conv(\d+)\.(weight|bias)"),
        }
        self.num_blocks = self._get_num_blocks()
        self.plus = any("conv1x1" in k for k in self.state)
        self.state = self._new_to_old_arch(self.state)
        self.key_arr = list(self.state.keys())
        self.in_nc = self.state[self.key_arr[0]].shape[1]
        self.out_nc = self.state[self.key_arr[-1]].shape[0]
        self.scale = self._get_scale()
        self.num_filters = self.state[self.key_arr[0]].shape[0]
        self.supports_fp16 = self.supports_bfp16 = True
        self.min_size_restriction = self.shuffle_factor = None

        ups = [USDU_util.upconv_block(self.num_filters, self.num_filters, act_type=self.act)
               for _ in range(int(math.log(self.scale, 2)))]
        cb = lambda inc, outc, act=None: USDU_util.conv_block(inc, outc, 3, norm_type=None, act_type=act)
        self.model = USDU_util.sequential(
            cb(self.in_nc, self.num_filters),
            USDU_util.ShortcutBlock(USDU_util.sequential(
                *[RRDB(self.num_filters, 3, 32, norm_type=self.norm, act_type=self.act, plus=self.plus)
                  for _ in range(self.num_blocks)],
                cb(self.num_filters, self.num_filters))),
            *ups,
            cb(self.num_filters, self.num_filters, act=self.act),
            cb(self.num_filters, self.out_nc))
        self.load_state_dict(self.state, strict=False)

    def _new_to_old_arch(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert new arch state dict to old format."""
        for kind in ("weight", "bias"):
            self.state_map[f"model.1.sub.{self.num_blocks}.{kind}"] = self.state_map[f"model.1.sub./NB/.{kind}"]
            del self.state_map[f"model.1.sub./NB/.{kind}"]

        old_state = OrderedDict()
        for old_key, new_keys in self.state_map.items():
            for new_key in new_keys:
                if r"\1" in old_key:
                    for k, v in state.items():
                        sub = re.sub(new_key, old_key, k)
                        if sub != k: old_state[sub] = v
                elif new_key in state:
                    old_state[old_key] = state[new_key]

        max_upconv = 0
        for key in state:
            if m := re.match(r"(upconv|conv_up)(\d)\.(weight|bias)", key):
                old_state[f"model.{int(m[2]) * 3}.{m[3]}"] = state[key]
                max_upconv = max(max_upconv, int(m[2]) * 3)

        for key in state:
            if key in ("HRconv.weight", "conv_hr.weight"):
                old_state[f"model.{max_upconv + 2}.weight"] = state[key]
            elif key in ("HRconv.bias", "conv_hr.bias"):
                old_state[f"model.{max_upconv + 2}.bias"] = state[key]
            elif key == "conv_last.weight":
                old_state[f"model.{max_upconv + 4}.weight"] = state[key]
            elif key == "conv_last.bias":
                old_state[f"model.{max_upconv + 4}.bias"] = state[key]

        return OrderedDict(sorted(old_state.items(), key=lambda x: int(x[0].split(".")[1])))

    def _get_scale(self, min_part: int = 6) -> int:
        """Get upscale factor."""
        return 2 ** sum(1 for p in self.state if len((ps := p.split("."))[1:]) == 2
                       and int(ps[1]) > min_part and ps[2] == "weight")

    def _get_num_blocks(self) -> int:
        """Get number of RRDB blocks."""
        state_keys = self.state_map[r"model.1.sub.\1.RDB\2.conv\3.0.\4"] + (
            r"model\.\d+\.sub\.(\d+)\.RDB(\d+)\.conv(\d+)\.0\.(weight|bias)",)
        for sk in state_keys:
            if nbs := [int(m[1]) for k in self.state if (m := re.search(sk, k))]:
                return max(nbs) + 1
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


PyTorchSRModels = (RRDBNet,)
PyTorchSRModel = Union[RRDBNet,]

PyTorchModels = (*PyTorchSRModels,)
PyTorchModel = Union[PyTorchSRModel]
