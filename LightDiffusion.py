from __future__ import annotations

import glob
import os
import random
import sys
import threading
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from typing import Literal

import customtkinter as ctk
import safetensors.torch
from PIL import ImageTk
import PIL

import os
import packaging.version
import torch
import torch.nn as nn

import ollama

if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.12.0"):
    torch.backends.cuda.matmul.allow_tf32 = True

supported_pt_extensions = set([".ckpt", ".pt", ".bin", ".pth", ".safetensors", ".pkl"])

folder_names_and_paths = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "_internal")
folder_names_and_paths["checkpoints"] = (
    [os.path.join(models_dir, "checkpoints")],
    supported_pt_extensions,
)

folder_names_and_paths["loras"] = (
    [os.path.join(models_dir, "loras")],
    supported_pt_extensions,
)

folder_names_and_paths["ESRGAN"] = (
    [os.path.join(models_dir, "ESRGAN")],
    supported_pt_extensions,
)

output_directory = "./_internal/output"

filename_list_cache = {}

if glob.glob("./_internal/checkpoints/*.safetensors") == []:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="Meina/MeinaMix",
        filename="Meina V10 - baked VAE.safetensors",
        local_dir="./_internal/checkpoints/",
    )
if glob.glob("./_internal/yolos/*.pt") == []:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="hand_yolov9c.pt",
        local_dir="./_internal/yolos/",
    )
    hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="face_yolov9c.pt",
        local_dir="./_internal/yolos/",
    )
    hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="person_yolov8m-seg.pt",
        local_dir="./_internal/yolos/",
    )
    hf_hub_download(
        repo_id="segments-arnaud/sam_vit_b",
        filename="sam_vit_b_01ec64.pth",
        local_dir="./_internal/yolos/",
    )
if glob.glob("./_internal/ESRGAN/*.pth") == []:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="lllyasviel/Annotators",
        filename="RealESRGAN_x4plus.pth",
        local_dir="./_internal/ESRGAN/",
    )
if glob.glob("./_internal/loras/*.safetensors") == []:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="EvilEngine/add_detail",
        filename="add_detail.safetensors",
        local_dir="./_internal/loras/",
    )
if glob.glob("./_internal/embeddings/*.pt") == []:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="EvilEngine/badhandv4",
        filename="badhandv4.pt",
        local_dir="./_internal/embeddings/",
    )
    # hf_hub_download(
    #     repo_id="segments-arnaud/sam_vit_b",
    #     filename="EasyNegative.safetensors",
    #     local_dir="./_internal/embeddings/",
    # )
if glob.glob("./_internal/vae_approx/*.pth") == []:
    from huggingface_hub import hf_hub_download
    
    hf_hub_download(
        repo_id="madebyollin/taesd",
        filename="taesd_decoder.safetensors",
        local_dir="./_internal/vae_approx/",
    )

args_parsing = False


class LatentFormat:
    scale_factor = 1.0
    latent_rgb_factors = None
    taesd_decoder_name = None

    def process_in(self, latent):
        return latent * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor


class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


import re
import pickle

load = pickle.load


class Empty:
    pass


# taken from https://github.com/TencentARC/T2I-Adapter
from collections import OrderedDict

import importlib


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3],
        )


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    expanded = x[(...,) + (None,) * dims_to_append]
    # MPS will get inf values if it tries to index into the new axes, but detaching fixes this.
    # https://github.com/pytorch/pytorch/issues/84364
    return expanded.detach().clone() if expanded.device.type == "mps" else expanded


import safetensors.torch


def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
    return sd


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    out = {}
    for rp in replace_prefix:
        replace = list(
            map(
                lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])),
                filter(lambda a: a.startswith(rp), state_dict.keys()),
            )
        )
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out


UNET_MAP_ATTENTIONS = {
    "proj_in.weight",
    "proj_in.bias",
    "proj_out.weight",
    "proj_out.bias",
    "norm.weight",
    "norm.bias",
}

TRANSFORMER_BLOCKS = {
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "attn1.to_q.weight",
    "attn1.to_k.weight",
    "attn1.to_v.weight",
    "attn1.to_out.0.weight",
    "attn1.to_out.0.bias",
    "attn2.to_q.weight",
    "attn2.to_k.weight",
    "attn2.to_v.weight",
    "attn2.to_out.0.weight",
    "attn2.to_out.0.bias",
    "ff.net.0.proj.weight",
    "ff.net.0.proj.bias",
    "ff.net.2.weight",
    "ff.net.2.bias",
}

UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
}


def unet_to_diffusers(unet_config):
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)

    transformers_mid = unet_config.get("transformer_depth_middle", None)

    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "down_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = (
                "input_blocks.{}.0.op.{}".format(n, k)
            )

    i = 0
    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = (
            "middle_block.1.{}".format(b)
        )
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map[
                "mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)
            ] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)

    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map[
                "mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])
            ] = "middle_block.{}.{}".format(n, b)

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[
                    "up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])
                ] = "output_blocks.{}.0.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[
                        "up_blocks.{}.attentions.{}.{}".format(x, i, b)
                    ] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[
                            "up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(
                                x, i, t, b
                            )
                        ] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(
                            n, t, b
                        )
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map[
                        "up_blocks.{}.upsamplers.0.conv.{}".format(x, k)
                    ] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1

    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]

    return diffusers_unet_map


def repeat_to_batch_size(tensor, batch_size):
    return tensor


def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    setattr(obj, attrs[-1], value)
    return prev


def set_attr_param(obj, attr, value):
    return set_attr(obj, attr, torch.nn.Parameter(value, requires_grad=False))

def copy_to_param(obj, attr, value):
    # inplace update tensor instead of replacing it
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        """slerps batches b1, b2 according to ratio r, batches should be flat e.g. NxC"""

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
            1
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
            1
        ) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1)
        )
        coords_1 = torch.nn.functional.interpolate(
            coords_1, size=(1, length_new), mode="bilinear"
        )
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape(
                (1, 1, 1, -1)
            )
            + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2, size=(1, length_new), mode="bilinear"
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def common_upscale(samples, width, height, upscale_method, crop):
    s = samples
    return bislerp(s, width, height)


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None


class ProgressBar:
    def __init__(self, total):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK


LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora, to_load):
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None

        regular_lora = "{}.lora_up.weight".format(x)
        diffusers_lora = "{}_lora.up.weight".format(x)
        transformers_lora = "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            mid_name = "{}.lora_mid.weight".format(x)

        if A_name is not None:
            mid = None
            patch_dict[to_load[x]] = (
                "lora",
                (lora[A_name], lora[B_name], alpha, mid, dora_scale),
            )
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
    return patch_dict


def model_lora_keys_clip(model, key_map={}):
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    clip_l_present = False
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(
                    b, LORA_CLIP_MAP[c]
                )  # SDXL base
                key_map[lora_key] = k
                clip_l_present = True
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                    b, c
                )  # diffusers lora
                key_map[lora_key] = k
    return key_map


def model_lora_keys_unet(model, key_map={}):
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = k  # cascade lora:

    diffusers_keys = unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(
                    p, k[: -len(".weight")].replace(".to_", ".processor.to_")
                )
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key
    return key_map


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))


class CONDCrossAttn(CONDRegular):
    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(
                    1, crossattn_max_len // c.shape[1], 1
                )  # padding with repeat doesn't change result, but avoids an error on tensor shape
            out.append(c)
        return torch.cat(out)


import argparse
import enum


class EnumAction(argparse.Action):
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)


"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""

def conv(n_in, n_out, **kwargs):
    return disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder2(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def Decoder2(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4):
        super().__init__()
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))
        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.taesd_encoder = Encoder2(latent_channels)
        self.taesd_decoder = Decoder2(latent_channels)
        decoder_path = "./_internal/vae_approx/taesd_decoder.safetensors" if decoder_path is None else decoder_path
        if encoder_path is not None:
            self.taesd_encoder.load_state_dict(load_torch_file(encoder_path, safe_load=True))
        if decoder_path is not None:
            self.taesd_decoder.load_state_dict(load_torch_file(decoder_path, safe_load=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        device = next(self.taesd_decoder.parameters()).device
        x = x.to(device)
        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        device = next(self.taesd_encoder.parameters()).device
        x = x.to(device) 
        return (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift

def taesd_preview(x):
    if app.previewer_checkbox.get() == True:
        taesd_instance = TAESD()
        for image in taesd_instance.decode(x[0].unsqueeze(0))[0]:
            i = 255.0 * image.cpu().detach().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")
        app.update_image(img)
    else:
        pass


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


import logging

logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas


def checkpoint(func, inputs, params, flag):
    return func(*inputs)


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


import torch
import torchsde
from torch import nn
from tqdm.auto import trange, tqdm


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        seed = [seed]
        self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs)
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack(
            [
                tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device)
                for tree in self.trees
            ]
        ) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    def __init__(
        self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False
    ):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(
            torch.as_tensor(sigma_max)
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(
            torch.as_tensor(sigma_next)
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if app.interrupt_flag == True:
                break
        try:
            app.title(f"LightDiffusion - {i}it")
        except:
            pass
        
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        if app.previewer_checkbox.get() == True:
                threading.Thread(target=taesd_preview, args=(x,)).start()
        else:
            pass
    return x


class PIDStepSizeController:
    def __init__(
        self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8
    ):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = (
            self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        )
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (
            x - self.model(x, sigma, *args, **self.extra_args, **kwargs)
        ) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        )
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (
            x
            - self.sigma(s2) * (r2 * h).expm1() * eps
            - self.sigma(s2)
            * (r2 / r1)
            * ((r2 * h).expm1() / (r2 * h) - 1)
            * (eps_r1 - eps)
        )
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        )
        return x_3, eps_cache

    def dpm_solver_adaptive(
        self,
        x,
        t_start,
        t_end,
        order=3,
        rtol=0.05,
        atol=0.0078,
        h_init=0.05,
        pcoeff=0.0,
        icoeff=1.0,
        dcoeff=0.0,
        accept_safety=0.81,
        eta=0.0,
        s_noise=1.0,
        noise_sampler=None,
    ):
        noise_sampler = (
            default_noise_sampler(x) if noise_sampler is None else noise_sampler
        )
        forward = t_end > t_start
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(
            h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety
        )
        info = {"steps": 0, "nfe": 0, "n_accept": 0, "n_reject": 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            try:
                app.title(f"LightDiffusion - {info['steps']*3}it")
            except:
                pass
            if app.interrupt_flag == True:
                break
            eps_cache = {}
            t = (
                torch.minimum(t_end, s + pid.h)
                if forward
                else torch.maximum(t_end, s + pid.h)
            )
            t_, su = t, 0.0

            eps, eps_cache = self.eps(eps_cache, "eps", x, s)
            denoised = x - self.sigma(s) * eps

            x_low, eps_cache = self.dpm_solver_2_step(
                x, s, t_, r1=1 / 3, eps_cache=eps_cache
            )
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info["n_accept"] += 1
            else:
                info["n_reject"] += 1
            info["nfe"] += order
            info["steps"] += 1
            if app.previewer_checkbox.get() == True:
                threading.Thread(target=taesd_preview, args=(x,)).start()
            else:
                pass
            
        try:
            app.title("LightDiffusion")
        except:
            pass
        return x, info


@torch.no_grad()
def sample_dpm_adaptive(
    model,
    x,
    sigma_min,
    sigma_max,
    extra_args=None,
    callback=None,
    disable=None,
    order=3,
    rtol=0.05,
    atol=0.0078,
    h_init=0.05,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    accept_safety=0.81,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
    return_info=False,
):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        x, info = dpm_solver.dpm_solver_adaptive(
            x,
            dpm_solver.t(torch.tensor(sigma_max)),
            dpm_solver.t(torch.tensor(sigma_min)),
            order,
            rtol,
            atol,
            h_init,
            pcoeff,
            icoeff,
            dcoeff,
            accept_safety,
            eta,
            s_noise,
            noise_sampler,
        )
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
):
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        if app.interrupt_flag == True:
                break
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = (
                sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x
                + (-h - eta_h).expm1().neg() * denoised
            )

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                        1 / r
                    ) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                        denoised - old_denoised
                    )

            if eta:
                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * s_noise
                )
        if app.previewer_checkbox.get() == True:
                threading.Thread(target=taesd_preview, args=(x,)).start()
        else:
            pass

        old_denoised = denoised
        h_last = h
    return x


class TimestepBlock1(nn.Module):
    pass


class TimestepEmbedSequential1(nn.Sequential, TimestepBlock1):
    pass


import torch


class EPS:
    def calculate_input(self, sigma, noise):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma

        noise += latent_image
        return noise

    def inverse_noise_scaling(self, sigma, latent):
        return latent


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        sampling_settings = model_config.sampling_settings
        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        self._register_schedule(
            given_betas=None,
            beta_schedule=beta_schedule,
            timesteps=1000,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=8e-3,
        )
        self.sigma_data = 1.0

    def _register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep):
        t = torch.clamp(
            timestep.float().to(self.log_sigmas.device),
            min=0,
            max=(len(self.sigmas) - 1),
        )
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp().to(timestep.device)


import logging
import sys
from enum import Enum

import psutil
import torch


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move _internal to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but _internal still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

directml_enabled = False
try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info(
    "Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram)
)
try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except:
        pass
    try:
        XFORMERS_VERSION = xformers.version.__version__
        logging.info("xformers version: {}".format(XFORMERS_VERSION))
        if XFORMERS_VERSION.startswith("0.0.18"):
            logging.warning(
                "\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images."
            )
            logging.warning(
                "Please downgrade or upgrade xformers to a different version.\n"
            )
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION == False:
                ENABLE_PYTORCH_ATTENTION = True
            if (
                torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_properties(torch.cuda.current_device()).major
                >= 8
            ):
                VAE_DTYPE = torch.bfloat16
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


FORCE_FP32 = False
FORCE_FP16 = False

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = False

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")


def get_torch_device_name(device):
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(
                device, torch.cuda.get_device_name(device), allocator_backend
            )
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")

logging.info("VAE dtype: {}".format(VAE_DTYPE))

current_loaded_models = []


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        try:
            if lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(
                    device_to=patch_model_to,
                    lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights,
                )
            else:
                self.real_model = self.model.patch_model(
                    device_to=patch_model_to, patch_weights=load_weights
                )
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e
        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter > 0:
            return True
        return False

    def model_unload(self, unpatch_weights=True):
        self.model.unpatch_model(
            self.model.offload_device, unpatch_weights=unpatch_weights
        )
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def __eq__(self, other):
        return self.model is other.model


def minimum_inference_memory():
    return 1024 * 1024 * 1024


def unload_model_clones(model, unload_weights_only=True, force_unload=True):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight == False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight


def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append(
                    (sys.getrefcount(shift_model.model), shift_model.model_memory(), i)
                )

    for x in sorted(can_unload):
        i = x[-1]
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(
                device, torch_free_too=True
            )
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def load_models_gpu(models, memory_required=0, force_patch_weights=False):
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models = set(models)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        loaded = None

        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            if loaded.should_reload_model(force_patch_weights=force_patch_weights):
                current_loaded_models.pop(loaded_model_index).model_unload(
                    unpatch_weights=True
                )
                loaded = None
            else:
                models_already_loaded.append(loaded)

        if loaded is None:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return

    logging.info(
        f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}"
    )

    total_memory_required = {}
    for loaded_model in models_to_load:
        if (
            unload_model_clones(
                loaded_model.model, unload_weights_only=True, force_unload=False
            )
            == True
        ):  # unload clones where the weights are different
            total_memory_required[loaded_model.device] = total_memory_required.get(
                loaded_model.device, 0
            ) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(
                total_memory_required[device] * 1.3 + extra_mem,
                device,
                models_already_loaded,
            )

    for loaded_model in models_to_load:
        weights_unloaded = unload_model_clones(
            loaded_model.model, unload_weights_only=False, force_unload=False
        )  # unload the rest of the clones where the weights can stay loaded
        if weights_unloaded is not None:
            loaded_model.weights_loaded = not weights_unloaded

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (
            vram_set_state == VRAMState.LOW_VRAM
            or vram_set_state == VRAMState.NORMAL_VRAM
        ):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(
                max(64 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3)
            )
            if model_size > (
                current_free_mem - inference_memory
            ):  # only switch to lowvram if really necessary
                vram_set_state = VRAMState.LOW_VRAM
            else:
                lowvram_model_memory = 0

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 64 * 1024 * 1024

        cur_loaded_model = loaded_model.model_load(
            lowvram_model_memory, force_patch_weights=force_patch_weights
        )
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models(keep_clone_weights_loaded=False):
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            elif (
                sys.getrefcount(current_loaded_models[i].real_model) <= 3
            ):  # references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_dtype(
    device=None,
    model_params=0,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32


# None means no manual cast
def unet_manual_cast(
    weight_dtype,
    inference_device,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    elif bf16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32


def text_encoder_offload_device():
    return torch.device("cpu")


def text_encoder_device():
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device():
    return torch.device("cpu")


def vae_device():
    return get_torch_device()


def vae_offload_device():
    return torch.device("cpu")


def vae_dtype():
    global VAE_DTYPE
    return VAE_DTYPE


def get_autocast_device(dev):
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def device_supports_non_blocking(device):
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    return False
    # return True


def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
    return False


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = (
                torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            )
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, "type"):
        if device.type == type:
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, "cpu")


def is_device_mps(device):
    return is_device_type(device, "mps")


def is_device_cuda(device):
    return is_device_type(device, "cuda")


def should_use_fp16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    fp16_works = False
    # FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    # when the model doesn't actually fit on the card
    nvidia_10_series = [
        "1080",
        "1070",
        "titan x",
        "p3000",
        "p3200",
        "p4000",
        "p4200",
        "p5000",
        "p5200",
        "p6000",
        "1060",
        "1050",
        "p40",
        "p100",
        "p6",
        "p4",
    ]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = [
        "1660",
        "1650",
        "1630",
        "T500",
        "T550",
        "T600",
        "MX550",
        "MX450",
        "CMP 30HX",
        "T2000",
        "T1000",
        "T1200",
    ]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(
    device=None, model_params=0, prioritize_performance=True, manual_cast=False
):
    if device is not None:
        if is_device_cpu(device):
            return False

    if device is not None:
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if (
            force or is_nvidia()
        ):  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models():
    free_memory(1e30, get_torch_device())


def resolve_lowvram_weight(weight, model, key):
    return weight


import threading


class InterruptProcessingException(Exception):
    pass


interrupt_processing_mutex = threading.RLock()

interrupt_processing = False

import torch


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
    load_models_gpu(
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
    non_blocking = device_supports_non_blocking(input.device)
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


import collections


def get_area_and_mult(conds, x_in, timestep_in):
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    input_x = x_in[:, :, area[2] : area[0] + area[2], area[3] : area[1] + area[3]]
    mask = torch.ones_like(input_x)
    mult = mask * strength

    if "mask" not in conds:
        rr = 8

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(
            batch_size=x_in.shape[0], device=x_in.device, area=area
        )

    control = conds.get("control", None)
    patches = None
    cond_obj = collections.namedtuple(
        "cond_obj", ["input_x", "mult", "conditioning", "area", "control", "patches"]
    )
    return cond_obj(input_x, mult, conditioning, area, control, patches)


def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    return True


def can_concat_cond(c1, c2):
    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out


def calc_cond_batch(model, conds, x_in, timestep, model_options):
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = get_area_and_mult(x, x_in, timestep)
                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c["transformer_options"] = transformer_options

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            out_conds[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += (
                output[o] * mult[o]
            )
            out_counts[cond_index][
                :,
                :,
                area[o][2] : area[o][0] + area[o][2],
                area[o][3] : area[o][1] + area[o][3],
            ] += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds


def cfg_function(
    model,
    cond_pred,
    uncond_pred,
    cond_scale,
    x,
    timestep,
    model_options={},
    cond=None,
    uncond=None,
):
    cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    return cfg_result


def sampling_function(
    model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None
):
    uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)
    return cfg_function(
        model,
        out[0],
        out[1],
        cond_scale,
        x,
        timestep,
        model_options=model_options,
        cond=cond,
        uncond=uncond_,
    )


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas

    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        return out


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    s = model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def resolve_areas_and_cond_masks(conditions, h, w, device):
    for i in range(len(conditions)):
        c = conditions[i]


def create_cond_with_same_area_if_none(conds, c):
    if "area" not in c:
        return


def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]


def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)


def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if "area" not in x:
            cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if "area" not in x:
            uncond_other.append((x, t))


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        params["width"] = params.get("width", noise.shape[3] * 8)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        for k in out:
            model_conds[k] = out[k]
        x["model_conds"] = model_conds
        conds[t] = x
    return conds


class Sampler:
    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


KSAMPLER_NAMES = [
    "euler_ancestral",
    "dpm_adaptive",
    "dpmpp_2m_sde",
]


class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None
        total_steps = len(sigmas) - 1

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_adaptive":

        def dpm_adaptive_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpm_adaptive(
                model,
                noise,
                sigma_min,
                sigmas[0],
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpm_adaptive_function
    elif sampler_name == "dpmpp_2m_sde":

        def dpmpp_sde_function(
            model, noise, sigmas, extra_args, callback, disable, **extra_options
        ):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return sample_dpmpp_2m_sde(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = dpmpp_sde_function
    elif sampler_name == "euler_ancestral":

        def euler_ancestral_function(
            model, noise, sigmas, extra_args, callback, disable
        ):
            return sample_euler_ancestral(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **extra_options,
            )

        sampler_function = euler_ancestral_function

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(
    model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None
):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks(conds[k], noise.shape[2], noise.shape[3], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(
                model.extra_conds,
                conds[k],
                noise,
                device,
                k,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed,
            )

    # make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(
                    list(
                        filter(
                            lambda c: c.get("control_apply_to_uncond", False) == True,
                            positive,
                        )
                    ),
                    conds[k],
                    "control",
                    lambda cond_cnets, x: cond_cnets[x],
                )
                apply_empty_x_to_equal_area(
                    positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x]
                )

    return conds


class CFGGuider:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options,
            seed=seed,
        )

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(
            self.inner_model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = prepare_sampling(
            self.model_patcher, noise.shape, self.conds
        )
        device = self.model_patcher.load_device

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        output = self.inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
        )

        cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output


def sample(
    model,
    noise,
    positive,
    negative,
    cfg,
    device,
    sampler,
    sigmas,
    model_options={},
    latent_image=None,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed
    )


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


def calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(
            n=steps,
            sigma_min=float(model_sampling.sigma_min),
            sigma_max=float(model_sampling.sigma_max),
        )
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model_sampling, steps)
    return sigmas


def sampler_object(name):
    sampler = ksampler(name)
    return sampler


class KSampler1:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(
        ("dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2")
    )

    def __init__(
        self,
        model,
        steps,
        device,
        sampler=None,
        scheduler=None,
        denoise=None,
        model_options={},
    ):
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        sigmas = calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )

        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps / denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1) :]

    def sample(
        self,
        noise,
        positive,
        negative,
        cfg,
        latent_image=None,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        denoise_mask=None,
        sigmas=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        if sigmas is None:
            sigmas = self.sigmas

        sampler = sampler_object(self.sampler)

        return sample(
            self.model,
            noise,
            positive,
            negative,
            cfg,
            self.device,
            sampler,
            sigmas,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )


def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent_image.size(),
        dtype=latent_image.dtype,
        layout=latent_image.layout,
        generator=generator,
        device="cpu",
    )


def sample1(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    noise_mask=None,
    sigmas=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    sampler = KSampler1(
        model,
        steps=steps,
        device=model.load_device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=denoise,
        model_options=model.model_options,
    )

    samples = sampler.sample(
        noise,
        positive,
        negative,
        cfg=cfg,
        latent_image=latent_image,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        denoise_mask=noise_mask,
        sigmas=sigmas,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(intermediate_device())
    return samples


import uuid



class ModelPatcher:
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        current_device=None,
        weight_inplace_update=False,
    ):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name):
        return get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        return sd

    def patch_weight_to_device(self, key, device_to=None):
        if key not in self.patches:
            return

        weight = get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

        if device_to is not None:
            temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
        if inplace_update:
            copy_to_param(self.model, key, out_weight)
        else:
            set_attr_param(self.model, key, out_weight)
    
    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning("could not patch. key doesn't exist in model: {}".format(key))
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        self.patch_model(device_to, patch_weights=False)

        logging.info("loading in lowvram mode {}".format(lowvram_model_memory/(1024 * 1024)))
        class LowVramPatch:
            def __init__(self, key, model_patcher):
                self.key = key
                self.model_patcher = model_patcher
            def __call__(self, weight):
                return self.model_patcher.calculate_weight(self.model_patcher.patches[self.key], weight, self.key)

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]
            patch_type = v[0]
            v = v[1]
            mat1 = cast_to_device(v[0], weight.device, torch.float32)
            mat2 = cast_to_device(v[1], weight.device, torch.float32)
            dora_scale = v[4]
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            weight += (
                (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
                .reshape(weight.shape)
                .type(weight.dtype)
            )
        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            keys = list(self.backup.keys())
            for k in keys:
                set_attr_param(self.model, k, self.backup[k])
            self.backup.clear()
            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        self.object_patches_backup.clear()


# import pytorch_lightning as pl
from typing import Dict, Tuple

import torch


class DiagonalGaussianRegularizer(torch.nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.sample()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AutoencodingEngine(nn.Module):
    def __init__(self, encoder, decoder, regularizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularization = regularizer
        self.post_quant_conv = disable_weight_init.Conv2d(4, 4, 1)
        self.quant_conv = disable_weight_init.Conv2d(8, 8, 1)

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        dec = self.post_quant_conv(z)
        dec = self.decoder(dec, **decoder_kwargs)
        return dec

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)
        z = self.quant_conv(z)
        z, reg_log = self.regularization(z)
        return z


import torch.nn as nn

ops = disable_weight_init

if xformers_enabled_vae():
    import xformers
    import xformers.ops


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = ops.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = torch.nn.SiLU(inplace=True)
        self.norm1 = Normalize(in_channels)
        self.conv1 = ops.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = ops.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = ops.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


def xformers_attention(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, C, -1).transpose(1, 2).contiguous(),
        (q, k, v),
    )
    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
    out = out.transpose(1, 2).reshape(B, C, H, W)
    return out


def pytorch_attention(q, k, v):
    # compute attention
    B, C, H, W = q.shape
    q, k, v = map(
        lambda t: t.view(B, 1, C, -1).transpose(2, 3).contiguous(),
        (q, k, v),
    )
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(2, 3).reshape(B, C, H, W)
    return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = ops.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        if xformers_enabled_vae():
            logging.info("Using xformers attention in VAE")
            self.optimized_attention = xformers_attention
        else:
            logging.info("Using pytorch attention in VAE")
            self.optimized_attention = pytorch_attention

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        h_ = self.optimized_attention(q, k, v)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    return AttnBlock(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = ops.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = ops.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        conv_out_op=ops.Conv2d,
        resnet_op=ResnetBlock,
        attn_op=AttnBlock,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.debug(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = ops.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = attn_op(block_in)
        self.mid.block_2 = resnet_op(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    resnet_op(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_out_op(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        return h


import logging

from torch import nn

if xformers_enabled():
    import xformers
    import xformers.ops

ops = disable_weight_init

_ATTN_PRECISION = "fp32"


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                operations.Linear(dim, inner_dim, dtype=dtype, device=device), nn.GELU()
            )
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


def attention_xformers(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return out


def attention_pytorch(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
        (q, k, v),
    )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


if xformers_enabled():
    logging.info("Using xformers cross attention")
    optimized_attention = attention_xformers
else:
    logging.info("Using pytorch cross attention")
    optimized_attention = attention_pytorch

optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
    return attention_pytorch


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=False, dtype=dtype, device=device
        )

        self.to_out = nn.Sequential(
            operations.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        out = optimized_attention(q, k, v, self.heads)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        ff_in=False,
        inner_dim=None,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(
            inner_dim,
            dim_out=dim,
            dropout=dropout,
            glu=gated_ff,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        context_dim_attn2 = None
        if not switch_temporal_ca_to_sa:
            context_dim_attn2 = context_dim

        self.attn2 = CrossAttention(
            query_dim=inner_dim,
            context_dim=context_dim_attn2,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            device=device,
            operations=operations,
        )  # is self-attn if context is none
        self.norm2 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(
            self._forward,
            (x, context, transformer_options),
            self.parameters(),
            self.checkpoint,
        )

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        n = self.norm1(x)
        context_attn1 = None
        value_attn1 = None

        transformer_block = (block[0], block[1], block_index)
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        n = self.attn1(n, context=context_attn1, value=value_attn1)

        x += n

        if self.attn2 is not None:
            n = self.norm2(x)
            context_attn2 = context
            value_attn2 = None

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block
            n = self.attn2(n, context=context_attn2, value=value_attn2)

        x += n
        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
            device=device,
        )
        if not use_linear:
            self.proj_in = operations.Conv2d(
                in_channels,
                inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device
            )
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


import torch


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        super().__init__()

        self.heads = heads
        self.q_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.k_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )
        self.v_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

        self.out_proj = operations.Linear(
            embed_dim, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": torch.nn.functional.gelu,
}


class CLIPMLP(torch.nn.Module):
    def __init__(
        self, embed_dim, intermediate_size, activation, dtype, device, operations
    ):
        super().__init__()
        self.fc1 = operations.Linear(
            embed_dim, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(
            intermediate_size, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layer_norm1 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device, operations)
        self.layer_norm2 = operations.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = CLIPMLP(
            embed_dim,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )

    def forward(self, x, mask=None, optimized_attention=None):
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        heads,
        intermediate_size,
        intermediate_activation,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                CLIPLayer(
                    embed_dim,
                    heads,
                    intermediate_size,
                    intermediate_activation,
                    dtype,
                    device,
                    operations,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, intermediate_output=None):
        optimized_attention = optimized_attention_for_device(
            x.device, mask=mask is not None, small_input=True
        )

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(
        self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        num_layers = config_dict["num_hidden_layers"]
        embed_dim = config_dict["hidden_size"]
        heads = config_dict["num_attention_heads"]
        intermediate_size = config_dict["intermediate_size"]
        intermediate_activation = config_dict["hidden_act"]

        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device)
        self.encoder = CLIPEncoder(
            num_layers,
            embed_dim,
            heads,
            intermediate_size,
            intermediate_activation,
            dtype,
            device,
            operations,
        )
        self.final_layer_norm = operations.LayerNorm(
            embed_dim, dtype=dtype, device=device
        )

    def forward(
        self,
        input_tokens,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
    ):
        x = self.embeddings(input_tokens)
        mask = None

        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            input_tokens.to(dtype=torch.int, device=x.device).argmax(dim=-1),
        ]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


from inspect import isfunction

from torch import nn

ops = manual_cast


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = ops.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


import json
import traceback
import zipfile

import torch
from transformers import CLIPTokenizer


def gen_empty_tokens(special_tokens, length):
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        out, pooled = self.encode(to_encode)
        first_pooled = pooled[0:1].to(intermediate_device())

        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        return torch.cat(output, dim=-2).to(intermediate_device()), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cpu",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        textmodel_json_config=None,
        dtype=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        enable_attention_masks=False,
        return_projected_pooled=True,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = "./_internal/clip/sd1_clip_config.json"

        with open(textmodel_json_config) as f:
            config = json.load(f)

        self.transformer = model_class(config, dtype, device, manual_cast)
        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        self.layer = "hidden"
        self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:  # EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        logging.warning(
                            "WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
                                y.shape[0], current_embeds.weight.shape[1]
                            )
                        )
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(
                next_new_token + 1,
                current_embeds.weight.shape[1],
                device=current_embeds.weight.device,
                dtype=current_embeds.weight.dtype,
            )
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))
            ]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None

        outputs = self.transformer(
            tokens,
            attention_mask,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text):
    text = text.replace("\\)", "\0\1")
    text = text.replace("\\(", "\0\2")
    return text


def unescape_important(text):
    text = text.replace("\0\1", ")")
    text = text.replace("\0\2", "(")
    return text


def expand_directory_list(directories):
    dirs = set()
    for x in directories:
        dirs.add(x)
        for root, subdir, file in os.walk(x, followlinks=True):
            dirs.add(root)
    return list(dirs)


def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    if isinstance(embedding_directory, str):
        embedding_directory = [embedding_directory]

    embedding_directory = expand_directory_list(embedding_directory)

    valid_file = None
    for embed_dir in embedding_directory:
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        embed_dir = os.path.abspath(embed_dir)
        try:
            if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
                continue
        except:
            continue
        if not os.path.isfile(embed_path):
            extensions = [".safetensors", ".pt", ".bin"]
            for x in extensions:
                t = embed_path + x
                if os.path.isfile(t):
                    valid_file = t
                    break
        else:
            valid_file = embed_path
        if valid_file is not None:
            break

    if valid_file is None:
        return None

    embed_path = valid_file

    embed_out = None

    try:
        if embed_path.lower().endswith(".safetensors"):
            import safetensors.torch

            embed = safetensors.torch.load_file(embed_path, device="cpu")
        else:
            if "weights_only" in torch.load.__code__.co_varnames:
                embed = torch.load(embed_path, weights_only=True, map_location="cpu")
            else:
                embed = torch.load(embed_path, map_location="cpu")
    except Exception as e:
        logging.warning(
            "{}\n\nerror loading embedding, skipping loading: {}".format(
                traceback.format_exc(), embedding_name
            )
        )
        return None

    if embed_out is None:
        if "string_to_param" in embed:
            values = embed["string_to_param"].values()
            embed_out = next(iter(values))
        elif isinstance(embed, list):
            out_list = []
            for x in range(len(embed)):
                for k in embed[x]:
                    t = embed[x][k]
                    if t.shape[-1] != embedding_size:
                        continue
                    out_list.append(t.reshape(-1, t.shape[-1]))
            embed_out = torch.cat(out_list, dim=0)
        elif embed_key is not None and embed_key in embed:
            embed_out = embed[embed_key]
        else:
            values = embed.values()
            embed_out = next(iter(values))
    return embed_out


class SDTokenizer:
    def __init__(
        self,
        tokenizer_path=None,
        max_length=77,
        pad_with_end=True,
        embedding_directory=None,
        embedding_size=768,
        embedding_key="clip_l",
        tokenizer_class=CLIPTokenizer,
        has_start_token=True,
        pad_to_max_length=True,
        min_length=None,
    ):
        if tokenizer_path is None:
            tokenizer_path = "_internal/sd1_tokenizer/"
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name: str):
        embed = load_embed(
            embedding_name,
            self.embedding_directory,
            self.embedding_size,
            self.embedding_key,
        )
        if embed is None:
            stripped = embedding_name.strip(",")
            if len(stripped) < len(embedding_name):
                embed = load_embed(
                    stripped,
                    self.embedding_directory,
                    self.embedding_size,
                    self.embedding_key,
                )
                return (embed, embedding_name[len(stripped) :])
        return (embed, "")

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = (
                unescape_important(weighted_segment).replace("\n", " ").split(" ")
            )
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # if we find an embedding, deal with the embedding
                if (
                    word.startswith(self.embedding_identifier)
                    and self.embedding_directory is not None
                ):
                    embedding_name = word[len(self.embedding_identifier) :].strip("\n")
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        logging.warning(
                            f"warning, embedding:{embedding_name} does not exist, ignoring"
                        )
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append(
                                [(embed[x], weight) for x in range(embed.shape[0])]
                            )
                        print("loading ", embedding_name)
                    # if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                # parse word
                tokens.append(
                    [
                        (t, weight)
                        for t in self.tokenizer(word)["input_ids"][
                            self.tokens_start : -1
                        ]
                    ]
                )

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend(
                            [(t, w, i + 1) for t, w in t_group[:remaining_length]]
                        )
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SD1Tokenizer:
    def __init__(self, embedding_directory=None, clip_name="l", tokenizer=SDTokenizer):
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, tokenizer(embedding_directory=embedding_directory))

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(
            text, return_word_ids
        )
        return out

    def untokenize(self, token_weight_pair):
        return getattr(self, self.clip).untokenize(token_weight_pair)


class SD1ClipModel(torch.nn.Module):
    def __init__(
        self, device="cpu", dtype=None, clip_name="l", clip_model=SDClipModel, **kwargs
    ):
        super().__init__()
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def set_clip_options(self, options):
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self):
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out, pooled = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out, pooled


from abc import abstractmethod

import torch as th
import torch.nn as nn

oai_ops = disable_weight_init


class TimestepBlock1(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


def forward_timestep_embed1(
    ts,
    x,
    emb,
    context=None,
    transformer_options={},
    output_shape=None,
    time_context=None,
    num_video_frames=None,
    image_only_indicator=None,
):
    for layer in ts:
        if isinstance(layer, TimestepBlock1):
            x = layer(x, emb)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context, transformer_options)
            if "transformer_index" in transformer_options:
                transformer_options["transformer_index"] += 1
        elif isinstance(layer, Upsample1):
            x = layer(x, output_shape=output_shape)
        else:
            x = layer(x)
    return x


class Upsample1(nn.Module):
    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = operations.conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
                dtype=dtype,
                device=device,
            )

    def forward(self, x, output_shape=None):
        assert x.shape[1] == self.channels
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample1(nn.Module):
    def __init__(
        self,
        channels,
        use_conv,
        dims=2,
        out_channels=None,
        padding=1,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        self.op = operations.conv_nd(
            dims,
            self.channels,
            self.out_channels,
            3,
            stride=stride,
            padding=padding,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock1(TimestepBlock1):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(
                dims,
                channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        self.updown = up or down

        self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                emb_channels,
                (2 * self.out_channels if use_scale_shift_norm else self.out_channels),
                dtype=dtype,
                device=device,
            ),
        )
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            operations.conv_nd(
                dims,
                self.out_channels,
                self.out_channels,
                kernel_size,
                padding=padding,
                dtype=dtype,
                device=device,
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = operations.conv_nd(
                dims, channels, self.out_channels, 1, dtype=dtype, device=device
            )

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)

        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if emb_out is not None:
            h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


def apply_control1(h, control, name):
    return h


class UNetModel1(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=th.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_temporal_resblock=False,
        use_temporal_attention=False,
        time_context_dim=None,
        extra_ff_mix_layer=False,
        use_spatial_context=False,
        merge_strategy=None,
        merge_factor=0.0,
        video_kernel_size=None,
        disable_temporal_crossattention=False,
        max_ddpm_temb_period=10000,
        device=None,
        operations=oai_ops,
    ):
        super().__init__()

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        transformer_depth = transformer_depth[:]
        transformer_depth_output = transformer_depth_output[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_temporal_resblocks = use_temporal_resblock
        self.predict_codebook_ids = n_embed is not None

        self.default_num_video_frames = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(
                model_channels, time_embed_dim, dtype=self.dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                time_embed_dim, time_embed_dim, dtype=self.dtype, device=device
            ),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential1(
                    operations.conv_nd(
                        dims,
                        in_channels,
                        model_channels,
                        3,
                        padding=1,
                        dtype=self.dtype,
                        device=device,
                    )
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disable_self_attn=False,
        ):
            return SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_channels,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
            dtype=None,
            device=None,
            operations=oai_ops,
        ):
            return ResBlock1(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_channels,
                use_checkpoint=use_checkpoint,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                dtype=dtype,
                device=device,
                operations=operations,
            )

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    dim_head = ch // num_heads
                    disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential1(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Downsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        dim_head = ch // num_heads
        mid_block = [
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                out_channels=None,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations,
            )
        ]

        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [
                    get_attention_layer(  # always uses a self-attn
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth_middle,
                        context_dim=context_dim,
                        disable_self_attn=disable_middle_self_attn,
                        use_checkpoint=use_checkpoint,
                    ),
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=None,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    ),
                ]
            self.middle_block = TimestepEmbedSequential1(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = model_channels * mult
                num_transformers = transformer_depth_output.pop()
                if num_transformers > 0:
                    dim_head = ch // num_heads
                    disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            get_attention_layer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=num_transformers,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                        if resblock_updown
                        else Upsample1(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            dtype=self.dtype,
                            device=device,
                            operations=operations,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential1(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            operations.GroupNorm(32, ch, dtype=self.dtype, device=device),
            nn.SiLU(),
            zero_module(
                operations.conv_nd(
                    dims,
                    model_channels,
                    out_channels,
                    3,
                    padding=1,
                    dtype=self.dtype,
                    device=device,
                )
            ),
        )

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False
        ).to(x.dtype)
        emb = self.time_embed(t_emb)
        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control1(h, control, "input")
            hs.append(h)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed1(
                self.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control1(h, control, "middle")

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control1(hsp, control, "output")

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed1(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        return self.out(h)


from typing import Union

import torch
from einops import rearrange

ae_ops = disable_weight_init
from enum import Enum


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5


def model_sampling(model_config, model_type):
    s = ModelSamplingDiscrete
    if model_type == ModelType.EPS:
        c = EPS

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel1
    ):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype

        if not unet_config.get("disable_unet_model_creation", False):
            if self.manual_cast_dtype is not None:
                operations = manual_cast
            else:
                operations = disable_weight_init
            self.diffusion_model = unet_model(
                **unet_config, device=device, operations=operations
            )
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))

    def apply_model(
        self,
        x,
        t,
        c_concat=None,
        c_crossattn=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        context = c_crossattn
        dtype = self.get_dtype()

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            extra_conds[o] = extra

        model_output = self.diffusion_model(
            xc,
            t,
            context=context,
            control=control,
            transformer_options=transformer_options,
            **extra_conds,
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        out["c_crossattn"] = CONDCrossAttn(cross_attn)
        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def memory_required(self, input_shape):
        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        area = input_shape[0] * input_shape[2] * input_shape[3]
        return (area * dtype_size(dtype) / 50) * (1024 * 1024)


class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}


class BASE:
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    manual_cast_dtype = None

    @classmethod
    def matches(s, unet_config, state_dict=None):
        for k in s.unet_config:
            if k not in unet_config or s.unet_config[k] != unet_config[k]:
                return False
        return True

    def model_type(self, state_dict, prefix=""):
        return ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        out = BaseModel(
            self, model_type=self.model_type(state_dict, prefix), device=device
        )
        return out

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


class sm_SD15(BASE):
    unet_config = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = SD15

    def process_clip_state_dict(self, state_dict):
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                "cond_stage_model.transformer.text_model."
            ):
                y = x.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                state_dict[y] = state_dict.pop(x)

        if (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
            in state_dict
        ):
            ids = state_dict[
                "cond_stage_model.transformer.text_model.embeddings.position_ids"
            ]
            if ids.dtype == torch.float32:
                state_dict[
                    "cond_stage_model.transformer.text_model.embeddings.position_ids"
                ] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )
        return state_dict

    def clip_target(self):
        return ClipTarget(SD1Tokenizer, SD1ClipModel)


models = [
    sm_SD15,
]


def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(
        list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))
    )
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(
            state_dict_keys, transformer_prefix + "{}"
        )
        context_dim = state_dict[
            "{}0.attn2.to_k.weight".format(transformer_prefix)
        ].shape[1]
        use_linear_in_transformer = (
            len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
        )
        time_stack = (
            "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict
            or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
        )
        return (
            last_transformer_depth,
            context_dim,
            use_linear_in_transformer,
            time_stack,
        )
    return None


def detect_unet_config(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False,
    }

    y_input = "{}label_emb.0.0.weight".format(key_prefix)
    unet_config["adm_in_channels"] = None

    model_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[0]
    in_channels = state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[1]

    out_key = "{}out.2.weight".format(key_prefix)
    out_channels = state_dict[out_key].shape[0]

    num_res_blocks = []
    channel_mult = []
    attention_resolutions = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(
        state_dict_keys, "{}input_blocks".format(key_prefix) + ".{}."
    )
    for count in range(input_block_count):
        prefix = "{}input_blocks.{}.".format(key_prefix, count)
        prefix_output = "{}output_blocks.{}.".format(
            key_prefix, input_block_count - count - 1
        )

        block_keys = sorted(
            list(filter(lambda a: a.startswith(prefix), state_dict_keys))
        )

        block_keys_output = sorted(
            list(filter(lambda a: a.startswith(prefix_output), state_dict_keys))
        )

        if "{}0.op.weight".format(prefix) in block_keys:  # new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(
                prefix_output, state_dict_keys, state_dict
            )
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = (
                    state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0]
                    // model_channels
                )

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(
                    prefix_output, state_dict_keys, state_dict
                )
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(
            state_dict_keys,
            "{}middle_block.1.transformer_blocks.".format(key_prefix) + "{}",
        )

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config["use_linear_in_transformer"] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    unet_config["use_temporal_resblock"] = False
    unet_config["use_temporal_attention"] = False

    return unet_config


def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)


def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    model_config = model_config_from_unet_config(unet_config, state_dict)
    return model_config


import os
from enum import Enum

import torch


def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = load_lora(lora, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)

    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)

    return (new_modelpatcher, new_clip)


class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = text_encoder_device()
        offload_device = text_encoder_offload_device()
        params["device"] = offload_device
        params["dtype"] = text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.layer_idx = None

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        self.cond_stage_model.reset_clip_options()
        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})
        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})
        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def load_sd(self, sd, full_model=False):
        return self.cond_stage_model.load_state_dict(sd, strict=False)

    def load_model(self):
        load_model_gpu(self.patcher)
        return self.patcher


class VAE:
    def __init__(self, sd=None, device=None, config=None, dtype=None):
        self.memory_used_encode = lambda shape, dtype: (
            1767 * shape[2] * shape[3]
        ) * dtype_size(
            dtype
        )  # These are for AutoencoderKL and need tweaking (should be lower)
        self.memory_used_decode = lambda shape, dtype: (
            2178 * shape[2] * shape[3] * 64
        ) * dtype_size(dtype)
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.latent_channels = 4
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        if config is None:
            config = {
                "encoder": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "decoder": {
                    "double_z": True,
                    "z_channels": 4,
                    "resolution": 256,
                    "in_channels": 3,
                    "out_ch": 3,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                },
                "regularizer": {"sample": True},
            }
            self.first_stage_model = AutoencodingEngine(
                Encoder(**config["encoder"]),
                Decoder(**config["decoder"]),
                DiagonalGaussianRegularizer(**config["regularizer"]),
            )
        self.first_stage_model = self.first_stage_model.eval()

        self.first_stage_model.load_state_dict(sd, strict=False)

        if device is None:
            device = vae_device()
        self.device = device
        offload_device = vae_offload_device()
        if dtype is None:
            dtype = vae_dtype()
        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
        y = (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
        return pixels

    def decode(self, samples_in):
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)

        pixel_samples = torch.empty(
            (
                samples_in.shape[0],
                3,
                round(samples_in.shape[2] * self.upscale_ratio),
                round(samples_in.shape[3] * self.upscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, samples_in.shape[0], batch_number):
            samples = (
                samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
            )
            pixel_samples[x : x + batch_number] = self.process_output(
                self.first_stage_model.decode(samples).to(self.output_device).float()
            )
        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def encode(self, pixel_samples):
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
        pixel_samples = pixel_samples.movedim(-1, 1)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        load_models_gpu([self.patcher], memory_required=memory_used)
        free_memory = get_free_memory(self.device)
        batch_number = int(free_memory / memory_used)
        batch_number = max(1, batch_number)
        samples = torch.empty(
            (
                pixel_samples.shape[0],
                self.latent_channels,
                round(pixel_samples.shape[2] // self.downscale_ratio),
                round(pixel_samples.shape[3] // self.downscale_ratio),
            ),
            device=self.output_device,
        )
        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = (
                self.process_input(pixel_samples[x : x + batch_number])
                .to(self.vae_dtype)
                .to(self.device)
            )
            samples[x : x + batch_number] = (
                self.first_stage_model.encode(pixels_in).to(self.output_device).float()
            )

        return samples


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2


def unet_dtype1(
    device=None,
    model_params=0,
    supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
):
    return torch.float16


def load_checkpoint_guess_config(
    ckpt_path,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
):
    sd = load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    load_device = get_torch_device()

    model_config = model_config_from_unet(sd, "model.diffusion_model.")
    unet_dtype = unet_dtype1(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
    )
    manual_cast_dtype = unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if output_model:
        inital_load_device = unet_inital_load_device(parameters, unet_dtype)
        offload_device = unet_offload_device()
        model = model_config.get_model(
            sd, "model.diffusion_model.", device=inital_load_device
        )
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
        )
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(
                        filter(
                            lambda a: ".logit_scale" not in a
                            and ".transformer.text_projection.weight" not in a,
                            m,
                        )
                    )
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = ModelPatcher(
            model,
            load_device=load_device,
            offload_device=unet_offload_device(),
            current_device=inital_load_device,
        )
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)


def get_output_directory():
    global output_directory
    return output_directory


def get_full_path(folder_name, filename):
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)
    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


MAX_RESOLUTION = 16384


class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class VAEDecode:
    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]),)


class VAEEncode:
    def encode(self, vae, pixels):
        t = vae.encode(pixels[:, :, :, :3])
        return ({"samples": t},)


class CheckpointLoaderSimple:
    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = f"{ckpt_name}"
        out = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory="./_internal/embeddings/",
        )
        print("loading", ckpt_path)
        return out[:3]


class CLIPSetLastLayer:
    def set_last_layer(self, clip, stop_at_clip_layer):
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)


class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_path = get_full_path("loras", lora_name)
        lora = None
        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


class EmptyLatentImage:
    def __init__(self):
        self.device = intermediate_device()

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)


class LatentUpscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    def upscale(self, samples, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()
            width = max(64, width)
            height = max(64, height)

            s["samples"] = common_upscale(
                samples["samples"], width // 8, height // 8, upscale_method, crop
            )
        return (s,)


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None

    disable_pbar = not PROGRESS_BAR_ENABLED
    samples = sample1(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out,)


class KSampler2:
    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )


class SaveImage:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_images(
        self, images, filename_prefix="LD", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


def act(act_type: str, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = nn.LeakyReLU(neg_slope, inplace)
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


ConvMode = Literal["CNA", "NAC", "CNAC"]


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type: str | None = "relu",
    mode: ConvMode = "CNA",
    c2x2=False,
):
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


class RRDB(nn.Module):

    def __init__(
        self,
        nf,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        _convtype="Conv2D",
        _spectral_norm=False,
        plus=False,
        c2x2=False,
    ):
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

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(
        self,
        nf=64,
        kernel_size=3,
        gc=32,
        stride=1,
        bias: bool = True,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode: ConvMode = "CNA",
        plus=False,
        c2x2=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1x1 = None

        self.conv1 = conv_block(
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
        self.conv2 = conv_block(
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
        self.conv3 = conv_block(
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
        self.conv4 = conv_block(
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
        self.conv5 = conv_block(
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

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def upconv_block(
    in_nc: int,
    out_nc: int,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type: str | None = None,
    act_type="relu",
    mode="nearest",
    c2x2=False,
):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
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


class RRDBNet(nn.Module):
    def __init__(
        self,
        state_dict,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: ConvMode = "CNA",
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
            "upconv": upconv_block,
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

        self.model = sequential(
            # fea conv
            conv_block(
                in_nc=self.in_nc,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
            ShortcutBlock(
                sequential(
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
                    conv_block(
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
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=self.act,
                c2x2=c2x2,
            ),
            # hr_conv1
            conv_block(
                in_nc=self.num_filters,
                out_nc=self.out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
                c2x2=c2x2,
            ),
        )

        self.load_state_dict(self.state, strict=False)

    def new_to_old_arch(self, state):
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
        def compare(item1, item2):
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
        n = 0
        for part in list(self.state):
            parts = part.split(".")[1:]
            if len(parts) == 2:
                part_num = int(parts[0])
                if part_num > min_part and parts[1] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
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

    def forward(self, x):
        return self.model(x)


PyTorchSRModels = (RRDBNet,)
PyTorchSRModel = Union[RRDBNet,]

PyTorchModels = (*PyTorchSRModels,)
PyTorchModel = Union[PyTorchSRModel]


class UnsupportedModel(Exception):
    pass


import logging as logger


def load_state_dict(state_dict) -> PyTorchModel:
    logger.debug(f"Loading state dict into pytorch model arch")
    state_dict_keys = list(state_dict.keys())
    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    model = RRDBNet(state_dict)
    return model


class UpscaleModelLoader:
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = f"_internal/ESRGAN/{model_name}"
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    pbar=None,
):
    output = torch.empty(
        (
            samples.shape[0],
            out_channels,
            round(samples.shape[2] * upscale_amount),
            round(samples.shape[3] * upscale_amount),
        ),
        device="cpu",
    )
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        out_div = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y : y + tile_y, x : x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                        1.0 / feather
                    ) * (t + 1)
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                        1.0 / feather
                    ) * (t + 1)
                out[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += (
                    ps * mask
                )
                out_div[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += mask

        output[b : b + 1] = out / out_div
    return output


class ImageUpscaleWithModel:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = torch.device(torch.cuda.current_device())
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        free_memory = get_free_memory(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            steps = in_img.shape[0] * get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = ProgressBar(steps)
            s = tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


def torch_gc():
    pass


def flatten(img, bgcolor):
    # Replace transparency with bgcolor
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert(
        "RGB"
    )


class Script:
    pass


class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
actual_upscaler = None

# Batch of images to upscale
batch = None

import numpy as np
import torch.nn.functional as F

BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor, batch_index=0):
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    """
    Expands a crop region to a specified target size.
    :param region: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param width: The width of the image the crop region is from.
    :param height: The height of the image the crop region is from.
    :param target_width: The desired width of the crop region.
    :param target_height: The desired height of the crop region.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1
    # target_width = math.ceil(actual_width / 8) * 8
    # target_height = math.ceil(actual_height / 8) * 8

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        cropped.append(n)
    return cropped


from PIL import Image

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def _upscale(self, img: Image, scale):
        global actual_upscaler
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()


from PIL import ImageFilter


class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:

    def __init__(
        self, p: StableDiffusionProcessing, images: list, seed: int, info: str
    ):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Setup
    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    x1, y1, x2, y2 = crop_region
    crop_width = x2 - x1
    crop_height = y2 - y1
    crop_ratio = crop_width / crop_height
    p_ratio = p.width / p.height
    if crop_ratio > p_ratio:
        target_width = crop_width
        target_height = round(crop_width / p_ratio)
    else:
        target_width = round(crop_height * p_ratio)
        target_height = crop_height
    crop_region, _ = expand_crop(
        crop_region,
        image_mask.width,
        image_mask.height,
        target_width,
        target_height,
    )
    tile_size = p.width, p.height

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    global batch
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(
        p.positive, crop_region, p.init_size, init_image.size, tile_size
    )
    negative_cropped = crop_cond(
        p.negative, crop_region, p.init_size, init_image.size, tile_size
    )

    # Encode the image
    vae_encoder = VAEEncode()
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    (samples,) = common_ksampler(
        p.model,
        p.seed,
        p.steps,
        p.cfg,
        p.sampler_name,
        p.scheduler,
        positive_cropped,
        negative_cropped,
        latent,
        denoise=p.denoise,
    )

    # Decode the sample
    vae_decoder = VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(
                initial_tile_size, Image.Resampling.LANCZOS
            )

        # Put the tile into position
        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        image_mask = image_mask.resize(temp.size)
        temp.putalpha(image_mask)
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert("RGB")
        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed


def sample_custom(
    model,
    noise,
    cfg,
    sampler,
    sigmas,
    positive,
    negative,
    latent_image,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    samples = sample(
        model,
        noise,
        positive,
        negative,
        cfg,
        model.load_device,
        sampler,
        sigmas,
        model_options=model.model_options,
        latent_image=latent_image,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(intermediate_device())
    return samples


from enum import Enum

from PIL import ImageDraw


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler:

    def __init__(
        self,
        p,
        image,
        upscaler_index: int,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    ) -> None:
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(
            max(p.width, p.height) / max(image.width, image.height)
        )
        global sd_upscalers
        self.upscaler = sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(
                self.image, value, self.upscaler.data_path
            )
        # Resize image to set values
        self.image = self.image.resize(
            (self.p.width, self.p.height), resample=Image.LANCZOS
        )

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = (
            self.upscaler.name
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = (
            self.redraw.tile_width
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = (
            self.redraw.tile_height
        )
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = (
            self.p.mask_blur
        )
        self.p.extra_generation_params["Ultimate SD upscale padding"] = (
            self.redraw.padding
        )

    def process(self):
        global state
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
        state.end()


class USDURedraw:

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        return self.linear_process(p, image, rows, cols)


class USDUSeamsFix:

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        global state
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(
            gradient.resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        row_gradient.paste(
            gradient.rotate(180).resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, self.tile_height // 2),
        )
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(
            gradient.rotate(90).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        col_gradient.paste(
            gradient.rotate(270).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (self.tile_width // 2, 0),
        )

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    row_gradient,
                    (
                        xi * self.tile_width,
                        yi * self.tile_height + self.tile_height // 2,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    col_gradient,
                    (
                        xi * self.tile_width + self.tile_width // 2,
                        yi * self.tile_height,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        return self.half_tile_process(p, image, rows, cols)


class Script(Script):
    def run(
        self,
        p,
        _,
        tile_width,
        tile_height,
        mask_blur,
        padding,
        seams_fix_width,
        seams_fix_denoise,
        seams_fix_padding,
        upscaler_index,
        save_upscaled_image,
        redraw_mode,
        save_seams_fix_image,
        seams_fix_mask_blur,
        seams_fix_type,
        target_size_type,
        custom_width,
        custom_height,
        custom_scale,
    ):

        # Init
        fix_seed(p)
        torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        init_img = flatten(init_img, opts.img2img_background_color)

        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
        )
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(
            seams_fix_padding,
            seams_fix_denoise,
            seams_fix_mask_blur,
            seams_fix_width,
            seams_fix_type,
        )
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(
            p,
            result_images,
            seed,
            upscaler.initial_info if upscaler.initial_info is not None else "",
        )


# Make some patches to the script
import math

from PIL import Image

#
# Instead of using multiples of 64, use multiples of 8
#

# Upscaler
old_init = USDUpscaler.__init__


def new_init(
    self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height
):
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(
        self,
        p,
        image,
        upscaler_index,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    )


USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

#
# Make the script upscale on a batch of images instead of one image
#

old_upscale = USDUpscaler.upscale


def new_upscale(self):
    old_upscale(self)
    global batch
    batch = [self.image] + [
        img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        for img in batch[1:]
    ]


USDUpscaler.upscale = new_upscale
MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


class UltimateSDUpscale:
    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
    ):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        actual_upscaler = upscale_model

        # Set the batch of images
        batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image),
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_by,
            force_uniform_tiles,
        )

        #
        # Running the script
        #
        script = Script()
        script.run(
            p=sdprocessing,
            _=None,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            padding=tile_padding,
            seams_fix_width=seam_fix_width,
            seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding,
            upscaler_index=0,
            save_upscaled_image=False,
            redraw_mode=MODES[mode_type],
            save_seams_fix_image=False,
            seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
            target_size_type=2,
            custom_width=None,
            custom_height=None,
            custom_scale=upscale_by,
        )

        # Return the resulting images
        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


from collections import namedtuple
import math
import os
import re
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, box=box
    )

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index


class SafeToGPU:
    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device):
        if is_same_device(device, "cpu"):
            obj.to(device)
        else:
            if is_same_device(obj.device, "cpu"):  # cpu to gpu
                free_memory(self.size * 1.3, device)
                if get_free_memory(device) > self.size * 1.3:
                    try:
                        obj.to(device)
                    except:
                        print(
                            f"WARN: The model is not moved to the '{device}' due to insufficient memory. [1]"
                        )
                else:
                    print(
                        f"WARN: The model is not moved to the '{device}' due to insufficient memory. [2]"
                    )


class SAMWrapper:
    def __init__(self, model, is_auto_mode, safe_to_gpu=None):
        self.model = model
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU_stub()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        if self.is_auto_mode:
            device = get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)


class SAMLoader:
    def load_model(self, model_name, device_mode="auto"):
        modelname = "./_internal/yolos/" + model_name

        if "vit_h" in model_name:
            model_kind = "vit_h"
        elif "vit_l" in model_name:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        size = os.path.getsize(modelname)
        safe_to = SafeToGPU(size)

        # Unless user explicitly wants to use CPU, we use GPU
        device = get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            safe_to.to_device(sam, device)

        is_auto_mode = device_mode == "AUTO"

        sam_obj = SAMWrapper(sam, is_auto_mode=is_auto_mode, safe_to_gpu=safe_to)
        sam.sam_wrapper = sam_obj

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam,)


from PIL import Image
import cv2
import numpy as np
import torch
from PIL import ImageTk

orig_torch_load = torch.load

from ultralytics import YOLO

# HOTFIX: https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/754
# importing YOLO breaking original torch.load capabilities
torch.load = orig_torch_load


def load_yolo(model_path: str):
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        print("please download yolo model")


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def _tensor_check_image(image):
    return


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def dilate_masks(segmasks, dilation_factor, iter=1):
    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


crop_tensor4 = crop_ndarray4


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)


SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(
        self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None
    ):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if (
                x2 - x1 > drop_size and y2 - y1 > drop_size
            ):  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(
                    cropped_image,
                    cropped_mask,
                    confidence,
                    crop_region,
                    item_bbox,
                    label,
                    None,
                )

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        return segs


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model


class NO_SEGM_DETECTOR:
    pass


class UltralyticsDetectorProvider:
    def doit(self, model_name):
        model = load_yolo("./_internal/yolos/" + model_name)
        return UltraBBoxDetector(model), UltraSegmDetector(model)


class SEGSLabelFilter:
    @staticmethod
    def filter(segs, labels):
        labels = set([label.strip() for label in labels])
        return (
            segs,
            (segs[0], []),
        )


class BboxDetectorForEach:
    def doit(
        self,
        bbox_detector,
        image,
        threshold,
        dilation,
        crop_factor,
        drop_size,
        labels=None,
        detailer_hook=None,
    ):
        segs = bbox_detector.detect(
            image, threshold, dilation, crop_factor, drop_size, detailer_hook
        )

        if labels is not None and labels != "":
            labels = labels.split(",")
            if len(labels) > 0:
                segs, _ = SEGSLabelFilter.filter(segs, labels)

        return (segs,)


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask


def combine_masks2(masks):
    mask = torch.from_numpy(np.array(masks[0]).astype(np.uint8))
    return mask


def dilate_mask(mask, dilation_factor, iter=1):
    return make_2d_mask(mask)


def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask


def make_sam_mask(
    sam,
    segs,
    image,
    detection_hint,
    dilation,
    threshold,
    bbox_expansion,
    mask_hint_threshold,
    mask_hint_use_negative,
):
    sam_obj = sam.sam_wrapper
    sam_obj.prepare_device()

    try:
        image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []
        # seg_shape = segs[0]
        segs = segs[1]
        for i in range(len(segs)):
            bbox = segs[i].bbox
            center = center_of_bbox(bbox)
            x1 = max(bbox[0] - bbox_expansion, 0)
            y1 = max(bbox[1] - bbox_expansion, 0)
            x2 = min(bbox[2] + bbox_expansion, image.shape[1])
            y2 = min(bbox[3] + bbox_expansion, image.shape[0])
            dilated_bbox = [x1, y1, x2, y2]
            points = []
            plabs = []
            points.append(center)
            plabs = [1]  # 1 = foreground point, 0 = background point
            detected_masks = sam_obj.predict(
                image, points, plabs, dilated_bbox, threshold
            )
            total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    mask = mask.float()
    mask = dilate_mask(mask.cpu().numpy(), dilation)
    mask = torch.from_numpy(mask)

    mask = make_3d_mask(mask)
    return mask


class SAMDetectorCombined:
    def doit(
        self,
        sam_model,
        segs,
        image,
        detection_hint,
        dilation,
        threshold,
        bbox_expansion,
        mask_hint_threshold,
        mask_hint_use_negative,
    ):
        return (
            make_sam_mask(
                sam_model,
                segs,
                image,
                detection_hint,
                dilation,
                threshold,
                bbox_expansion,
                mask_hint_threshold,
                mask_hint_use_negative,
            ),
        )


def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)
    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[
            crop_region[1] : crop_region[3], crop_region[0] : crop_region[2]
        ]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(
            seg.cropped_image,
            new_mask,
            seg.confidence,
            seg.crop_region,
            seg.bbox,
            seg.label,
            None,
        )
        items.append(item)

    return segs[0], items


class SegsBitwiseAndMask:
    def doit(self, segs, mask):
        return (segs_bitwise_and_mask(segs, mask),)


def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        self.tensor = new_tensor


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)


def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs


def starts_with_regex(pattern, text):
    regex = re.compile(pattern)
    return regex.match(text)


class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        item = self.items[self.i]
        self.i += 1

        return item


def process_wildcard_for_segs(wildcard):
    return None, WildcardChooser([(None, wildcard)], False)


class DifferentialDiffusion:
    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(
        self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict
    ):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


def to_tensor(image):
    return torch.from_numpy(image)


import torchvision


def _tensor_check_mask(mask):
    return


def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]

    _tensor_check_mask(mask)

    kernel_size = kernel_size * 2 + 1

    prev_device = mask.device
    device = get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    )(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask


def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    return VAEEncode().encode(vae, pixels)[0]


def calculate_sigmas2(model, sampler, scheduler, steps):
    return calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs["extra_args"].get("seed", None)
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler2(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_2m_sde":

        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs["noise_sampler"] = noise_sampler

            return sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    else:
        return sampler_object(sampler_name)

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return prepare_noise(latent_image, self.seed, batch_inds)


def sample_with_custom_noise(
    model,
    add_noise,
    noise_seed,
    cfg,
    positive,
    negative,
    sampler,
    sigmas,
    latent_image,
    noise=None,
    callback=None,
):
    latent = latent_image
    latent_image = latent["samples"]

    out = latent.copy()
    out["samples"] = latent_image

    if noise is None:
        noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}

    disable_pbar = not PROGRESS_BAR_ENABLED

    device = get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    samples = sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        disable_pbar=disable_pbar,
        seed=noise_seed,
    )

    samples = samples.to(intermediate_device())

    out["samples"] = samples
    out_denoised = out
    return out, out_denoised


def separated_sample(
    model,
    add_noise,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    start_at_step,
    end_at_step,
    return_with_leftover_noise,
    sigma_ratio=1.0,
    sampler_opt=None,
    noise=None,
    callback=None,
    scheduler_func=None,
):

    total_sigmas = calculate_sigmas2(model, sampler_name, scheduler, steps)

    sigmas = total_sigmas

    if start_at_step is not None:
        sigmas = sigmas[start_at_step:] * sigma_ratio

    impact_sampler = ksampler2(sampler_name, total_sigmas)

    res = sample_with_custom_noise(
        model,
        add_noise,
        seed,
        cfg,
        positive,
        negative,
        impact_sampler,
        sigmas,
        latent_image,
        noise=noise,
        callback=callback,
    )

    return res[1]


def ksampler_wrapper(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    refiner_ratio=None,
    refiner_model=None,
    refiner_clip=None,
    refiner_positive=None,
    refiner_negative=None,
    sigma_factor=1.0,
    noise=None,
    scheduler_func=None,
):

    # Use separated_sample instead of KSampler for `AYS scheduler`
    # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    refined_latent = separated_sample(
        model,
        True,
        seed,
        advanced_steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        False,
        sigma_ratio=sigma_factor,
        noise=noise,
        scheduler_func=scheduler_func,
    )

    return refined_latent


def enhance_detail(
    image,
    model,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    bbox,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    denoise,
    noise_mask,
    force_inpaint,
    wildcard_opt=None,
    wildcard_opt_concat_mode=None,
    detailer_hook=None,
    refiner_ratio=None,
    refiner_model=None,
    refiner_clip=None,
    refiner_positive=None,
    refiner_negative=None,
    control_net_wrapper=None,
    cycle=1,
    inpaint_model=False,
    noise_mask_feather=0,
    scheduler_func=None,
):

    if noise_mask is not None:
        noise_mask = tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # for cropped_size
    upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        print(f"Detailer: force inpaint")
        upscale = 1.0
        new_w = w
        new_h = h

    print(
        f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}"
    )

    # upscale
    upscaled_image = tensor_resize(image, new_w, new_h)

    cnet_pils = None

    # prepare mask
    latent_image = to_latent_image(upscaled_image, vae)
    if noise_mask is not None:
        latent_image["noise_mask"] = noise_mask

    refined_latent = latent_image

    # ksampler
    for i in range(0, cycle):
        (
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            upscaled_latent2,
            denoise2,
        ) = (
            model,
            seed + i,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )
        noise = None

        refined_latent = ksampler_wrapper(
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            refined_latent,
            denoise2,
            refiner_ratio,
            refiner_model,
            refiner_clip,
            refiner_positive,
            refiner_negative,
            noise=noise,
            scheduler_func=scheduler_func,
        )

    # non-latent downscale - latent downscale cause bad quality
    try:
        # try to decode image normally
        refined_image = vae.decode(refined_latent["samples"])
    except Exception as e:
        # usually an out-of-memory exception from the decode, so try a tiled approach
        refined_image = vae.decode_tiled(
            refined_latent["samples"],
            tile_x=64,
            tile_y=64,
        )

    # downscale
    refined_image = tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pils


def tensor_paste(image1, image2, left_top, mask):
    """Mask and image2 has to be the same size"""
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    mask = mask[:, :h, :w, :]
    image1[:, y : y + h, x : x + w, :] = (1 - mask) * image1[
        :, y : y + h, x : x + w, :
    ] + mask * image2[:, :h, :w, :]
    return


def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    alpha = torch.ones((*image.shape[:-1], 1))
    return torch.cat((image, alpha), axis=-1)


def tensor_convert_rgb(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    return image


def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


class DetailerForEach:
    @staticmethod
    def do_detail(
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for_bbox,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard_opt=None,
        detailer_hook=None,
        refiner_ratio=None,
        refiner_model=None,
        refiner_clip=None,
        refiner_positive=None,
        refiner_negative=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        wmode, wildcard_chooser = process_wildcard_for_segs(wildcard_opt)

        ordered_segs = segs[1]

        if (
            noise_mask_feather > 0
            and "denoise_mask_function" not in model.model_options
        ):
            model = DifferentialDiffusion().apply(model)[0]

        for i, seg in enumerate(ordered_segs):
            cropped_image = crop_ndarray4(
                image.cpu().numpy(), seg.crop_region
            )  # Never use seg.cropped_image to handle overlapping area
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print(f"Detailer: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask

            seg_seed, wildcard_item = wildcard_chooser.get(seg)

            seg_seed = seed + i if seg_seed is None else seg_seed

            cropped_positive = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]

            cropped_negative = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]

            orig_cropped_image = cropped_image.clone()
            enhanced_image, cnet_pils = enhance_detail(
                cropped_image,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for_bbox,
                max_size,
                seg.bbox,
                seg_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                cropped_positive,
                cropped_negative,
                denoise,
                cropped_mask,
                force_inpaint,
                wildcard_opt=wildcard_item,
                wildcard_opt_concat_mode=wildcard_concat_mode,
                detailer_hook=detailer_hook,
                refiner_ratio=refiner_ratio,
                refiner_model=refiner_model,
                refiner_clip=refiner_clip,
                refiner_positive=refiner_positive,
                refiner_negative=refiner_negative,
                control_net_wrapper=seg.control_net_wrapper,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func=scheduler_func_opt,
            )

            if not (enhanced_image is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                tensor_paste(
                    image,
                    enhanced_image,
                    (seg.crop_region[0], seg.crop_region[1]),
                    mask,
                )  # this code affecting to `cropped_image`.
                enhanced_list.append(enhanced_image)

            # Convert enhanced_pil_alpha to RGBA mode
            enhanced_image_alpha = tensor_convert_rgba(enhanced_image)
            new_seg_image = (
                enhanced_image.numpy()
            )  # alpha should not be applied to seg_image
            # Apply the mask
            mask = tensor_resize(mask, *tensor_get_size(enhanced_image))
            tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)

            cropped_list.append(orig_cropped_image)  # NOTE: Don't use `cropped_image`

            new_seg = SEG(
                new_seg_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        image_tensor = tensor_convert_rgb(image)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return (
            image_tensor,
            cropped_list,
            enhanced_list,
            enhanced_alpha_list,
            cnet_pil_list,
            (segs[0], new_segs),
        )


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


class DetailerForEachTest(DetailerForEach):
    def doit(
        self,
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard,
        detailer_hook=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):

        (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
            new_segs,
        ) = DetailerForEach.do_detail(
            image,
            segs,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            feather,
            noise_mask,
            force_inpaint,
            wildcard,
            detailer_hook,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func_opt=scheduler_func_opt,
        )

        cnet_pil_list = [empty_pil_tensor()]

        return (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
        )


import contextlib
import functools
import logging
from dataclasses import dataclass

import torch

try:
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
    from sfast.compilers.diffusion_pipeline_compiler import (
        _enable_xformers,
        _modify_model,
    )
    from sfast.cuda.graphs import make_dynamic_graphed_callable
    from sfast.jit import utils as jit_utils
    from sfast.jit.trace_helper import trace_with_kwargs
except:
    pass


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)


class ModuleFactory:
    def get_converted_kwargs(self):
        return self.converted_kwargs


import torch as th
import torch.nn as nn
import copy


class BaseModelApplyModelModule(torch.nn.Module):
    def __init__(self, func, module):
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x,
        timestep,
        c_concat=None,
        c_crossattn=None,
        y=None,
        control=None,
        transformer_options={},
    ):
        kwargs = {"y": y}

        new_transformer_options = {}

        return self.func(
            input_x,
            timestep,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=new_transformer_options,
            **kwargs,
        )


class BaseModelApplyModelModuleFactory(ModuleFactory):
    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable, kwargs) -> None:
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self):
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            key_kwargs[k] = v

        patch_module_cache_key = {}
        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.unet_config),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)


logger = logging.getLogger()


@dataclass
class TracedModuleCacheItem:
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    traced_modules = {}

    def __init__(self, config=None, patch_id=None, **kwargs_) -> None:
        self.config = config
        self.patch_id = patch_id
        self.kwargs_ = kwargs_
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )
        self.cuda_graph_modules = {}

    def ts_compiler(
        self,
        m,
    ):
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                # raw freeze causes Tensor reference leak
                # because the constant Tensors in the GraphFunction of
                # the compilation unit are never freed.
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function, /, **kwargs):
        module_factory = BaseModelApplyModelModuleFactory(model_function, kwargs)
        kwargs = module_factory.get_converted_kwargs()
        key = module_factory.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                logger.info(
                    f'Tracing {getattr(m_model, "__name__", m_model.__class__.__name__)}'
                )
                traced_m, call_helper = trace_with_kwargs(
                    m_model, None, m_kwargs, **self.kwargs_
                )

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            self.cuda_graph_modules[key] = traced_module

        return traced_module(**kwargs)


def build_lazy_trace_module(config, device, patch_id):
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )


def gen_stable_fast_config():
    config = CompilationConfig.Default()
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = False
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device):
        if type(device) == torch.device:
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disable enable_cuda_graph.\33[0m"
                    )
        return self


class ApplyStableFastUnet:
    def apply_stable_fast(self, model, enable_cuda_graph):
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)


def enhance_prompt(p=None):
    prompt, neg, width, height, cfg = load_parameters_from_file()
    if p is None:
        pass
    else:
        prompt = p
    print(prompt)
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"""Your goal is to generate a text-to-image prompt based on a user's input, detailing their desired final outcome for an image. The user will provide specific details about the characteristics, features, or elements they want the image to include. The prompt should guide the generation of an image that aligns with the user's desired outcome.

                        Generate a text-to-image prompt by arranging the following blocks in a single string, separated by commas:

                        Image Type: [Specify desired image type]

                        Aesthetic or Mood: [Describe desired aesthetic or mood]

                        Lighting Conditions: [Specify desired lighting conditions]

                        Composition or Framing: [Provide details about desired composition or framing]

                        Background: [Specify desired background elements or setting]

                        Colors: [Mention any specific colors or color palette]

                        Objects or Elements: [List specific objects or features]

                        Style or Artistic Influence: [Mention desired artistic style or influence]

                        Subject's Appearance: [Describe appearance of main subject]

                        Ensure the blocks are arranged in order of visual importance, from the most significant to the least significant, to effectively guide image generation, a block can be surrounded by parentheses to gain additionnal significance.

                        This is an example of a user's input: "a beautiful blonde lady in lingerie sitting in seiza in a seducing way with a focus on her assets"

                        And this is an example of a desired output: "portrait| serene and mysterious| soft, diffused lighting| close-up shot, emphasizing facial features| simple and blurred background| earthy tones with a hint of warm highlights| renaissance painting| a beautiful lady with freckles and dark makeup"
                        
                        Here is the user's input: {prompt}

                        Write the prompt in the same style as the example above, in a single line , with absolutely no additional information, words or symbols other than the enhanced prompt.

                        Output:""",
            },
        ],
    )
    print("here's the enhanced prompt :", response["message"]["content"])
    return response["message"]["content"]


def pipeline(prompt, w, h):
    ckpt = "./_internal/checkpoints/meinamix_meinaV11.safetensors"
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt
        )
        cliptextencode = CLIPTextEncode()
        emptylatentimage = EmptyLatentImage()
        ksampler_instance = KSampler2()
        vaedecode = VAEDecode()
        saveimage = SaveImage()
        latent_upscale = LatentUpscale()
        upscalemodelloader = UpscaleModelLoader()
        ultimatesdupscale = UltimateSDUpscale()
    prompt = enhance_prompt(prompt)
    while prompt == None:
        pass
    with torch.inference_mode():
        try:
            loraloader = LoraLoader()
            loraloader_274 = loraloader.load_lora(
                lora_name="add_detail.safetensors",
                strength_model=0.7,
                strength_clip=0.7,
                model=checkpointloadersimple_241[0],
                clip=checkpointloadersimple_241[1],
            )
            print("loading add_detail.safetensors")
        except:
            loraloader_274 = checkpointloadersimple_241
        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2, clip=loraloader_274[1]
        )
        applystablefast_158 = loraloader_274
        cliptextencode_242 = cliptextencode.encode(
            text=prompt,
            clip=clipsetlastlayer_257[0],
        )
        cliptextencode_243 = cliptextencode.encode(
            text="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)",
            clip=clipsetlastlayer_257[0],
        )
        emptylatentimage_244 = emptylatentimage.generate(
            width=w, height=h, batch_size=1
        )
        ksampler_239 = ksampler_instance.sample(
            seed=random.randint(1, 2**64),
            steps=40,
            cfg=7,
            sampler_name="dpm_adaptive",
            scheduler="karras",
            denoise=1,
            model=applystablefast_158[0],
            positive=cliptextencode_242[0],
            negative=cliptextencode_243[0],
            latent_image=emptylatentimage_244[0],
        )
        latentupscale_254 = latent_upscale.upscale(
            upscale_method="bislerp",
            width=w * 2,
            height=h * 2,
            crop="disabled",
            samples=ksampler_239[0],
        )
        ksampler_253 = ksampler_instance.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=8,
            sampler_name="euler_ancestral",
            scheduler="normal",
            denoise=0.45,
            model=applystablefast_158[0],
            positive=cliptextencode_242[0],
            negative=cliptextencode_243[0],
            latent_image=latentupscale_254[0],
        )
        vaedecode_240 = vaedecode.decode(
            samples=ksampler_253[0],
            vae=checkpointloadersimple_241[2],
        )
        saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
        for image in vaedecode_240[0]:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))


def write_parameters_to_file(prompt_entry, neg, width, height, cfg):
    with open("./_internal/prompt.txt", "w") as f:
        f.write(f"prompt: {prompt_entry}")
        f.write(f"neg: {neg}")
        f.write(f"w: {int(width)}\n")
        f.write(f"h: {int(height)}\n")
        f.write(f"cfg: {int(cfg)}\n")


def load_parameters_from_file():
    with open("./_internal/prompt.txt", "r") as f:
        lines = f.readlines()
        parameters = {}
        for line in lines:
            # Skip empty lines
            if line.strip() == "":
                continue
            key, value = line.split(": ")
            parameters[key] = value.strip()
        prompt = parameters["prompt"]
        neg = parameters["neg"]
        width = int(parameters["w"])
        height = int(parameters["h"])
        cfg = int(parameters["cfg"])
    return prompt, neg, width, height, cfg


files = glob.glob("./_internal/checkpoints/*.safetensors")
loras = glob.glob("./_internal/loras/*.safetensors")
loras += glob.glob("./_internal/loras/*.pt")
generated = 0

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("LightDiffusion")
        self.geometry("800x700")
        
        file_names = [os.path.basename(file) for file in files]
        lora_names = [os.path.basename(lora) for lora in loras]

        selected_file = tk.StringVar()
        selected_lora = tk.StringVar()
        if file_names:
            selected_file.set(file_names[0])
        if lora_names:
            selected_lora.set(lora_names[0])

        # Create a frame for the sidebar
        self.sidebar = tk.Frame(self, width=300, bg="black")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Text input for the prompt
        self.prompt_entry = ctk.CTkTextbox(self.sidebar, height=200, width=300)
        self.prompt_entry.pack(pady=10, padx=10)

        self.neg = ctk.CTkTextbox(self.sidebar, height=50, width=300)
        self.neg.pack(pady=10, padx=10)

        self.dropdown = ctk.CTkOptionMenu(self.sidebar, values=file_names)
        self.dropdown.pack()

        self.lora_selection = ctk.CTkOptionMenu(self.sidebar, values=lora_names)
        self.lora_selection.pack(pady=10)

        # Sliders for the resolution
        self.width_label = ctk.CTkLabel(self.sidebar, text="")
        self.width_label.pack()
        self.width_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=2048, number_of_steps=16
        )
        self.width_slider.pack()

        self.height_label = ctk.CTkLabel(self.sidebar, text="")
        self.height_label.pack()
        self.height_slider = ctk.CTkSlider(
            self.sidebar,
            from_=1,
            to=2048,
            number_of_steps=16,
        )
        self.height_slider.pack()

        self.cfg_label = ctk.CTkLabel(self.sidebar, text="")
        self.cfg_label.pack()
        self.cfg_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=15, number_of_steps=14
        )
        self.cfg_slider.pack()

        # Create a frame for the checkboxes
        self.checkbox_frame = tk.Frame(self.sidebar, bg="black")
        self.checkbox_frame.pack(pady=10)

        # checkbox for hiresfix
        self.hires_fix_var = tk.BooleanVar()
        self.hires_fix_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Hires Fix",
            variable=self.hires_fix_var,
            command=self.print_hires_fix,
        )
        self.hires_fix_checkbox.grid(row=0, column=0, padx=5, pady=5)

        # add a checkbox for Adetailer
        self.adetailer_var = tk.BooleanVar()
        self.adetailer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Adetailer",
            variable=self.adetailer_var,
            command=self.print_adetailer,
        )
        self.adetailer_checkbox.grid(row=0, column=1, padx=5, pady=5)

        # add a checkbox to enable stable-fast optimization
        self.stable_fast_var = tk.BooleanVar()
        self.stable_fast_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Stable Fast",
            variable=self.stable_fast_var,
        )
        self.stable_fast_checkbox.grid(row=1, column=0, padx=5, pady=5)

        # add a checkbox to enable prompt enhancer
        self.enhancer_var = tk.BooleanVar()
        self.enhancer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Prompt enhancer",
            variable=self.enhancer_var,
        )
        self.enhancer_checkbox.grid(row=1, column=1, padx=5, pady=5)

        # Button to launch the generation
        self.generate_button = ctk.CTkButton(
            self.sidebar, text="Generate", command=self.generate_image
        )
        self.generate_button.pack(pady=10)

        # Create a frame for the image display, without border
        self.display = tk.Frame(self, bg="black", border=0)
        self.display.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # centered Label to display the generated image
        self.image_label = tk.Label(self.display, bg="black")
        self.image_label.pack(expand=True, padx=10, pady=10)
        
        self.previewer_checkbox = ctk.CTkCheckBox(
            self.display, text="Previewer", variable=tk.BooleanVar())
        self.previewer_checkbox.pack(pady=10)

        self.ckpt = None

        # load the checkpoint on an another thread
        threading.Thread(target=self._prep, daemon=True).start()
        
        self.button_frame = tk.Frame(self.sidebar, bg="black")
        self.button_frame.pack(pady=10)

        # add an img2img button, the button opens the file selector, run img2img on the selected image
        self.img2img_button = ctk.CTkButton(
            self.button_frame, text="img2img", command=self.img2img
        )
        self.img2img_button.grid(row=0, column=0, padx=5)
        
        self.interrupt_flag = False
        
        self.interrupt_button = ctk.CTkButton(self.button_frame, text="Interrupt", command=self.interrupt_generation)
        self.interrupt_button.grid(row=0, column=1, padx=5)

        prompt, neg, width, height, cfg = load_parameters_from_file()
        self.prompt_entry.insert(tk.END, prompt)
        self.neg.insert(tk.END, neg)
        self.width_slider.set(width)
        self.height_slider.set(height)
        self.cfg_slider.set(cfg)

        self.width_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.height_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.cfg_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.update_labels()
        self.prompt_entry.bind(
            "<KeyRelease>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.neg.bind(
            "<KeyRelease>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.width_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.height_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.cfg_slider.bind(
            "<ButtonRelease-1>",
            lambda event: write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.bind("<Configure>", self.on_resize)
        self.display_most_recent_image_flag = False
        self.display_most_recent_image()
       

    def _img2img(self, file_path):
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        img = Image.open(file_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            try:
                loraloader = LoraLoader()
                loraloader_274 = loraloader.load_lora(
                    lora_name="add_detail.safetensors",
                    strength_model=2,
                    strength_clip=2,
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                )
            except:
                loraloader_274 = checkpointloadersimple_241

            if self.stable_fast_var.get() == True:
                try:
                    app.title("LigtDiffusion - Generating StableFast model")
                except:
                    pass
                applystablefast = ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                    enable_cuda_graph=False,
                    model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274

            clipsetlastlayer = CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )

            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            upscalemodelloader_244 = upscalemodelloader.load_model(
                "RealESRGAN_x4plus.pth"
            )
            try:
                app.title("LightDiffusion - Upscaling")
            except:
                pass
            ultimatesdupscale_250 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=6,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=0.3,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=16,
                tile_padding=32,
                seam_fix_mode="Half Tile",
                seam_fix_denoise=0.2,
                seam_fix_width=64,
                seam_fix_mask_blur=16,
                seam_fix_padding=32,
                force_uniform_tiles="enable",
                image=img_tensor,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                vae=checkpointloadersimple_241[2],
                upscale_model=upscalemodelloader_244[0],
            )
            saveimage.save_images(
                filename_prefix="LD-i2i",
                images=ultimatesdupscale_250[0],
            )
            for image in ultimatesdupscale_250[0]:
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img = img.resize((int(w / 2), int(h / 2)))
        img = ImageTk.PhotoImage(img)
        self.image_label.after(0, self._update_image_label, img)
        try:
            app.title("LightDiffusion")
        except:
            pass

    def img2img(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            threading.Thread(
                target=self._img2img, args=(file_path,), daemon=True
            ).start()

    def print_hires_fix(self):
        if self.hires_fix_var.get() == True:
            print("Hires fix is ON")
        else:
            print("Hires fix is OFF")
            
    def print_adetailer(self):
        if self.adetailer_var.get() == True:
            print("Adetailer is ON")
        else:
            print("Adetailer is OFF")

    def generate_image(self):
        threading.Thread(target=self._generate_image, daemon=True).start()

    def _prep(self):
        if self.dropdown.get() != self.ckpt:
            self.ckpt = self.dropdown.get()
            with torch.inference_mode():
                self.checkpointloadersimple = CheckpointLoaderSimple()
                self.checkpointloadersimple_241 = (
                    self.checkpointloadersimple.load_checkpoint(ckpt_name="./_internal/checkpoints/" + self.ckpt)
                )
                self.cliptextencode = CLIPTextEncode()
                self.emptylatentimage = EmptyLatentImage()
                self.ksampler_instance = KSampler2()
                self.vaedecode = VAEDecode()
                self.saveimage = SaveImage()
                self.latent_upscale = LatentUpscale()
                self.upscalemodelloader = UpscaleModelLoader()
                self.ultimatesdupscale = UltimateSDUpscale()
        return (
            self.checkpointloadersimple_241,
            self.cliptextencode,
            self.emptylatentimage,
            self.ksampler_instance,
            self.vaedecode,
            self.saveimage,
            self.latent_upscale,
            self.upscalemodelloader,
            self.ultimatesdupscale,
        )

    def _generate_image(self):
        prompt = self.prompt_entry.get("1.0", tk.END)
        if self.enhancer_var.get() == True:
            prompt = enhance_prompt()
            while prompt == None:
                pass
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            try:
                loraloader = LoraLoader()
                loraloader_274 = loraloader.load_lora(
                    lora_name=self.lora_selection.get().replace(
                        "./_internal/loras/", ""
                    ),
                    strength_model=0.7,
                    strength_clip=0.7,
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                )
                print(
                    "loading",
                    self.lora_selection.get().replace("./_internal/loras/", ""),
                )
            except:
                loraloader_274 = checkpointloadersimple_241
            try:
                samloader = SAMLoader()
                samloader_87 = samloader.load_model(
                    model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                )

                cliptextencode_124 = cliptextencode.encode(
                    text="royal, detailed, magnificient, beautiful, seducing",
                    clip=loraloader_274[1],
                )

                ultralyticsdetectorprovider = UltralyticsDetectorProvider()
                ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                    # model_name="face_yolov8m.pt"
                    model_name="person_yolov8m-seg.pt"
                )

                bboxdetectorsegs = BboxDetectorForEach()
                samdetectorcombined = SAMDetectorCombined()
                impactsegsandmask = SegsBitwiseAndMask()
                detailerforeachdebug = DetailerForEachTest()
            except:
                pass
            clipsetlastlayer = CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )
            if self.stable_fast_var.get() == True:
                try:
                    self.title("LightDiffusion - Generating StableFast model")
                except:
                    pass
                applystablefast = ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                    enable_cuda_graph=False,
                    model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274
            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            emptylatentimage_244 = emptylatentimage.generate(
                width=w, height=h, batch_size=1
            )
            ksampler_239 = ksampler_instance.sample(
                seed=random.randint(1, 2**64),
                steps=40,
                cfg=cfg,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=emptylatentimage_244[0],
            )
            if self.hires_fix_var.get() == True:
                latentupscale_254 = latentupscale.upscale(
                    upscale_method="bislerp",
                    width=w * 2,
                    height=h * 2,
                    crop="disabled",
                    samples=ksampler_239[0],
                )
                ksampler_253 = ksampler_instance.sample(
                    seed=random.randint(1, 2**64),
                    steps=10,
                    cfg=8,
                    sampler_name="euler_ancestral",
                    scheduler="normal",
                    denoise=0.45,
                    model=applystablefast_158[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=latentupscale_254[0],
                )
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(filename_prefix="LD-HiresFix", images=vaedecode_240[0])
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            else:
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_239[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if self.adetailer_var.get() == True:
                bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                    threshold=0.5,
                    dilation=10,
                    crop_factor=2,
                    drop_size=10,
                    labels="all",
                    bbox_detector=ultralyticsdetectorprovider_151[0],
                    image=vaedecode_240[0],
                )
                samdetectorcombined_139 = samdetectorcombined.doit(
                    detection_hint="center-1",
                    dilation=0,
                    threshold=0.93,
                    bbox_expansion=0,
                    mask_hint_threshold=0.7,
                    mask_hint_use_negative="False",
                    sam_model=samloader_87[0],
                    segs=bboxdetectorsegs_132[0],
                    image=vaedecode_240[0],
                )
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132[0],
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
                    scheduler="karras",
                    denoise=0.5,
                    feather=5,
                    noise_mask=True,
                    force_inpaint=True,
                    wildcard="",
                    cycle=1,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    image=vaedecode_240[0],
                    segs=impactsegsandmask_152[0],
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                    vae=checkpointloadersimple_241[2],
                    positive=cliptextencode_124[0],
                    negative=cliptextencode_243[0],
                )
                saveimage_115 = saveimage.save_images(
                    filename_prefix="LD-refined",
                    images=detailerforeachdebug_145[0],
                )
                ultralyticsdetectorprovider = UltralyticsDetectorProvider()
                ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                    model_name="face_yolov9c.pt"
                )
                bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                    threshold=0.5,
                    dilation=10,
                    crop_factor=2,
                    drop_size=10,
                    labels="all",
                    bbox_detector=ultralyticsdetectorprovider_151[0],
                    image=detailerforeachdebug_145[0],
                )
                samdetectorcombined_139 = samdetectorcombined.doit(
                    detection_hint="center-1",
                    dilation=0,
                    threshold=0.93,
                    bbox_expansion=0,
                    mask_hint_threshold=0.7,
                    mask_hint_use_negative="False",
                    sam_model=samloader_87[0],
                    segs=bboxdetectorsegs_132[0],
                    image=detailerforeachdebug_145[0],
                )
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132[0],
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
                    scheduler="karras",
                    denoise=0.5,
                    feather=5,
                    noise_mask=True,
                    force_inpaint=True,
                    wildcard="",
                    cycle=1,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    image=detailerforeachdebug_145[0],
                    segs=impactsegsandmask_152[0],
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                    vae=checkpointloadersimple_241[2],
                    positive=cliptextencode_124[0],
                    negative=cliptextencode_243[0],
                )
                saveimage_115 = saveimage.save_images(
                    filename_prefix="lD-2ndrefined",
                    images=detailerforeachdebug_145[0],
                )
        self.update_image(img)
        global generated
        generated = img
        self.display_most_recent_image_flag = True
            

    def update_labels(self):
        self.width_label.configure(text=f"Width: {int(self.width_slider.get())}")
        self.height_label.configure(text=f"Height: {int(self.height_slider.get())}")
        self.cfg_label.configure(text=f"CFG: {int(self.cfg_slider.get())}")
        
    def update_image(self, img):
        global generated
        # Calculate the aspect ratio of the original image
        aspect_ratio = img.width / img.height

        # Determine the new dimensions while maintaining the aspect ratio
        label_width = int(4 * self.winfo_width() / 7)
        label_height = int(4 * self.winfo_height() / 7)

        if label_width / aspect_ratio <= label_height:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)

        # Resize the image to the new dimensions
        img = img.resize((new_width, new_height), Image.LANCZOS)
        self.image_label.after(0, self._update_image_label, img)
        if self.display_most_recent_image_flag == True:
            self.update_image(generated)
            self.display_most_recent_image_flag = False

    def _update_image_label(self, img):
        # Convert the PIL image to a Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(img)
        # Update the image label with the Tkinter PhotoImage
        self.image_label.config(image=tk_image)
        # Keep a reference to the image to prevent it from being garbage collected
        self.image_label.image = tk_image

    def display_most_recent_image(self):
        # Get a list of all image files in the output directory
        image_files = glob.glob("./_internal/output/*")

        # If there are no image files, return
        if not image_files:
            return

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Open the most recent image file
        img = Image.open(image_files[0])
        self.update_image(img)


    def on_resize(self, event):
        if hasattr(self, 'img'):
            self.update_image(self.img)
    
    def interrupt_generation(self):
        self.interrupt_flag = True

if __name__ == "__main__":
    app = App()
    app.mainloop()
