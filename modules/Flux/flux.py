from __future__ import annotations
from abc import abstractmethod
import collections
import copy
from dataclasses import dataclass
import logging
import math
import os
from einops import rearrange, repeat
import gguf
import torch
from enum import Enum

import packaging.version
import torch.nn as nn


from transformers import T5TokenizerFast
from typing import Callable, List, Optional, Protocol, TypedDict

from modules.NeuralNetwork import unet
from modules.Utilities import Latent, util
from modules.Device import Device
from modules.SD15 import SDClip, SDToken
from modules.cond import cast, cond, cond_util
from modules.sample import CFG, ksampler_util, samplers, sampling, sampling_util
from modules.Model import ModelPatcher as Modelpatcher


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

folder_names_and_paths["ERSGAN"] = (
    [os.path.join(models_dir, "ERSGAN")],
    supported_pt_extensions,
)

output_directory = "./_internal/output"

filename_list_cache = {}

class Empty:
    pass


PROGRESS_BAR_ENABLED = True
PROGRESS_BAR_HOOK = None

logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)


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
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}, pipeline=False):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options
        self.pipeline = pipeline

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
        pipeline=False,
    ):
        extra_args["denoise_mask"] = denoise_mask
        model_k = sampling.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get(
            "random", False
        ):  # TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = (
                torch.randn(noise.shape, generator=generator, device="cpu")
                .to(noise.dtype)
                .to(noise.device)
            )
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas)
        )

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            def k_callback(x):
                return callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(
            model_k,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            pipeline=pipeline,
            **self.extra_options,
        )
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(
            sigmas[-1], samples
        )
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}, pipeline=False):
    if sampler_name == "euler":

        def euler_function(model, noise, sigmas, extra_args, callback, disable, pipeline=False):
            return samplers.sample_euler(
                model,
                noise,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                pipeline=pipeline,
                **extra_options,
            )

        sampler_function = euler_function

    return KSAMPLER(sampler_function, extra_options, inpaint_options, pipeline=pipeline)


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
            self.original_conds[k] = cond.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return CFG.sampling_function(
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
        pipeline=False,
    ):
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = cond.process_conds(
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
            pipeline=pipeline,
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
        pipeline=False,
    ):
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = cond_util.prepare_sampling(
            self.model_patcher, noise.shape, self.conds, flux_enabled=True
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
            pipeline=pipeline,
        )

        cond_util.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.conds
        del self.loaded_models
        return output


def sample(
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
    pipeline=False,
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
        pipeline=pipeline,
    )
    samples = samples.to(Device.intermediate_device())
    return samples


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


def sampler_object(name, pipeline=False):
    sampler = ksampler(name, pipeline=pipeline)
    return sampler


def sample1(
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
    pipeline=False,
):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(
        noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, pipeline=pipeline
    )

ops = cast.disable_weight_init

if Device.xformers_enabled_vae():
    import xformers
    import xformers.ops


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


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

        if Device.xformers_enabled_vae():
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


if Device.xformers_enabled():
    import xformers
    import xformers.ops

ops = cast.disable_weight_init

_ATTN_PRECISION = "fp32"


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


BROKEN_XFORMERS = False
try:
    x_vers = xformers.__version__
    # XFormers bug confirmed on all versions from 0.0.21 to 0.0.26 (q with bs bigger than 65535 gives CUDA error)
    BROKEN_XFORMERS = x_vers.startswith("0.0.2") and not x_vers.startswith("0.0.20")
except:
    pass


def attention_xformers(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads

    disabled_xformers = False

    if BROKEN_XFORMERS:
        if b * heads > 65535:
            disabled_xformers = True

    if not disabled_xformers:
        if torch.jit.is_tracing() or torch.jit.is_scripting():
            disabled_xformers = True

    if disabled_xformers:
        return attention_pytorch(q, k, v, heads, mask, skip_reshape=skip_reshape)

    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.reshape(b, -1, heads, dim_head),
            (q, k, v),
        )

    if mask is not None:
        pad = 8 - q.shape[1] % 8
        mask_out = torch.empty(
            [q.shape[0], q.shape[1], q.shape[1] + pad], dtype=q.dtype, device=q.device
        )
        mask_out[:, :, : mask.shape[-1]] = mask
        mask = mask_out[:, :, : mask.shape[-1]]

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)

    if skip_reshape:
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
    else:
        out = out.reshape(b, -1, heads * dim_head)

    return out


def attention_pytorch(
    q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False
):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
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


if Device.xformers_enabled():
    logging.info("Using xformers cross attention")
    optimized_attention = attention_xformers
else:
    logging.info("Using pytorch cross attention")
    optimized_attention = attention_pytorch

optimized_attention_masked = optimized_attention


def optimized_attention_for_device(device, mask=False, small_input=False):
    return attention_pytorch


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

    
ops = cast.manual_cast


def model_sampling(model_config, model_type):
    c = CONST
    s = ModelSamplingFlux

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_config, model_type=sampling.ModelType.EPS, device=None, unet_model=unet.UNetModel1
    ):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device

        if not unet_config.get("disable_unet_model_creation", False):
            if model_config.custom_operations is None:
                operations = pick_operations(
                    unet_config.get("dtype", None), self.manual_cast_dtype
                )
            else:
                operations = model_config.custom_operations
            self.diffusion_model = unet_model(
                **unet_config, device=device, operations=operations
            )
            logging.info(
                "model weight dtype {}, manual cast: {}".format(
                    self.get_dtype(), self.manual_cast_dtype
                )
            )
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor

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
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
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

    def extra_conds(self, **kwargs):
        out = {}
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out["y"] = cond.CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = cond.CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out["crossattn_controlnet"] = cond.CONDCrossAttn(cross_attn_cnet)

        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def memory_required(self, input_shape):
        if Device.xformers_enabled() or Device.pytorch_attention_flash_attention():
            dtype = self.get_dtype()
            if self.manual_cast_dtype is not None:
                dtype = self.manual_cast_dtype
            # TODO: this needs to be tweaked
            area = input_shape[0] * math.prod(input_shape[2:])
            return (area * Device.dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (
                1024 * 1024
            )
        else:
            # TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
            area = input_shape[0] * math.prod(input_shape[2:])
            return (area * 0.15 * self.memory_usage_factor) * (1024 * 1024)


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
    latent_format = Latent.LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    custom_operations = None

    @classmethod
    def matches(s, unet_config, state_dict=None):
        for k in s.unet_config:
            if k not in unet_config or s.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in s.required_keys:
                if k not in state_dict:
                    return False
        return True

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype

    def model_type(self, state_dict, prefix=""):
        return sampling.ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4


class CLIP:
    def __init__(
        self,
        target=None,
        embedding_directory=None,
        no_init=False,
        tokenizer_data={},
        parameters=0,
        model_options={},
    ):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", Device.text_encoder_device())
        offload_device = model_options.get(
            "offload_device", Device.text_encoder_offload_device()
        )
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = Device.text_encoder_dtype(load_device)

        params["dtype"] = dtype
        params["device"] = model_options.get(
            "initial_device",
            Device.text_encoder_initial_device(
                load_device, offload_device, parameters * Device.dtype_size(dtype)
            ),
        )
        params["model_options"] = model_options

        self.cond_stage_model = clip(**(params))

        for dt in self.cond_stage_model.dtypes:
            if not Device.supports_cast(load_device, dt):
                load_device = offload_device
                if params["device"] != offload_device:
                    self.cond_stage_model.to(offload_device)
                    logging.warning("Had to shift TE back.")

        self.tokenizer = tokenizer(
            embedding_directory=embedding_directory, tokenizer_data=tokenizer_data
        )
        self.patcher = Modelpatcher.ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        if params["device"] == load_device:
            Device.load_models_gpu([self.patcher], force_full_load=True, flux_enabled=True)
        self.layer_idx = None
        logging.debug(
            "CLIP model load device: {}, offload device: {}, current: {}".format(
                load_device, offload_device, params["device"]
            )
        )

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False, flux_enabled=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model(flux_enabled=flux_enabled)
        o = self.cond_stage_model.encode_token_weights(tokens)
        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            return out

        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self, flux_enabled=False):
        Device.load_model_gpu(self.patcher, flux_enabled=flux_enabled)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6


MAX_RESOLUTION = 16384


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
    pipeline=False,
):
    latent_image = latent["samples"]
    latent_image = Latent.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = ksampler_util.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]
    samples = sample(
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
        seed=seed,
        pipeline=pipeline,
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

logger = logging.getLogger()


class UnetApplyFunction(Protocol):
    """Function signature protocol on comfy.model_base.BaseModel.apply_model"""

    def __call__(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class UnetApplyConds(TypedDict):
    """Optional conditions for unet apply function."""

    c_concat: Optional[torch.Tensor]
    c_crossattn: Optional[torch.Tensor]
    control: Optional[torch.Tensor]
    transformer_options: Optional[dict]


class UnetParams(TypedDict):
    # Tensor of shape [B, C, H, W]
    input: torch.Tensor
    # Tensor of shape [B]
    timestep: torch.Tensor
    c: UnetApplyConds
    # List of [0, 1], [0], [1], ...
    # 0 means conditional, 1 means conditional unconditional
    cond_or_uncond: List[int]


UnetWrapperFunction = Callable[[UnetApplyFunction, UnetParams], torch.Tensor]


def unet_prefix_from_state_dict(state_dict):
    candidates = [
        "model.diffusion_model.",  # ldm/sgm models
        "model.model.",  # audio models
    ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model."  # aura flow and others


class SD3(Latent.LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645, 0.0177, 0.1052],
            [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360],
            [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364],
            [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283],
            [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700],
            [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220],
            [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259],
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Flux1(SD3):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609],
            [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503],
            [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530],
            [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001],
            [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013],
            [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680],
        ]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class CONST:
    def calculate_input(self, sigma, noise):
        return noise

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma, latent):
        return latent / (1.0 - sigma)


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.15))

    def set_parameters(self, shift=1.15, timesteps=10000):
        self.shift = shift
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return flux_time_shift(self.shift, 1.0, timestep)

ops = cast.disable_weight_init


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


# Original code can be found on: https://github.com/black-forest-labs/flux


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    if Device.is_device_mps(pos.device) or Device.is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device
    )
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega
    )
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.in_layer = operations.Linear(
            in_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )
        self.silu = nn.SiLU()
        self.out_layer = operations.Linear(
            hidden_dim, hidden_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((dim), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.scale, 1e-6)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.query_norm = RMSNorm(
            dim, dtype=dtype, device=device, operations=operations
        )
        self.key_norm = RMSNorm(dim, dtype=dtype, device=device, operations=operations)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = operations.Linear(
            dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device
        )
        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(
        self, dim: int, double: bool, dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = operations.Linear(
            dim, self.multiplier * dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, vec: torch.Tensor) -> tuple:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(
            hidden_size, double=True, dtype=dtype, device=device, operations=operations
        )
        self.img_norm1 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.img_norm2 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.img_mlp = nn.Sequential(
            operations.Linear(
                hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device
            ),
            nn.GELU(approximate="tanh"),
            operations.Linear(
                mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

        self.txt_mod = Modulation(
            hidden_size, double=True, dtype=dtype, device=device, operations=operations
        )
        self.txt_norm1 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.txt_norm2 = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.txt_mlp = nn.Sequential(
            operations.Linear(
                hidden_size, mlp_hidden_dim, bias=True, dtype=dtype, device=device
            ),
            nn.GELU(approximate="tanh"),
            operations.Linear(
                mlp_hidden_dim, hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(
            img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(
            txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = operations.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim,
            dtype=dtype,
            device=device,
        )
        # proj and mlp_out
        self.linear2 = operations.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size, dtype=dtype, device=device
        )

        self.norm = QKNorm(head_dim, dtype=dtype, device=device, operations=operations)

        self.hidden_size = hidden_size
        self.pre_norm = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(
            hidden_size, double=False, dtype=dtype, device=device, operations=operations
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.norm_final = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device
        )
        self.linear = operations.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if (
        padding_mode == "circular"
        and torch.jit.is_tracing()
        or torch.jit.is_scripting()
    ):
        padding_mode = "reflect"
    pad_h = (patch_size[0] - img.shape[-2] % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - img.shape[-1] % patch_size[1]) % patch_size[1]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)


try:
    rms_norm_torch = torch.nn.functional.rms_norm
except:
    rms_norm_torch = None


def rms_norm(x, weight, eps=1e-6):
    if rms_norm_torch is not None and not (
        torch.jit.is_tracing() or torch.jit.is_scripting()
    ):
        return rms_norm_torch(
            x,
            weight.shape,
            weight=cast.cast_to(weight, dtype=x.dtype, device=x.device),
            eps=eps,
        )
    else:
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        return (x * rrms) * cast.cast_to(weight, dtype=x.dtype, device=x.device)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux3(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        image_model=None,
        final_layer=True,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = operations.Linear(
            self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.vector_in = MLPEmbedder(
            params.vec_in_dim,
            self.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = operations.Linear(
            params.context_in_dim, self.hidden_size, dtype=dtype, device=device
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        if final_layer:
            self.final_layer = LastLayer(
                self.hidden_size,
                1,
                self.out_channels,
                dtype=dtype,
                device=device,
                operations=operations,
            )

    def forward_orig(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
    ) -> torch.Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(sampling_util.timestep_embedding_flux(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                sampling_util.timestep_embedding_flux(guidance, 256).to(img.dtype)
            )

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(self, x, timestep, context, y, guidance, control=None, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(
            x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
        )

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[
                :, None
            ]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[
                None, :
            ]
        )
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(
            img, img_ids, context, txt_ids, timestep, y, guidance, control
        )
        return rearrange(
            out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2
        )[:, :, :h, :w]


class Flux2(BaseModel):
    def __init__(self, model_config, model_type=sampling.ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=Flux3)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = cond.CONDRegular(cross_attn)
        out["guidance"] = cond.CONDRegular(torch.FloatTensor([kwargs.get("guidance", 3.5)]))
        return out


class Flux(BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    sampling_settings = {}

    unet_extra_config = {}
    latent_format = Flux1

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    def get_model(self, state_dict, prefix="", device=None):
        out = Flux2(self, device=device)
        return out


models = [Flux]


def load_diffusion_model_state_dict(
    sd, model_options={}
):  # load unet in diffusers or regular format
    dtype = model_options.get("dtype", None)

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = unet_prefix_from_state_dict(sd)
    temp_sd = util.state_dict_prefix_replace(
        sd, {diffusion_model_prefix: ""}, filter_keys=True
    )
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = util.calculate_parameters(sd)
    load_device = Device.get_torch_device()
    model_config = unet.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd

    offload_device = Device.unet_offload_device()
    if dtype is None:
        unet_dtype2 = Device.unet_dtype(
            model_params=parameters,
            supported_dtypes=model_config.supported_inference_dtypes,
        )
    else:
        unet_dtype2 = dtype

    manual_cast_dtype = Device.unet_manual_cast(
        unet_dtype2, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype2, manual_cast_dtype)
    model_config.custom_operations = model_options.get(
        "custom_operations", model_config.custom_operations
    )
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in unet: {}".format(left_over))
    return Modelpatcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3 - i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i + 1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
    ("proj_out.", "proj_attn."),
]


class T5XXLModel(SDClip.SDClipModel):
    def __init__(
        self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}
    ):
        textmodel_json_config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "./clip/t5_config_xxl.json",
        )
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"end": 1, "pad": 0},
            model_class=T5,
            model_options=model_options,
        )


class T5XXLTokenizer(SDToken.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "./clip/t5_tokenizer"
        )
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_size=4096,
            embedding_key="t5xxl",
            tokenizer_class=T5TokenizerFast,
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=256,
        )


class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        clip_l_tokenizer_class = tokenizer_data.get(
            "clip_l_tokenizer_class", SDToken.SDTokenizer
        )
        self.clip_l = clip_l_tokenizer_class(embedding_directory=embedding_directory)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_t5 = Device.pick_weight_dtype(dtype_t5, dtype, device)
        clip_l_class = model_options.get("clip_l_class", SDClip.SDClipModel)
        self.clip_l = clip_l_class(
            device=device,
            dtype=dtype,
            return_projected_pooled=False,
            model_options=model_options,
        )
        self.t5xxl = T5XXLModel(
            device=device, dtype=dtype_t5, model_options=model_options
        )
        self.dtypes = set([dtype, dtype_t5])

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)


def flux_clip(dtype_t5=None):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            super().__init__(
                dtype_t5=dtype_t5,
                device=device,
                dtype=dtype,
                model_options=model_options,
            )

    return FluxClipModel_


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return cast.cast_to_input(self.weight, x) * x


activations = {
    "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim, ff_dim, ff_activation, dtype, device, operations):
        super().__init__()
        self.wi_0 = operations.Linear(
            model_dim, ff_dim, bias=False, dtype=dtype, device=device
        )
        self.wi_1 = operations.Linear(
            model_dim, ff_dim, bias=False, dtype=dtype, device=device
        )
        self.wo = operations.Linear(
            ff_dim, model_dim, bias=False, dtype=dtype, device=device
        )
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.act = activations[ff_activation]

    def forward(self, x):
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        # x = self.dropout(x)
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    def __init__(
        self, model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations
    ):
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                model_dim, ff_dim, ff_activation, dtype, device, operations
            )

        self.layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # x = x + self.dropout(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.k = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.v = operations.Linear(
            model_dim, inner_dim, bias=False, dtype=dtype, device=device
        )
        self.o = operations.Linear(
            inner_dim, model_dim, bias=False, dtype=dtype, device=device
        )
        self.num_heads = num_heads

        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(
                self.relative_attention_num_buckets,
                self.num_heads,
                device=device,
                dtype=dtype,
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket, out_dtype=dtype
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)

        if past_bias is not None:
            if mask is not None:
                mask = mask + past_bias
            else:
                mask = past_bias

        out = optimized_attention(
            q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask
        )
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            model_dim,
            inner_dim,
            num_heads,
            relative_attention_bias,
            dtype,
            device,
            operations,
        )
        self.layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        self.layer_norm(x)
        output, past_bias = self.SelfAttention(
            self.layer_norm(x),
            mask=mask,
            past_bias=past_bias,
            optimized_attention=optimized_attention,
        )
        # x = x + self.dropout(attention_output)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):
    def __init__(
        self,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention_bias,
        dtype,
        device,
        operations,
    ):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                model_dim,
                inner_dim,
                ff_dim,
                num_heads,
                relative_attention_bias,
                dtype,
                device,
                operations,
            )
        )
        self.layer.append(
            T5LayerFF(
                model_dim, ff_dim, ff_activation, gated_act, dtype, device, operations
            )
        )

    def forward(self, x, mask=None, past_bias=None, optimized_attention=None):
        x, past_bias = self.layer[0](x, mask, past_bias, optimized_attention)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        model_dim,
        inner_dim,
        ff_dim,
        ff_activation,
        gated_act,
        num_heads,
        relative_attention,
        dtype,
        device,
        operations,
    ):
        super().__init__()

        self.block = torch.nn.ModuleList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    ff_activation,
                    gated_act,
                    num_heads,
                    relative_attention_bias=((not relative_attention) or (i == 0)),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
    ):
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
            ).expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        intermediate = None
        optimized_attention = optimized_attention_for_device(
            x.device, mask=attention_mask is not None, small_input=True
        )
        past_bias = None
        for i, l in enumerate(self.block):
            x, past_bias = l(x, mask, past_bias, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_layers"]
        model_dim = config_dict["d_model"]

        self.encoder = T5Stack(
            self.num_layers,
            model_dim,
            model_dim,
            config_dict["d_ff"],
            config_dict["dense_act_fn"],
            config_dict["is_gated_act"],
            config_dict["num_heads"],
            config_dict["model_type"] != "umt5",
            dtype,
            device,
            operations,
        )
        self.dtype = dtype
        self.shared = operations.Embedding(
            config_dict["vocab_size"], model_dim, device=device, dtype=dtype
        )

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings):
        self.shared = embeddings

    def forward(self, input_ids, *args, **kwargs):
        x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x)  # Fix for fp8 T5 base
        return self.encoder(x, *args, **kwargs)


def load_text_encoder_state_dicts(
    state_dicts=[],
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "text_projection" in clip_data[i]:
            clip_data[i]["text_projection.weight"] = clip_data[i][
                "text_projection"
            ].transpose(
                0, 1
            )  # old models saved with the CLIPSave node

    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 2:
        if clip_type == CLIPType.FLUX:
            weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
            weight = clip_data[0].get(weight_name, clip_data[1].get(weight_name, None))
            dtype_t5 = None
            if weight is not None:
                dtype_t5 = weight.dtype

            clip_target.clip = flux_clip(dtype_t5=dtype_t5)
            clip_target.tokenizer = FluxTokenizer

    parameters = 0
    tokenizer_data = {}
    for c in clip_data:
        parameters += util.calculate_parameters(c)
        tokenizer_data, model_options = SDToken.model_options_long_clip(
            c, tokenizer_data, model_options
        )

    clip = CLIP(
        clip_target,
        embedding_directory=embedding_directory,
        parameters=parameters,
        tokenizer_data=tokenizer_data,
        model_options=model_options,
    )
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip


TORCH_COMPATIBLE_QTYPES = {
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


def is_torch_compatible(tensor):
    return (
        tensor is None
        or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES
    )


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


dequantize_functions = {
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
}


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """

    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    torch_compatible_tensor_types = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(
            f"{prefix}bias"
        )
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias
        return

        # This would return the actual state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # apply patches
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = (
                    dtype if self.patch_dtype == "target" else self.patch_dtype
                )
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = Device.device_supports_non_blocking(device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = cast.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight = s.get_weight(s.weight.to(device), dtype)
        weight = cast.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.forward_ggml_cast_weights(input, *args, **kwargs)
        return super().forward_comfy_cast_weights(input, *args, **kwargs)


class GGMLOps(cast.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """

    class Linear(GGMLLayer, cast.manual_cast.Linear):
        def __init__(
            self, in_features, out_features, bias=True, device=None, dtype=None
        ):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Embedding(GGMLLayer, cast.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(
                self, device=input.device, dtype=out_dtype
            )
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)


MODEL_DETECTION = (
    (
        "flux",
        (
            ("transformer_blocks.0.attn.norm_added_k.weight",),
            ("double_blocks.0.img_attn.proj.weight",),
        ),
    ),
    ("sd3", (("transformer_blocks.0.attn.add_q_proj.weight",),)),
    (
        "sdxl",
        (
            (
                "down_blocks.0.downsamplers.0.conv.weight",
                "add_embedding.linear_1.weight",
            ),
            (
                "input_blocks.3.0.op.weight",
                "input_blocks.6.0.op.weight",
                "output_blocks.2.2.conv.weight",
                "output_blocks.5.2.conv.weight",
            ),  # Non-diffusers
            ("label_emb.0.0.weight",),
        ),
    ),
    (
        "sd1",
        (
            ("down_blocks.0.downsamplers.0.conv.weight",),
            (
                "input_blocks.3.0.op.weight",
                "input_blocks.6.0.op.weight",
                "input_blocks.9.0.op.weight",
                "output_blocks.2.1.conv.weight",
                "output_blocks.5.2.conv.weight",
                "output_blocks.8.2.conv.weight",
            ),  # Non-diffusers
        ),
    ),
)


def gguf_sd_loader_get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}"
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, *args, **kwargs):
        # Intel Arc fix, ref#50
        new = super().__deepcopy__(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


def gguf_sd_loader(path, handle_prefix="model.diffusion_model."):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = None
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        if (
            len(arch_field.types) != 1
            or arch_field.types[0] != gguf.GGUFValueType.STRING
        ):
            raise TypeError(
                f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}"
            )
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
        if arch_str not in {"flux", "sd1", "sdxl", "t5", "t5encoder"}:
            raise ValueError(
                f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}"
            )

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
        torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = gguf_sd_loader_get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any(
                    [
                        tensor_name.endswith(x)
                        for x in (".proj_in.weight", ".proj_out.weight")
                    ]
                ):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape
        )
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k, v in qtype_dict.items():
        print(f" {k:30}{v:3}")

    return state_dict


class GGUFModelPatcher(Modelpatcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = util.get_attr(self.model, key)

        calculate_weight = self.calculate_weight

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(
                patches,
                self.load_device if self.patch_on_device else self.offload_device,
            )
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple(
                    "Dimension", ["weight", "inplace_update"]
                )(
                    weight.to(device=self.offload_device, copy=inplace_update),
                    inplace_update,
                )

            if device_to is not None:
                temp_weight = Device.cast_to_device(
                    weight, device_to, torch.float32, copy=True
                )
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            util.copy_to_param(self.model, key, out_weight)
        else:
            util.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(
            device_to=device_to, unpatch_weights=unpatch_weights
        )

    mmap_released = False

    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        n = GGUFModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n


class UnetLoaderGGUF:

    def load_unet(
        self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None
    ):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = "./_internal/unet/" + unet_name
        sd = gguf_sd_loader(unet_path)
        model = load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path)
            )
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

clip_sd_map = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

clip_name_dict = {
    "stable_diffusion": CLIPType.STABLE_DIFFUSION,
    "stable_cascade": CLIPType.STABLE_CASCADE,
    "stable_audio": CLIPType.STABLE_AUDIO,
    "sdxl": CLIPType.STABLE_DIFFUSION,
    "sd3": CLIPType.SD3,
    "flux": CLIPType.FLUX,
}


def gguf_clip_loader(path):
    raw_sd = gguf_sd_loader(path)
    assert "enc.blk.23.ffn_up.weight" in raw_sd, "Invalid Text Encoder!"
    sd = {}
    for k, v in raw_sd.items():
        for s, d in clip_sd_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


class CLIPLoaderGGUF:
    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                clip_data.append(gguf_clip_loader(p))
            else:
                sd = util.load_torch_file(p, safe_load=True)
                clip_data.append(
                    {
                        k: GGMLTensor(
                            v,
                            tensor_type=gguf.GGMLQuantizationType.F16,
                            tensor_shape=v.shape,
                        )
                        for k, v in sd.items()
                    }
                )
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = load_text_encoder_state_dicts(
            clip_type=clip_type,
            state_dicts=clip_data,
            model_options={
                "custom_operations": GGMLOps,
                "initial_device": Device.text_encoder_offload_device(),
            },
            embedding_directory="models/embeddings",
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        # for some reason this is just missing in some SAI checkpoints
        if getattr(clip.cond_stage_model, "clip_l", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_l.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_l.transformer.text_projection = (
                    cast.manual_cast.Linear(768, 768)
                )
        if getattr(clip.cond_stage_model, "clip_g", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_g.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_g.transformer.text_projection = (
                    cast.manual_cast.Linear(1280, 1280)
                )

        return clip


class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = "./_internal/clip/" + clip_name1
        clip_path2 = "./_internal/clip/" + clip_name2
        clip_paths = (clip_path1, clip_path2)
        clip_type = clip_name_dict.get(type, CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class CLIPTextEncodeFlux:
    def encode(self, clip, clip_l, t5xxl, guidance, flux_enabled:bool = False):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True, flux_enabled=flux_enabled)
        cond = output.pop("cond")
        output["guidance"] = guidance
        return ([[cond, output]],)


class ConditioningZeroOut:

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c,)


KSAMPLER_NAMES = [
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
]

SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]


SCHEDULER_NAMES = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

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
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = ksampler_util.calculate_sigmas(
            self.model.get_model_object("model_sampling"), self.scheduler, steps
        )

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
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
        pipeline=False,
    ):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[: last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler, pipeline=pipeline)

        return sample1(
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
            pipeline=pipeline,
        )


class KSampler:
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
        pipeline=False,
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
            pipeline=pipeline,
        )