"""Base model classes for diffusion models."""
import logging
import math
import torch
from src.Utilities import Latent
from src.Device import Device
from src.NeuralNetwork import unet
from src.cond import cast, cond
from src.sample import sampling


class BaseModel(torch.nn.Module):
    """Base class for diffusion models."""
    def __init__(self, model_config, model_type=sampling.ModelType.EPS, device=None, 
                 unet_model=unet.UNetModel1, flux=False):
        super().__init__()
        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        
        if not unet_config.get("disable_unet_model_creation", False):
            operations = model_config.custom_operations if flux else (
                cast.manual_cast if self.manual_cast_dtype else cast.disable_weight_init)
            self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
            
        self.model_type = model_type
        self.model_sampling = sampling.model_sampling(model_config, model_type, flux=flux)
        self.adm_channels = unet_config.get("adm_in_channels", 0) or 0
        self.concat_keys = ()
        self.memory_usage_factor = model_config.memory_usage_factor if flux else 2.0
        logging.info(f"model_type {model_type.name}")

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        """Apply model to input tensor."""
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat((xc, c_concat), dim=1)
        
        dtype = self.manual_cast_dtype or self.get_dtype()
        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = c_crossattn.to(dtype) if c_crossattn is not None else None
        
        extra = {k: v.to(dtype) if hasattr(v, "dtype") and v.dtype not in (torch.int, torch.long) else v 
                 for k, v in kwargs.items()}
        
        output = self.diffusion_model(xc, t, context=context, control=control, 
                                       transformer_options=transformer_options, **extra).float()
        return self.model_sampling.calculate_denoised(sigma, output, x)

    def get_dtype(self):
        return self.diffusion_model.dtype

    def encode_adm(self, **kwargs):
        return None

    def extra_conds(self, **kwargs):
        out = {}
        if (adm := self.encode_adm(**kwargs)) is not None:
            out["y"] = cond.CONDRegular(adm)
        if (ca := kwargs.get("cross_attn")) is not None:
            out["c_crossattn"] = cond.CONDCrossAttn(ca)
        if (ca_cnet := kwargs.get("cross_attn_controlnet")) is not None:
            out["crossattn_controlnet"] = cond.CONDCrossAttn(ca_cnet)
        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {k[len(unet_prefix):]: sd.pop(k) for k in list(sd.keys()) if k.startswith(unet_prefix)}
        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if m: logging.warning(f"unet missing: {m}")
        if u: logging.warning(f"unet unexpected: {u}")
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def memory_required(self, input_shape):
        dtype = self.manual_cast_dtype or self.get_dtype()
        area = input_shape[0] * math.prod(input_shape[2:])
        return area * Device.dtype_size(dtype) * 0.01 * self.memory_usage_factor * 1024 * 1024


class BASE:
    """Base configuration class."""
    unet_config = {}
    unet_extra_config = {"num_heads": -1, "num_head_channels": 64}
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
    def matches(cls, unet_config, state_dict=None):
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        return state_dict is None or all(k in state_dict for k in cls.required_keys)

    def model_type(self, state_dict, prefix=""):
        return sampling.ModelType.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = {**unet_config, **self.unet_extra_config}
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()

    def get_model(self, state_dict, prefix="", device=None):
        return BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


class Timestep(torch.nn.Module):
    """Timestep embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if self.dim % 2 else emb


class CLIPEmbeddingNoiseAugmentation(torch.nn.Module):
    """CLIP embedding noise augmentation."""
    def __init__(self, timestep_dim=1280, max_noise_level=1000):
        super().__init__()
        self.max_noise_level = max_noise_level
        self.time_embed = Timestep(timestep_dim)
        self.register_buffer("data_mean", torch.zeros(1, timestep_dim), persistent=False)
        self.register_buffer("data_std", torch.ones(1, timestep_dim), persistent=False)

    def forward(self, x, noise_level=None, seed=None):
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        x_scaled = (x - self.data_mean.to(x.device)) / self.data_std.to(x.device)
        gen = torch.Generator(device=x.device).manual_seed(seed) if seed else None
        noise = torch.randn_like(x_scaled, generator=gen)
        z = x_scaled + noise * (noise_level.float() / self.max_noise_level)[:, None]
        z = z * self.data_std.to(x.device) + self.data_mean.to(x.device)
        return z, self.time_embed(noise_level)


def sdxl_pooled(args, noise_augmentor):
    """Extract pooled output for SDXL."""
    if "unclip_conditioning" in args:
        z, _ = noise_augmentor(args["unclip_conditioning"].to(args["device"]), seed=args.get("seed", 0) - 10)
        return z[:, :1280]
    
    pooled = args.get("pooled_output")
    # Robustly handle wrapped CONDRegular objects from cond.py
    if hasattr(pooled, "cond"):
        return pooled.cond
    return pooled


class SDXLBase(BaseModel):
    """SDXL base with size/crop conditioning."""
    def __init__(self, model_config, model_type=sampling.ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(timestep_dim=1280)

    def _embed_values(self, *values):
        return torch.cat([self.embedder(torch.Tensor([v])) for v in values])

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        w, h = kwargs.get("width", 768), kwargs.get("height", 768)
        cw, ch = kwargs.get("crop_w", 0), kwargs.get("crop_h", 0)
        flat = torch.flatten(self._embed_values(h, w, ch, cw, *self._extra_adm_values(kwargs)))
        return torch.cat((clip_pooled.to(flat.device), flat.unsqueeze(0).repeat(clip_pooled.shape[0], 1)), dim=1)

    def _extra_adm_values(self, kwargs):
        return [kwargs.get("target_height", kwargs.get("height", 768)), 
                kwargs.get("target_width", kwargs.get("width", 768))]


class SDXL(SDXLBase):
    """SDXL model."""
    pass


class SDXLRefiner(SDXLBase):
    """SDXL Refiner with aesthetic conditioning."""
    def _extra_adm_values(self, kwargs):
        aesthetic = 2.5 if kwargs.get("prompt_type", "") == "negative" else kwargs.get("aesthetic_score", 6)
        return [aesthetic]
