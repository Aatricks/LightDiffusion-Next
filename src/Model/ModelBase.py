import logging
import math
import torch

from src.Utilities import Latent
from src.Device import Device
from src.NeuralNetwork import unet
from src.cond import cast, cond
from src.sample import sampling


class BaseModel(torch.nn.Module):
    """#### Base class for models."""

    def __init__(
        self,
        model_config: object,
        model_type: sampling.ModelType = sampling.ModelType.EPS,
        device: torch.device = None,
        unet_model: object = unet.UNetModel1,
        flux: bool = False,
    ):
        """#### Initialize the BaseModel class.

        #### Args:
            - `model_config` (object): The model configuration.
            - `model_type` (sampling.ModelType, optional): The model type. Defaults to sampling.ModelType.EPS.
            - `device` (torch.device, optional): The device to use. Defaults to None.
            - `unet_model` (object, optional): The UNet model. Defaults to unet.UNetModel1.
        """
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        if flux:
            if not unet_config.get("disable_unet_model_creation", False):
                operations = model_config.custom_operations
                self.diffusion_model = unet_model(
                    **unet_config, device=device, operations=operations
                )
                logging.info(
                    "model weight dtype {}, manual cast: {}".format(
                        self.get_dtype(), self.manual_cast_dtype
                    )
                )
        else:
            if not unet_config.get("disable_unet_model_creation", False):
                if self.manual_cast_dtype is not None:
                    operations = cast.manual_cast
                else:
                    operations = cast.disable_weight_init
                self.diffusion_model = unet_model(
                    **unet_config, device=device, operations=operations
                )
        self.model_type = model_type
        self.model_sampling = sampling.model_sampling(
            model_config, model_type, flux=flux
        )

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor if flux else 2.0

    def apply_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_concat: torch.Tensor = None,
        c_crossattn: torch.Tensor = None,
        control: torch.Tensor = None,
        transformer_options: dict = {},
        **kwargs,
    ) -> torch.Tensor:
        """#### Apply the model to the input tensor.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The timestep tensor.
            - `c_concat` (torch.Tensor, optional): The concatenated condition tensor. Defaults to None.
            - `c_crossattn` (torch.Tensor, optional): The cross-attention condition tensor. Defaults to None.
            - `control` (torch.Tensor, optional): The control tensor. Defaults to None.
            - `transformer_options` (dict, optional): The transformer options. Defaults to {}.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        # Optimize concatenation operation by avoiding unnecessary list creation
        if c_concat is not None:
            xc = torch.cat((xc, c_concat), dim=1)

        # Determine dtype once to avoid repeated calls to get_dtype()
        dtype = (
            self.manual_cast_dtype
            if self.manual_cast_dtype is not None
            else self.get_dtype()
        )

        # Batch operations to reduce overhead
        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        context = c_crossattn.to(dtype) if c_crossattn is not None else None

        # Process extra conditions more efficiently
        extra_conds = {}
        for name, value in kwargs.items():
            if hasattr(value, "dtype") and value.dtype not in (torch.int, torch.long):
                extra_conds[name] = value.to(dtype)
            else:
                extra_conds[name] = value

        # Run diffusion model and calculate denoised output
        model_output = self.diffusion_model(
            xc,
            t,
            context=context,
            control=control,
            transformer_options=transformer_options,
            **extra_conds,
        ).float()

        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def get_dtype(self) -> torch.dtype:
        """#### Get the data type of the model.

        #### Returns:
            - `torch.dtype`: The data type.
        """
        return self.diffusion_model.dtype

    def encode_adm(self, **kwargs) -> None:
        """#### Encode the ADM.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `None`: The encoded ADM.
        """
        return None

    def extra_conds(self, **kwargs) -> dict:
        """#### Get the extra conditions.

        #### Args:
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `dict`: The extra conditions.
        """
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

    def load_model_weights(self, sd: dict, unet_prefix: str = "") -> "BaseModel":
        """#### Load the model weights.

        #### Args:
            - `sd` (dict): The state dictionary.
            - `unet_prefix` (str, optional): The UNet prefix. Defaults to "".

        #### Returns:
            - `BaseModel`: The model with loaded weights.
        """
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

    def process_latent_in(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent input.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent output.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return self.latent_format.process_out(latent)

    def memory_required(self, input_shape: tuple) -> float:
        """#### Calculate the memory required for the model.

        #### Args:
            - `input_shape` (tuple): The input shape.

        #### Returns:
            - `float`: The memory required.
        """
        dtype = self.get_dtype()
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype
        # TODO: this needs to be tweaked
        area = input_shape[0] * math.prod(input_shape[2:])
        return (area * Device.dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (
            1024 * 1024
        )


class BASE:
    """#### Base class for model configurations."""

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
    def matches(cls, unet_config: dict, state_dict: dict = None) -> bool:
        """#### Check if the UNet configuration matches.

        #### Args:
            - `unet_config` (dict): The UNet configuration.
            - `state_dict` (dict, optional): The state dictionary. Defaults to None.

        #### Returns:
            - `bool`: Whether the configuration matches.
        """
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict: dict, prefix: str = "") -> sampling.ModelType:
        """#### Get the model type.

        #### Args:
            - `state_dict` (dict): The state dictionary.
            - `prefix` (str, optional): The prefix. Defaults to "".

        #### Returns:
            - `sampling.ModelType`: The model type.
        """
        return sampling.ModelType.EPS

    def inpaint_model(self) -> bool:
        """#### Check if the model is an inpaint model.

        #### Returns:
            - `bool`: Whether the model is an inpaint model.
        """
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config: dict):
        """#### Initialize the BASE class.

        #### Args:
            - `unet_config` (dict): The UNet configuration.
        """
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(
        self, state_dict: dict, prefix: str = "", device: torch.device = None
    ) -> BaseModel:
        """#### Get the model.

        #### Args:
            - `state_dict` (dict): The state dictionary.
            - `prefix` (str, optional): The prefix. Defaults to "".
            - `device` (torch.device, optional): The device to use. Defaults to None.

        #### Returns:
            - `BaseModel`: The model.
        """
        out = BaseModel(
            self, model_type=self.model_type(state_dict, prefix), device=device
        )
        return out

    def process_unet_state_dict(self, state_dict: dict) -> dict:
        """#### Process the UNet state dictionary.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        return state_dict

    def process_vae_state_dict(self, state_dict: dict) -> dict:
        """#### Process the VAE state dictionary.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        return state_dict

    def set_inference_dtype(
        self, dtype: torch.dtype, manual_cast_dtype: torch.dtype
    ) -> None:
        """#### Set the inference data type.

        #### Args:
            - `dtype` (torch.dtype): The data type.
            - `manual_cast_dtype` (torch.dtype): The manual cast data type.
        """
        self.unet_config["dtype"] = dtype
        self.manual_cast_dtype = manual_cast_dtype


class Timestep(torch.nn.Module):
    """Timestep embedding layer for SDXL models."""

    def __init__(self, dim: int):
        """Initialize Timestep embedding.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass - convert timestep to embedding.

        Args:
            t: Timestep tensor

        Returns:
            Timestep embedding
        """
        return self._timestep_embedding(t, self.dim)

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps: 1-D tensor of N indices, one per batch element
            dim: Dimension of the output
            max_period: Controls the minimum frequency of embeddings

        Returns:
            An [N x dim] Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class CLIPEmbeddingNoiseAugmentation(torch.nn.Module):
    """CLIP embedding noise augmentation for SDXL."""

    def __init__(self, timestep_dim: int = 1280, max_noise_level: int = 1000):
        """Initialize noise augmentation.

        Args:
            timestep_dim: Dimension for timestep embedding
            max_noise_level: Maximum noise level
        """
        super().__init__()
        self.max_noise_level = max_noise_level
        self.time_embed = Timestep(timestep_dim)
        # Initialize with standard normal (mean=0, std=1)
        self.register_buffer("data_mean", torch.zeros(1, timestep_dim), persistent=False)
        self.register_buffer("data_std", torch.ones(1, timestep_dim), persistent=False)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input to centered mean and unit variance."""
        return (x - self.data_mean.to(x.device)) / self.data_std.to(x.device)

    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        """Unscale to original data stats."""
        return (x * self.data_std.to(x.device)) + self.data_mean.to(x.device)

    def q_sample(self, x: torch.Tensor, noise_level: torch.Tensor, seed: int = None) -> torch.Tensor:
        """Add noise to input based on noise level."""
        # Simple noise addition - can be made more sophisticated
        if seed is not None:
            generator = torch.Generator(device=x.device).manual_seed(seed)
        else:
            generator = None
        noise = torch.randn_like(x, generator=generator)
        # Scale noise by noise level (normalized to 0-1)
        noise_scale = noise_level.float() / self.max_noise_level
        return x + noise * noise_scale[:, None]

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor = None, seed: int = None) -> tuple:
        """Apply noise augmentation.

        Args:
            x: Input tensor
            noise_level: Noise level tensor
            seed: Random seed

        Returns:
            Tuple of (augmented tensor, noise level embedding)
        """
        if noise_level is None:
            noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        x = self.scale(x)
        z = self.q_sample(x, noise_level, seed=seed)
        z = self.unscale(z)
        noise_level_emb = self.time_embed(noise_level)
        return z, noise_level_emb


def sdxl_pooled(args: dict, noise_augmentor: CLIPEmbeddingNoiseAugmentation) -> torch.Tensor:
    """Extract pooled output for SDXL conditioning.

    Args:
        args: Arguments dict with pooled_output or unclip_conditioning
        noise_augmentor: Noise augmentation module

    Returns:
        Pooled CLIP embedding
    """
    if "unclip_conditioning" in args:
        # Apply noise augmentation for unclip
        unclip_cond = args.get("unclip_conditioning", None)
        device = args["device"]
        seed = args.get("seed", 0) - 10
        augmented, _ = noise_augmentor(unclip_cond.to(device), seed=seed)
        return augmented[:, :1280]
    else:
        return args["pooled_output"]


class SDXLRefiner(BaseModel):
    """SDXL Refiner model with aesthetic score conditioning."""

    def __init__(self, model_config: object, model_type: sampling.ModelType = sampling.ModelType.EPS, device: torch.device = None):
        """Initialize SDXL Refiner model.

        Args:
            model_config: Model configuration
            model_type: Type of model (EPS, V_PREDICTION, etc.)
            device: Device to load model on
        """
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(timestep_dim=1280)

    def encode_adm(self, **kwargs) -> torch.Tensor:
        """Encode ADM conditioning for SDXL Refiner.

        Args:
            **kwargs: Conditioning arguments (width, height, crop_w, crop_h, aesthetic_score, pooled_output, etc.)

        Returns:
            Encoded ADM tensor
        """
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        # Use different aesthetic scores for negative vs positive prompts
        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SDXL(BaseModel):
    """SDXL model with size and crop conditioning."""

    def __init__(self, model_config: object, model_type: sampling.ModelType = sampling.ModelType.EPS, device: torch.device = None):
        """Initialize SDXL model.

        Args:
            model_config: Model configuration
            model_type: Type of model (EPS, V_PREDICTION, etc.)
            device: Device to load model on
        """
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(timestep_dim=1280)

    def encode_adm(self, **kwargs) -> torch.Tensor:
        """Encode ADM conditioning for SDXL.

        Args:
            **kwargs: Conditioning arguments (width, height, crop_w, crop_h, target_width, target_height, pooled_output, etc.)

        Returns:
            Encoded ADM tensor
        """
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)
