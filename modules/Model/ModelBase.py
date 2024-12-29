import logging
import torch

from modules.Utilities import Latent
from modules.Device import Device
from modules.NeuralNetwork import unet
from modules.cond import cast, cond
from modules.sample import sampling


class BaseModel(torch.nn.Module):
    """#### Base class for models."""

    def __init__(
        self,
        model_config: object,
        model_type: sampling.ModelType = sampling.ModelType.EPS,
        device: torch.device = None,
        unet_model: object = unet.UNetModel1,
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

        if not unet_config.get("disable_unet_model_creation", False):
            if self.manual_cast_dtype is not None:
                operations = cast.manual_cast
            else:
                operations = cast.disable_weight_init
            self.diffusion_model = unet_model(
                **unet_config, device=device, operations=operations
            )
        self.model_type = model_type
        self.model_sampling = sampling.model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))

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
        cross_attn = kwargs.get("cross_attn", None)
        out["c_crossattn"] = cond.CONDCrossAttn(cross_attn)
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
        area = input_shape[0] * input_shape[2] * input_shape[3]
        return (area * Device.dtype_size(dtype) / 50) * (1024 * 1024)


class BASE:
    """#### Base class for model configurations."""

    unet_config: dict = {}
    unet_extra_config: dict = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys: dict = {}

    clip_prefix: list = []
    clip_vision_prefix: str = None
    noise_aug_config: dict = None
    sampling_settings: dict = {}
    latent_format: object = Latent.LatentFormat
    vae_key_prefix: list = ["first_stage_model."]
    text_encoder_key_prefix: list = ["cond_stage_model."]
    supported_inference_dtypes: list = [torch.float16, torch.bfloat16, torch.float32]

    manual_cast_dtype: torch.dtype = None

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
