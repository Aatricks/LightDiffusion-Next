from enum import Enum
import logging
import torch

from src.Model import ModelPatcher
from src.Attention import Attention
from src.Device import Device
from src.SD15 import SDToken
from src.Utilities import util
from src.clip import FluxClip
from src.cond import cast


class CLIPAttention(torch.nn.Module):
    """#### The CLIPAttention module."""
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPAttention module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        optimized_attention: callable = None,
    ) -> torch.Tensor:
        """#### Forward pass for the CLIPAttention module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `optimized_attention` (callable, optional): The optimized attention function. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
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
    """#### The CLIPMLP module.
    (MLP stands for Multi-Layer Perceptron.)"""
    def __init__(
        self,
        embed_dim: int,
        intermediate_size: int,
        activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPMLP module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `intermediate_size` (int): The intermediate size.
            - `activation` (str): The activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.fc1 = operations.Linear(
            embed_dim, intermediate_size, bias=True, dtype=dtype, device=device
        )
        self.activation = ACTIVATIONS[activation]
        self.fc2 = operations.Linear(
            intermediate_size, embed_dim, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the CLIPMLP module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CLIPLayer(torch.nn.Module):
    """#### The CLIPLayer module."""
    def __init__(
        self,
        embed_dim: int,
        heads: int,
        intermediate_size: int,
        intermediate_activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPLayer module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `intermediate_size` (int): The intermediate size.
            - `intermediate_activation` (str): The intermediate activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        optimized_attention: callable = None,
    ) -> torch.Tensor:
        """#### Forward pass for the CLIPLayer module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `optimized_attention` (callable, optional): The optimized attention function. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        x += self.self_attn(self.layer_norm1(x), mask, optimized_attention)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):
    """#### The CLIPEncoder module."""
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        heads: int,
        intermediate_size: int,
        intermediate_activation: str,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
        """#### Initialize the CLIPEncoder module.

        #### Args:
            - `num_layers` (int): The number of layers.
            - `embed_dim` (int): The embedding dimension.
            - `heads` (int): The number of attention heads.
            - `intermediate_size` (int): The intermediate size.
            - `intermediate_activation` (str): The intermediate activation function.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        intermediate_output: int = None,
    ) -> tuple:
        """#### Forward pass for the CLIPEncoder module.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `intermediate_output` (int, optional): The intermediate output layer. Defaults to None.

        #### Returns:
            - `tuple`: The output tensor and the intermediate output tensor.
        """
        optimized_attention = Attention.optimized_attention_for_device()

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for i, length in enumerate(self.layers):
            x = length(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    """#### The CLIPEmbeddings module."""
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = 49408,
        num_positions: int = 77,
        dtype: torch.dtype = None,
        device: torch.device = None,
        operations: object = torch.nn,
    ):
        """#### Initialize the CLIPEmbeddings module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `vocab_size` (int, optional): The vocabulary size. Defaults to 49408.
            - `num_positions` (int, optional): The number of positions. Defaults to 77.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device to use. Defaults to None.
        """
        super().__init__()
        self.token_embedding = operations.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = operations.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """#### Forward pass for the CLIPEmbeddings module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.
            - `dtype` (torch.dtype, optional): The data type. Defaults to torch.float32.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return self.token_embedding(input_tokens, out_dtype=dtype) + cast.cast_to(
            self.position_embedding.weight, dtype=dtype, device=input_tokens.device
        )



class CLIP:
    """#### The CLIP class."""
    def __init__(
        self,
        target: object = None,
        embedding_directory: str = None,
        no_init: bool = False,
        tokenizer_data={},
        parameters=0,
        model_options={},
    ):
        """#### Initialize the CLIP class.

        #### Args:
            - `target` (object, optional): The target object. Defaults to None.
            - `embedding_directory` (str, optional): The embedding directory. Defaults to None.
            - `no_init` (bool, optional): Whether to skip initialization. Defaults to False.
        """
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

        # for dt in self.cond_stage_model.dtypes:
        #     if not Device.supports_cast(load_device, dt):
        #         load_device = offload_device
        #         if params["device"] != offload_device:
        #             self.cond_stage_model.to(offload_device)
        #             logging.warning("Had to shift TE back.")

        try:
            self.tokenizer = tokenizer(
                embedding_directory=embedding_directory, tokenizer_data=tokenizer_data
            )
        except TypeError:
            self.tokenizer = tokenizer(
                embedding_directory=embedding_directory
            )
        self.patcher = ModelPatcher.ModelPatcher(
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

    def clone(self) -> "CLIP":
        """#### Clone the CLIP object.

        #### Returns:
            - `CLIP`: The cloned CLIP object.
        """
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(
        self, patches: list, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> None:
        """#### Add patches to the model.

        #### Args:
            - `patches` (list): The patches to add.
            - `strength_patch` (float, optional): The strength of the patches. Defaults to 1.0.
            - `strength_model` (float, optional): The strength of the model. Defaults to 1.0.
        """
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx: int) -> None:
        """#### Set the clip layer.

        #### Args:
            - `layer_idx` (int): The layer index.
        """
        self.layer_idx = layer_idx

    def tokenize(self, text: str, return_word_ids: bool = False) -> list:
        """#### Tokenize the input text.

        #### Args:
            - `text` (str): The input text.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `list`: The tokenized text.
        """
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens: list, return_pooled: bool = False, return_dict: bool = False, flux_enabled:bool = False) -> tuple:
        """#### Encode the input tokens.

        #### Args:
            - `tokens` (list): The input tokens.
            - `return_pooled` (bool, optional): Whether to return the pooled output. Defaults to False.
            - `flux_enabled` (bool, optional): Whether to enable flux. Defaults to False.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
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

    def load_sd(self, sd: dict, full_model: bool = False) -> None:
        """#### Load the state dictionary.

        #### Args:
            - `sd` (dict): The state dictionary.
            - `full_model` (bool, optional): Whether to load the full model. Defaults to False.
        """
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def load_model(self, flux_enabled:bool = False) -> ModelPatcher:
        """#### Load the model.

        #### Returns:
            - `ModelPatcher`: The model patcher.
        """
        Device.load_model_gpu(self.patcher, flux_enabled=flux_enabled)
        return self.patcher
    
    def encode(self, text):
        """#### Encode the input text.
        
        #### Args:
            - `text` (str): The input text.
        
        #### Returns:
            - `torch.Tensor`: The encoded text.
        """
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def get_sd(self):
        """#### Get the state dictionary.
        
        #### Returns:
            - `dict`: The state dictionary.
        """
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def get_key_patches(self):
        """#### Get the key patches.
        
        #### Returns:
            - `list`: The key patches.
        """
        return self.patcher.get_key_patches()


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    SD3 = 3
    FLUX = 6

def load_text_encoder_state_dicts(
    state_dicts=[],
    embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION,
    model_options={},
):
    """#### Load the text encoder state dictionaries.
    
    #### Args:
        - `state_dicts` (list, optional): The state dictionaries. Defaults to [].
        - `embedding_directory` (str, optional): The embedding directory. Defaults to None.
        - `clip_type` (CLIPType, optional): The CLIP type. Defaults to CLIPType.STABLE_DIFFUSION.
        - `model_options` (dict, optional): The model options. Defaults to {}.
    
    #### Returns:
        - `CLIP`: The CLIP object.
    """
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

            clip_target.clip = FluxClip.flux_clip(dtype_t5=dtype_t5)
            clip_target.tokenizer = FluxClip.FluxTokenizer

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

class CLIPTextEncode:
    """#### Text encoding class for the CLIP model."""
    def encode(self, clip: CLIP, text: str, flux_enabled: bool = False) -> tuple:
        """#### Encode the input text.

        #### Args:
            - `clip` (CLIP): The CLIP object.
            - `text` (str): The input text.
            - `flux_enabled` (bool, optional): Whether to enable flux. Defaults to False.

        #### Returns:
            - `tuple`: The encoded text and the pooled output.
        """
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True, flux_enabled=flux_enabled)
        return ([[cond, {"pooled_output": pooled}]],)


class CLIPSetLastLayer:
    """#### Set the last layer class for the CLIP model."""
    def set_last_layer(self, clip: CLIP, stop_at_clip_layer: int) -> tuple:
        """#### Set the last layer of the CLIP model.
        
        works same as Automatic1111 clip skip
        
        #### Args:
            - `clip` (CLIP): The CLIP object.
            - `stop_at_clip_layer` (int): The layer to stop at.

        #### Returns:
            - `tuple`: Thefrom enum import Enum
        """
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)


class ClipTarget:
    """#### Target class for the CLIP model."""

    def __init__(self, tokenizer: object, clip: object):
        """#### Initialize the ClipTarget class.

        #### Args:
            - `tokenizer` (object): The tokenizer.
            - `clip` (object): The CLIP model.
        """
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}
