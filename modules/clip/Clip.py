from enum import Enum
import torch

from modules.Model import ModelPatcher
from modules.Attention import Attention
from modules.Device import Device


class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, heads: int, dtype: torch.dtype, device: torch.device, operations: object):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, optimized_attention: callable = None) -> torch.Tensor:
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
    def __init__(self, embed_dim: int, intermediate_size: int, activation: str, dtype: torch.dtype, device: torch.device, operations: object):
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
    def __init__(self, embed_dim: int, heads: int, intermediate_size: int, intermediate_activation: str, dtype: torch.dtype, device: torch.device, operations: object):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, optimized_attention: callable = None) -> torch.Tensor:
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
    def __init__(self, num_layers: int, embed_dim: int, heads: int, intermediate_size: int, intermediate_activation: str, dtype: torch.dtype, device: torch.device, operations: object):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, intermediate_output: int = None) -> tuple:
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
        for i, l in enumerate(self.layers):
            x = l(x, mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int = 49408, num_positions: int = 77, dtype: torch.dtype = None, device: torch.device = None):
        """#### Initialize the CLIPEmbeddings module.

        #### Args:
            - `embed_dim` (int): The embedding dimension.
            - `vocab_size` (int, optional): The vocabulary size. Defaults to 49408.
            - `num_positions` (int, optional): The number of positions. Defaults to 77.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `device` (torch.device, optional): The device to use. Defaults to None.
        """
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            vocab_size, embed_dim, dtype=dtype, device=device
        )
        self.position_embedding = torch.nn.Embedding(
            num_positions, embed_dim, dtype=dtype, device=device
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """#### Forward pass for the CLIPEmbeddings module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):
    def __init__(self, config_dict: dict, dtype: torch.dtype, device: torch.device, operations: object):
        """#### Initialize the CLIPTextModel_ module.

        #### Args:
            - `config_dict` (dict): The configuration dictionary.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
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

    def forward(self, input_tokens: torch.Tensor, attention_mask: torch.Tensor = None, intermediate_output: int = None, final_layer_norm_intermediate: bool = True) -> tuple:
        """#### Forward pass for the CLIPTextModel_ module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.
            - `attention_mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `intermediate_output` (int, optional): The intermediate output layer. Defaults to None.
            - `final_layer_norm_intermediate` (bool, optional): Whether to apply final layer normalization to the intermediate output. Defaults to True.

        #### Returns:
            - `tuple`: The output tensor, the intermediate output tensor, and the pooled output tensor.
        """
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
    def __init__(self, config_dict: dict, dtype: torch.dtype, device: torch.device, operations: object):
        """#### Initialize the CLIPTextModel module.

        #### Args:
            - `config_dict` (dict): The configuration dictionary.
            - `dtype` (torch.dtype): The data type.
            - `device` (torch.device): The device to use.
            - `operations` (object): The operations object.
        """
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config_dict, dtype, device, operations)
        embed_dim = config_dict["hidden_size"]
        self.text_projection = operations.Linear(
            embed_dim, embed_dim, bias=False, dtype=dtype, device=device
        )
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """#### Get the input embeddings.

        #### Returns:
            - `torch.nn.Embedding`: The input embeddings.
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings: torch.nn.Embedding) -> None:
        """#### Set the input embeddings.

        #### Args:
            - `embeddings` (torch.nn.Embedding): The input embeddings.
        """
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs) -> tuple:
        """#### Forward pass for the CLIPTextModel module.

        #### Args:
            - `*args`: Variable length argument list.
            - `**kwargs`: Arbitrary keyword arguments.

        #### Returns:
            - `tuple`: The output tensors.
        """
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class CLIP:
    def __init__(self, target: object = None, embedding_directory: str = None, no_init: bool = False):
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

        load_device = Device.text_encoder_device()
        offload_device = Device.text_encoder_offload_device()
        params["device"] = offload_device
        params["dtype"] = Device.text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = ModelPatcher.ModelPatcher(
            self.cond_stage_model,
            load_device=load_device,
            offload_device=offload_device,
        )
        self.layer_idx = None

    def clone(self) -> 'CLIP':
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

    def add_patches(self, patches: list, strength_patch: float = 1.0, strength_model: float = 1.0) -> None:
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

    def encode_from_tokens(self, tokens: list, return_pooled: bool = False) -> tuple:
        """#### Encode the input tokens.

        #### Args:
            - `tokens` (list): The input tokens.
            - `return_pooled` (bool, optional): Whether to return the pooled output. Defaults to False.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
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

    def load_sd(self, sd: dict, full_model: bool = False) -> None:
        """#### Load the state dictionary.

        #### Args:
            - `sd` (dict): The state dictionary.
            - `full_model` (bool, optional): Whether to load the full model. Defaults to False.
        """
        return self.cond_stage_model.load_state_dict(sd, strict=False)

    def load_model(self) -> ModelPatcher:
        """#### Load the model.

        #### Returns:
            - `ModelPatcher`: The model patcher.
        """
        Device.load_model_gpu(self.patcher)
        return self.patcher


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2


class CLIPTextEncode:
    def encode(self, clip: CLIP, text: str) -> tuple:
        """#### Encode the input text.

        #### Args:
            - `clip` (CLIP): The CLIP object.
            - `text` (str): The input text.

        #### Returns:
            - `tuple`: The encoded text and the pooled output.
        """
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class CLIPSetLastLayer:
    def set_last_layer(self, clip: CLIP, stop_at_clip_layer: int) -> tuple:
        """#### Set the last layer of the CLIP model.

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
    """#### Target class for the CLIP model.
    """
    def __init__(self, tokenizer: object, clip: object):
        """#### Initialize the ClipTarget class.
        
        #### Args:
            - `tokenizer` (object): The tokenizer.
            - `clip` (object): The CLIP model.
        """
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}