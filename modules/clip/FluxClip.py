import math
import os
import torch
from modules.Attention import Attention
from modules.Device import Device
from modules.SD15 import SDClip, SDToken
from modules.cond import cast
from transformers import T5TokenizerFast

activations = {
    "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}

class T5DenseGatedActDense(torch.nn.Module):
    """#### Dense Gated Activation Layer"""
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, dtype: torch.dtype, device: torch.device, operations):
        """#### Initialize Dense Gated Activation Layer

        #### Args:
            - `model_dim` (int): Model dimension.
            - `ff_dim` (int): Feedforward dimension.
            - `ff_activation` (str): Feedforward activation function.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        hidden_gelu = self.act(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        # x = self.dropout(x)
        x = self.wo(x)
        return x


class T5LayerFF(torch.nn.Module):
    """#### Feedforward Layer"""
    def __init__(
        self, model_dim: int, ff_dim: int, ff_activation: str, gated_act: bool, dtype: torch.dtype, device: torch.device, operations
    ):
        """#### Initialize Feedforward Layer
        
        #### Args:
            - `model_dim` (int): Model dimension.
            - `ff_dim` (int): Feedforward dimension.
            - `ff_activation` (str): Feedforward activation function.
            - `gated_act` (bool): Whether to use gated activation.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
        super().__init__()
        if gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                model_dim, ff_dim, ff_activation, dtype, device, operations
            )

        self.layer_norm = T5LayerNorm(
            model_dim, dtype=dtype, device=device, operations=operations
        )
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        # x = x + self.dropout(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):
    """#### Attention Layer"""
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        num_heads: int,
        relative_attention_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
        operations,
    ):
        """#### Initialize Attention Layer
        
        #### Args:
            - `model_dim` (int): Model dimension.
            - `inner_dim` (int): Inner dimension.
            - `num_heads` (int): Number of attention heads.
            - `relative_attention_bias` (bool): Whether to use relative attention bias.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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
        relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> torch.Tensor:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        #### Args:
            - `relative_position` (torch.Tensor): Relative position tensor.
            - `bidirectional` (bool): Whether the attention is bidirectional.
            - `num_buckets` (int): Number of buckets.
            - `max_distance` (int): Maximum distance.

        #### Returns:
            - `torch.Tensor`: Bucketed relative positions.
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

    def compute_bias(self, query_length: int, key_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """#### Compute binned relative position bias
        
        #### Args:
            - `query_length` (int): Length of the query.
            - `key_length` (int): Length of the key.
            - `device` (torch.device): Device.
            - `dtype` (torch.dtype): Data type.
            
        #### Returns:
            - `torch.Tensor`: Computed bias.
        """
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_bias: torch.Tensor = None, optimized_attention = None) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            - `mask` (torch.Tensor, optional): Attention mask. Defaults to None.
            - `past_bias` (torch.Tensor, optional): Past bias. Defaults to None.
            - `optimized_attention` (callable, optional): Optimized attention function. Defaults to None.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
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
    """#### Self-Attention Layer"""
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        num_heads: int,
        relative_attention_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
        operations,
    ):
        """#### Initialize Self-Attention Layer
        
        #### Args:
            - `model_dim` (int): Model dimension.
            - `inner_dim` (int): Inner dimension.
            - `ff_dim` (int): Feedforward dimension.
            - `num_heads` (int): Number of attention heads.
            - `relative_attention_bias` (bool): Whether to use relative attention bias.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_bias: torch.Tensor = None, optimized_attention = None) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            - `mask` (torch.Tensor, optional): Attention mask. Defaults to None.
            - `past_bias` (torch.Tensor, optional): Past bias. Defaults to None.
            - `optimized_attention` (callable, optional): Optimized attention function. Defaults to None.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
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
    """#### T5 Block"""
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention_bias: bool,
        dtype: torch.dtype,
        device: torch.device,
        operations,
    ):
        """#### Initialize T5 Block
        
        #### Args:
            - `model_dim` (int): Model dimension.
            - `inner_dim` (int): Inner dimension.
            - `ff_dim` (int): Feedforward dimension.
            - `ff_activation` (str): Feedforward activation function.
            - `gated_act` (bool): Whether to use gated activation.
            - `num_heads` (int): Number of attention heads.
            - `relative_attention_bias` (bool): Whether to use relative attention bias.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, past_bias: torch.Tensor = None, optimized_attention = None) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            - `mask` (torch.Tensor, optional): Attention mask. Defaults to None.
            - `past_bias` (torch.Tensor, optional): Past bias. Defaults to None.
            - `optimized_attention` (callable, optional): Optimized attention function. Defaults to None.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        x, past_bias = self.layer[0](x, mask, past_bias, optimized_attention)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):
    """#### T5 Stack"""
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention: bool,
        dtype: torch.dtype,
        device: torch.device,
        operations,
    ):
        """#### Initialize T5 Stack
        
        #### Args:
            - `num_layers` (int): Number of layers.
            - `model_dim` (int): Model dimension.
            - `inner_dim` (int): Inner dimension.
            - `ff_dim` (int): Feedforward dimension.
            - `ff_activation` (str): Feedforward activation function.
            - `gated_act` (bool): Whether to use gated activation.
            - `num_heads` (int): Number of attention heads.
            - `relative_attention` (bool): Whether to use relative attention.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        intermediate_output: int = None,
        final_layer_norm_intermediate: bool = True,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """#### Forward Pass
        
        #### Args:
            - `x` (torch.Tensor): Input tensor.
            - `attention_mask` (torch.Tensor, optional): Attention mask. Defaults to None.
            - `intermediate_output` (int, optional): Intermediate output index. Defaults to None.
            - `final_layer_norm_intermediate` (bool, optional): Whether to apply final layer norm to intermediate output. Defaults to True.
            - `dtype` (torch.dtype, optional): Data type. Defaults to None.
            
        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
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
        optimized_attention = Attention.optimized_attention_for_device()
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
        """#### Initialize T5 Model

        #### Args:
            - `config_dict` (dict): Configuration dictionary.
            - `dtype` (torch.dtype): Data type.
            - `device` (torch.device): Device.
            - `operations` (Operations): Operations.
        """
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

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """#### Get input embeddings

        #### Returns:
            - `torch.nn.Embedding`: The input embeddings.
        """
        return self.shared

    def set_input_embeddings(self, embeddings: torch.nn.Embedding) -> None:
        """#### Set input embeddings

        #### Args:
            - `embeddings` (torch.nn.Embedding): The input embeddings.
        """
        self.shared = embeddings

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """#### Forward pass

        #### Args:
            - `input_ids` (torch.Tensor): Input tensor.
            - `*args`: Additional arguments.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.nan_to_num(x)  # Fix for fp8 T5 base
        return self.encoder(x, *args, **kwargs)

class T5XXLModel(SDClip.SDClipModel):
    def __init__(
        self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}
    ):
        """#### Initialize T5XXL Model

        #### Args:
            - `device` (str, optional): Device. Defaults to "cpu".
            - `layer` (str, optional): Layer. Defaults to "last".
            - `layer_idx` (int, optional): Layer index. Defaults to None.
            - `dtype` (torch.dtype, optional): Data type. Defaults to None.
            - `model_options` (dict, optional): Model options. Defaults to {}.
        """
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
        """#### Initialize T5XXL Tokenizer

        #### Args:
            - `embedding_directory` (str, optional): Embedding directory. Defaults to None.
            - `tokenizer_data` (dict, optional): Tokenizer data. Defaults to {}.
        """
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

class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None, operations=None):
        """#### Initialize T5 Layer Normalization

        #### Args:
            - `hidden_size` (int): Hidden size.
            - `eps` (float, optional): Epsilon. Defaults to 1e-6.
            - `dtype` (torch.dtype, optional): Data type. Defaults to None.
            - `device` (torch.device, optional): Device. Defaults to None.
            - `operations` (Operations, optional): Operations. Defaults to None.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """#### Forward pass

        #### Args:
            - `x` (torch.Tensor): Input tensor.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return cast.cast_to_input(self.weight, x) * x

class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        """#### Initialize Flux Tokenizer

        #### Args:
            - `embedding_directory` (str, optional): Embedding directory. Defaults to None.
            - `tokenizer_data` (dict, optional): Tokenizer data. Defaults to {}.
        """
        clip_l_tokenizer_class = tokenizer_data.get(
            "clip_l_tokenizer_class", SDToken.SDTokenizer
        )
        self.clip_l = clip_l_tokenizer_class(embedding_directory=embedding_directory)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False) -> dict:
        """#### Tokenize text with weights

        #### Args:
            - `text` (str): Text to tokenize.
            - `return_word_ids` (bool, optional): Whether to return word IDs. Defaults to False.

        #### Returns:
            - `dict`: Tokenized text with weights.
        """
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out

class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options={}):
        """#### Initialize FluxClip Model

        #### Args:
            - `dtype_t5` (torch.dtype, optional): T5 data type. Defaults to None.
            - `device` (str, optional): Device. Defaults to "cpu".
            - `dtype` (torch.dtype, optional): Data type. Defaults to None.
            - `model_options` (dict, optional): Model options. Defaults to {}.
        """
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

    def reset_clip_options(self) -> None:
        """#### Reset CLIP options"""
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
        """#### Encode token weights

        #### Args:
            - `token_weight_pairs` (dict): Token weight pairs.

        #### Returns:
            - `tuple`: Encoded token weights.
        """
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd: dict) -> None:
        """#### Load state dictionary

        #### Args:
            - `sd` (dict): State dictionary.
        """
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def flux_clip(dtype_t5=None):
    """#### Create FluxClip Model

    #### Args:
        - `dtype_t5` (torch.dtype, optional): T5 data type. Defaults to None.

    #### Returns:
        - `FluxClipModel`: FluxClip Model class.
    """
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            """#### Initialize FluxClip Model

            #### Args:
                - `device` (str, optional): Device. Defaults to "cpu".
                - `dtype` (torch.dtype, optional): Data type. Defaults to None.
                - `model_options` (dict, optional): Model options. Defaults to {}.
            """
            super().__init__(
                dtype_t5=dtype_t5,
                device=device,
                dtype=dtype,
                model_options=model_options,
            )

    return FluxClipModel_