import torch
from modules.Attention import Attention

class CLIPTextModel_(torch.nn.Module):
    def __init__(
        self,
        config_dict: dict,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
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
        num_positions = config_dict["max_position_embeddings"]
        self.eos_token_id = config_dict["eos_token_id"]

        super().__init__()
        from modules.clip.Clip import CLIPEmbeddings, CLIPEncoder
        self.embeddings = CLIPEmbeddings(
            embed_dim,
            num_positions=num_positions,
            dtype=dtype,
            device=device,
            operations=operations,
        )
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
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor = None,
        intermediate_output: int = None,
        final_layer_norm_intermediate: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> tuple:
        """#### Forward pass for the CLIPTextModel_ module.

        #### Args:
            - `input_tokens` (torch.Tensor): The input tokens.
            - `attention_mask` (torch.Tensor, optional): The attention mask. Defaults to None.
            - `intermediate_output` (int, optional): The intermediate output layer. Defaults to None.
            - `final_layer_norm_intermediate` (bool, optional): Whether to apply final layer normalization to the intermediate output. Defaults to True.

        #### Returns:
            - `tuple`: The output tensor, the intermediate output tensor, and the pooled output tensor.
        """
        x = self.embeddings(input_tokens, dtype=dtype)
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

        causal_mask = (
            torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device)
            .fill_(float("-inf"))
            .triu_(1)
        )
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask

        x, i = self.encoder(x, mask=mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        pooled_output = x[
            torch.arange(x.shape[0], device=x.device),
            (
                torch.round(input_tokens).to(dtype=torch.int, device=x.device)
                == self.eos_token_id
            )
            .int()
            .argmax(dim=-1),
        ]
        return x, i, pooled_output

class CLIPTextModel(torch.nn.Module):
    def __init__(
        self,
        config_dict: dict,
        dtype: torch.dtype,
        device: torch.device,
        operations: object,
    ):
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