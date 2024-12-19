from enum import Enum
import torch

from modules.Model import ModelPatcher
from modules.Attention import Attention
from modules.Device import Device


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
        optimized_attention = Attention.optimized_attention_for_device(
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



class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
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
        Device.load_model_gpu(self.patcher)
        return self.patcher


class CLIPType(Enum):
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    
class CLIPTextEncode:
    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)
    
class CLIPSetLastLayer:
    def set_last_layer(self, clip, stop_at_clip_layer):
        clip = clip.clone()
        clip.clip_layer(stop_at_clip_layer)
        return (clip,)
    
class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}