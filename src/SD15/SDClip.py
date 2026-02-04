"""SD1.5 CLIP text encoder implementation."""
import json
import logging
import numbers
import torch
from src.Device import Device
from src.cond import cast
from src.clip.CLIPTextModel import CLIPTextModel


def gen_empty_tokens(special_tokens: dict, length: int) -> list:
    """Generate list of empty tokens for padding."""
    start = special_tokens.get("start")
    end = special_tokens.get("end")
    pad = special_tokens.get("pad")
    output = []
    if start is not None: output.append(start)
    if end is not None: output.append(end)
    return output + [pad] * (length - len(output))


class ClipTokenWeightEncoder:
    """CLIP token weight encoder mixin."""

    def encode_token_weights(self, token_weight_pairs: list) -> tuple:
        """Encode tokens with weights."""
        to_encode = []
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = [a[0] for a in x]
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(a[1] == 1.0 for a in x)
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        o = self.encode(to_encode)
        out, pooled = o[:2]
        first_pooled = pooled[0:1].to(Device.intermediate_device()) if pooled is not None else None

        output = []
        for k in range(sections):
            z = out[k:k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if not output:
            r = (out[-1:].to(Device.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(Device.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(Device.intermediate_device())
                extra[k] = v
            r = r + (extra,)
        return r


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """CLIP transformer encoder for text (SD1.5 compatible)."""
    LAYERS = ["last", "pooled", "hidden"]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77, freeze=True,
                 layer="last", layer_idx=None, textmodel_json_config=None, dtype=None, model_class=CLIPTextModel,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True,
                 enable_attention_masks=False, zero_out_masked=False, return_projected_pooled=True,
                 return_attention_masks=False, model_options={}):
        super().__init__()
        assert layer in self.LAYERS

        textmodel_json_config = textmodel_json_config or "./include/clip/sd1_clip_config.json"
        with open(textmodel_json_config) as f:
            config = json.load(f)

        self.operations = model_options.get("custom_operations") or cast.manual_cast
        self.transformer = model_class(config, dtype, device, self.operations)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        if freeze: self.freeze()
        
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = torch.nn.Parameter(torch.full((1,), 4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.zero_out_masked = zero_out_masked
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.return_attention_masks = return_attention_masks

        if layer == "hidden":
            assert layer_idx is not None and abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options: dict):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer, self.layer_idx, self.return_projected_pooled = self.options_default

    def set_up_textual_embeddings(self, tokens: list, current_embeds: torch.nn.Embedding) -> list:
        """Process tokens and set up custom embeddings."""
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0]
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, numbers.Integral):
                    tokens_temp.append(int(y))
                elif y.shape[0] == current_embeds.weight.shape[1]:
                    embedding_weights.append(y)
                    tokens_temp.append(next_new_token)
                    next_new_token += 1
                else:
                    logging.warning(f"Embedding shape mismatch: {y.shape[0]} != {current_embeds.weight.shape[1]}")
            while len(tokens_temp) < len(x):
                tokens_temp.append(self.special_tokens["pad"])
            out_tokens.append(tokens_temp)

        n = token_dict_size
        if embedding_weights:
            new_embedding = self.operations.Embedding(next_new_token + 1, current_embeds.weight.shape[1],
                                                       device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            with torch.no_grad():
                new_embedding.weight[:token_dict_size] = current_embeds.weight
                for x in embedding_weights:
                    new_embedding.weight[n] = x
                    n += 1
            self.transformer.set_input_embeddings(new_embedding)

        return [[n if a == -1 else a for a in x] for x in out_tokens]

    def forward(self, tokens: list) -> tuple:
        """Forward pass returning embeddings and pooled output."""
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if self.enable_attention_masks or self.zero_out_masked or self.return_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            end_token = self.special_tokens.get("end", -1)
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == end_token:
                        break

        outputs = self.transformer(tokens, attention_mask if self.enable_attention_masks else None,
                                   intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state,
                                   dtype=torch.float32)
        self.transformer.set_input_embeddings(backup_embeds)

        z = outputs[0].float() if self.layer == "last" else outputs[1].float()
        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        if self.return_attention_masks:
            return z, pooled_output, {"attention_mask": attention_mask}
        return z, pooled_output

    def encode(self, tokens: list) -> tuple:
        return self(tokens)

    def load_sd(self, sd: dict):
        return self.transformer.load_state_dict(sd, strict=False)


class SD1ClipModel(torch.nn.Module):
    """SD1 CLIP model wrapper."""

    def __init__(self, device="cpu", dtype=None, clip_name="l", clip_model=SDClipModel, **kwargs):
        super().__init__()
        self.clip_name = clip_name
        self.clip = f"clip_{clip_name}"
        self.lowvram_patch_counter = 0
        self.model_loaded_weight_memory = 0
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def set_clip_options(self, options: dict):
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self):
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
        token_weight_pairs = token_weight_pairs[self.clip_name]
        return getattr(self, self.clip).encode_token_weights(token_weight_pairs)
