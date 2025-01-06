import json
import logging
import torch
from modules.Device import Device
from modules.clip import Clip
from modules.cond import cast


def gen_empty_tokens(special_tokens: dict, length: int) -> list:
    """#### Generate a list of empty tokens.

    #### Args:
        - `special_tokens` (dict): The special tokens.
        - `length` (int): The length of the token list.

    #### Returns:
        - `list`: The list of empty tokens.
    """
    start_token = special_tokens.get("start", None)
    end_token = special_tokens.get("end", None)
    pad_token = special_tokens.get("pad")
    output = []
    if start_token is not None:
        output.append(start_token)
    if end_token is not None:
        output.append(end_token)
    output += [pad_token] * (length - len(output))
    return output


class ClipTokenWeightEncoder:
    """#### Class representing a CLIP token weight encoder."""

    def encode_token_weights(self, token_weight_pairs: list) -> tuple:
        """#### Encode token weights.

        #### Args:
            - `token_weight_pairs` (list): The token weight pairs.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        out, pooled = self.encode(to_encode)
        first_pooled = pooled[0:1].to(Device.intermediate_device())

        output = []
        for k in range(0, sections):
            z = out[k : k + 1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        return torch.cat(output, dim=-2).to(Device.intermediate_device()), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """#### Uses the CLIP transformer encoder for text (from huggingface)."""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version: str = "openai/clip-vit-large-patch14",
        device: str = "cpu",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        layer_idx: int = None,
        textmodel_json_config: str = None,
        dtype: torch.dtype = None,
        model_class: type = Clip.CLIPTextModel,
        special_tokens: dict = {"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state: bool = True,
        enable_attention_masks: bool = False,
        return_projected_pooled: bool = True,
    ):
        """#### Initialize the SDClipModel.

        #### Args:
            - `version` (str, optional): The version of the model. Defaults to "openai/clip-vit-large-patch14".
            - `device` (str, optional): The device to use. Defaults to "cpu".
            - `max_length` (int, optional): The maximum length of the input. Defaults to 77.
            - `freeze` (bool, optional): Whether to freeze the model parameters. Defaults to True.
            - `layer` (str, optional): The layer to use. Defaults to "last".
            - `layer_idx` (int, optional): The index of the layer. Defaults to None.
            - `textmodel_json_config` (str, optional): The path to the JSON config file. Defaults to None.
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `model_class` (type, optional): The model class. Defaults to Clip.CLIPTextModel.
            - `special_tokens` (dict, optional): The special tokens. Defaults to {"start": 49406, "end": 49407, "pad": 49407}.
            - `layer_norm_hidden_state` (bool, optional): Whether to normalize the hidden state. Defaults to True.
            - `enable_attention_masks` (bool, optional): Whether to enable attention masks. Defaults to False.
            - `return_projected_pooled` (bool, optional): Whether to return the projected pooled output. Defaults to True.
        """
        super().__init__()
        assert layer in self.LAYERS

        if textmodel_json_config is None:
            textmodel_json_config = "./_internal/clip/sd1_clip_config.json"

        with open(textmodel_json_config) as f:
            config = json.load(f)

        self.transformer = model_class(config, dtype, device, cast.manual_cast)
        self.num_layers = self.transformer.num_layers

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens

        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks

        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.options_default = (
            self.layer,
            self.layer_idx,
            self.return_projected_pooled,
        )

    def freeze(self) -> None:
        """#### Freeze the model parameters."""
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def set_clip_options(self, options: dict) -> None:
        """#### Set the CLIP options.

        #### Args:
            - `options` (dict): The options to set.
        """
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        self.layer = "hidden"
        self.layer_idx = layer_idx

    def reset_clip_options(self) -> None:
        """#### Reset the CLIP options to default."""
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens: list, current_embeds: torch.nn.Embedding) -> list:
        """#### Set up the textual embeddings.

        #### Args:
            - `tokens` (list): The input tokens.
            - `current_embeds` (torch.nn.Embedding): The current embeddings.

        #### Returns:
            - `list`: The processed tokens.
        """
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:  # EOS token
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        logging.warning(
                            "WARNING: shape mismatch when trying to apply embedding, embedding will be ignored {} != {}".format(
                                y.shape[0], current_embeds.weight.shape[1]
                            )
                        )
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(
                next_new_token + 1,
                current_embeds.weight.shape[1],
                device=current_embeds.weight.device,
                dtype=current_embeds.weight.dtype,
            )
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]  # EOS embedding
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [
                list(map(lambda a: n if a == -1 else a, x))
            ]  # The EOS token should always be the largest one

        return processed_tokens

    def forward(self, tokens: list) -> tuple:
        """#### Forward pass of the model.

        #### Args:
            - `tokens` (list): The input tokens.

        #### Returns:
            - `tuple`: The output and the pooled output.
        """
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None

        outputs = self.transformer(
            tokens,
            attention_mask,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z.float(), pooled_output

    def encode(self, tokens: list) -> tuple:
        """#### Encode the input tokens.

        #### Args:
            - `tokens` (list): The input tokens.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        return self(tokens)

    def load_sd(self, sd: dict) -> None:
        """#### Load the state dictionary.

        #### Args:
            - `sd` (dict): The state dictionary.
        """
        return self.transformer.load_state_dict(sd, strict=False)


class SD1ClipModel(torch.nn.Module):
    """#### Class representing the SD1ClipModel."""

    def __init__(
        self, device: str = "cpu", dtype: torch.dtype = None, clip_name: str = "l", clip_model: type = SDClipModel, **kwargs
    ):
        """#### Initialize the SD1ClipModel.

        #### Args:
            - `device` (str, optional): The device to use. Defaults to "cpu".
            - `dtype` (torch.dtype, optional): The data type. Defaults to None.
            - `clip_name` (str, optional): The name of the CLIP model. Defaults to "l".
            - `clip_model` (type, optional): The CLIP model class. Defaults to SDClipModel.
            - `**kwargs`: Additional keyword arguments.
        """
        super().__init__()
        self.clip_name = clip_name
        self.clip = "clip_{}".format(self.clip_name)
        self.lowvram_patch_counter = 0
        self.model_loaded_weight_memory = 0
        setattr(self, self.clip, clip_model(device=device, dtype=dtype, **kwargs))

    def set_clip_options(self, options: dict) -> None:
        """#### Set the CLIP options.

        #### Args:
            - `options` (dict): The options to set.
        """
        getattr(self, self.clip).set_clip_options(options)

    def reset_clip_options(self) -> None:
        """#### Reset the CLIP options to default."""
        getattr(self, self.clip).reset_clip_options()

    def encode_token_weights(self, token_weight_pairs: dict) -> tuple:
        """#### Encode token weights.

        #### Args:
            - `token_weight_pairs` (dict): The token weight pairs.

        #### Returns:
            - `tuple`: The encoded tokens and the pooled output.
        """
        token_weight_pairs = token_weight_pairs[self.clip_name]
        out, pooled = getattr(self, self.clip).encode_token_weights(token_weight_pairs)
        return out, pooled