import torch
from src.Utilities import util
from src.NeuralNetwork import unet

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}


def load_lora(lora: dict, to_load: dict) -> dict:
    """#### Load a LoRA model.

    #### Args:
        - `lora` (dict): The LoRA model state dictionary.
        - `to_load` (dict): The keys to load from the LoRA model.

    #### Returns:
        - `dict`: The loaded LoRA model.
    """
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        "{}.dora_scale".format(x)
        dora_scale = None

        regular_lora = "{}.lora_up.weight".format(x)
        "{}_lora.up.weight".format(x)
        "{}.lora_linear_layer.up.weight".format(x)
        A_name = None

        if regular_lora in lora.keys():
            A_name = regular_lora
            B_name = "{}.lora_down.weight".format(x)
            "{}.lora_mid.weight".format(x)

        if A_name is not None:
            mid = None
            patch_dict[to_load[x]] = (
                "lora",
                (lora[A_name], lora[B_name], alpha, mid, dora_scale),
            )
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)
    return patch_dict


def model_lora_keys_clip(model: torch.nn.Module, key_map: dict = {}) -> dict:
    """#### Get the keys for a LoRA model's CLIP component.

    #### Args:
        - `model` (torch.nn.Module): The LoRA model.
        - `key_map` (dict, optional): The key map. Defaults to {}.

    #### Returns:
        - `dict`: The keys for the CLIP component.
    """
    sdk = model.state_dict().keys()

    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(32):
        for c in LORA_CLIP_MAP:
            k = "clip_l.transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k
                lora_key = "lora_te1_text_model_encoder_layers_{}_{}".format(
                    b, LORA_CLIP_MAP[c]
                )  # SDXL base
                key_map[lora_key] = k
                lora_key = "text_encoder.text_model.encoder.layers.{}.{}".format(
                    b, c
                )  # diffusers lora
                key_map[lora_key] = k
    return key_map


def model_lora_keys_unet(model: torch.nn.Module, key_map: dict = {}) -> dict:
    """#### Get the keys for a LoRA model's UNet component.

    #### Args:
        - `model` (torch.nn.Module): The LoRA model.
        - `key_map` (dict, optional): The key map. Defaults to {}.

    #### Returns:
        - `dict`: The keys for the UNet component.
    """
    sdk = model.state_dict().keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            key_lora = k[len("diffusion_model.") : -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
            key_map["lora_prior_unet_{}".format(key_lora)] = k  # cascade lora:

    diffusers_keys = unet.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            key_lora = k[: -len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = unet_key

            diffusers_lora_prefix = ["", "unet."]
            for p in diffusers_lora_prefix:
                diffusers_lora_key = "{}{}".format(
                    p, k[: -len(".weight")].replace(".to_", ".processor.to_")
                )
                if diffusers_lora_key.endswith(".to_out.0"):
                    diffusers_lora_key = diffusers_lora_key[:-2]
                key_map[diffusers_lora_key] = unet_key
    return key_map


def load_lora_for_models(
    model: object, clip: object, lora: dict, strength_model: float, strength_clip: float
) -> tuple:
    """#### Load a LoRA model for the given models.

    #### Args:
        - `model` (object): The model.
        - `clip` (object): The CLIP model.
        - `lora` (dict): The LoRA model state dictionary.
        - `strength_model` (float): The strength of the model.
        - `strength_clip` (float): The strength of the CLIP model.

    #### Returns:
        - `tuple`: The new model patcher and CLIP model.
    """
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

    loaded = load_lora(lora, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)

    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)

    return (new_modelpatcher, new_clip)


class LoraLoader:
    """#### Class for loading LoRA models."""

    def __init__(self):
        """#### Initialize the LoraLoader class."""
        self.loaded_lora = None

    def load_lora(
        self,
        model: object,
        clip: object,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
    ) -> tuple:
        """#### Load a LoRA model.

        #### Args:
            - `model` (object): The model.
            - `clip` (object): The CLIP model.
            - `lora_name` (str): The name of the LoRA model.
            - `strength_model` (float): The strength of the model.
            - `strength_clip` (float): The strength of the CLIP model.

        #### Returns:
            - `tuple`: The new model patcher and CLIP model.
        """
        lora_path = util.get_full_path("loras", lora_name)
        lora = None
        if lora is None:
            lora = util.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)
