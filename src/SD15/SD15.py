import torch
from src.BlackForest import Flux
from src.Utilities import util
from src.Model import ModelBase
from src.SD15 import SDClip, SDToken
from src.Utilities import Latent
from src.clip import Clip


class sm_SD15(ModelBase.BASE):
    """#### Class representing the SD15 model.

    #### Args:
        - `ModelBase.BASE` (ModelBase.BASE): The base model class.
    """

    unet_config: dict = {
        "context_dim": 768,
        "model_channels": 320,
        "use_linear_in_transformer": False,
        "adm_in_channels": None,
        "use_temporal_attention": False,
    }

    unet_extra_config: dict = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format: Latent.SD15 = Latent.SD15

    def process_clip_state_dict(self, state_dict: dict) -> dict:
        """#### Process the state dictionary for the CLIP model.

        #### Args:
            - `state_dict` (dict): The state dictionary.

        #### Returns:
            - `dict`: The processed state dictionary.
        """
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith(
                "cond_stage_model.transformer.text_model."
            ):
                y = x.replace(
                    "cond_stage_model.transformer.",
                    "cond_stage_model.transformer.text_model.",
                )
                state_dict[y] = state_dict.pop(x)

        if (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
            in state_dict
        ):
            ids = state_dict[
                "cond_stage_model.transformer.text_model.embeddings.position_ids"
            ]
            if ids.dtype == torch.float32:
                state_dict[
                    "cond_stage_model.transformer.text_model.embeddings.position_ids"
                ] = ids.round()

        replace_prefix = {}
        replace_prefix["cond_stage_model."] = "clip_l."
        state_dict = util.state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )
        return state_dict

    def clip_target(self) -> Clip.ClipTarget:
        """#### Get the target CLIP model.

        #### Returns:
            - `Clip.ClipTarget`: The target CLIP model.
        """
        return Clip.ClipTarget(SDToken.SD1Tokenizer, SDClip.SD1ClipModel)
    
models = [
    sm_SD15, Flux.Flux
]
