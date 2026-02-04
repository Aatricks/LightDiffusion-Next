"""SD1.5 model implementation."""
import torch
from src.Utilities import util
from src.Model import ModelBase
from src.SD15 import SDClip, SDToken, SDXL
from src.Utilities import Latent
from src.clip import Clip


class sm_SD15(ModelBase.BASE):
    """SD1.5 model class."""
    unet_config: dict = {
        "context_dim": 768, "model_channels": 320, "use_linear_in_transformer": False,
        "adm_in_channels": None, "use_temporal_attention": False,
    }
    unet_extra_config: dict = {"num_heads": 8, "num_head_channels": -1}
    latent_format: Latent.SD15 = Latent.SD15

    def process_clip_state_dict(self, state_dict: dict) -> dict:
        """Process CLIP state dict for SD1.5."""
        k = list(state_dict.keys())
        for x in k:
            if x.startswith("cond_stage_model.transformer.") and not x.startswith("cond_stage_model.transformer.text_model."):
                state_dict[x.replace("cond_stage_model.transformer.", "cond_stage_model.transformer.text_model.")] = state_dict.pop(x)

        pos_key = "cond_stage_model.transformer.text_model.embeddings.position_ids"
        if pos_key in state_dict and state_dict[pos_key].dtype == torch.float32:
            state_dict[pos_key] = state_dict[pos_key].round()

        return util.state_dict_prefix_replace(state_dict, {"cond_stage_model.": "clip_l."}, filter_keys=True)

    def clip_target(self) -> Clip.ClipTarget:
        """Get CLIP target for SD1.5."""
        return Clip.ClipTarget(SDToken.SD1Tokenizer, SDClip.SD1ClipModel)


models = [sm_SD15, SDXL.SDXLRefiner, SDXL.SDXL, SDXL.SSD1B, SDXL.Segmind_Vega, SDXL.KOALA_700M, SDXL.KOALA_1B]
