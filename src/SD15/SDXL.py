"""SDXL model configurations for LightDiffusion.

This module provides SDXL and SDXL Refiner model configurations, adapted from
ComfyUI's implementation but using local LightDiffusion modules.
"""

from src.Model import ModelBase
from src.Utilities import Latent, util
from src.SD15 import SDXLClip
from src.clip import Clip
from src.sample import sampling


class SDXLRefiner(ModelBase.BASE):
    """SDXL Refiner model configuration."""

    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = Latent.SDXL
    memory_usage_factor = 1.0

    def get_model(self, state_dict, prefix="", device=None):
        """Get the refiner model instance.

        Args:
            state_dict: Model state dictionary
            prefix: Key prefix for state dict
            device: Device to load model on

        Returns:
            SDXLRefiner model instance
        """
        return ModelBase.SDXLRefiner(self, device=device)

    def process_clip_state_dict(self, state_dict):
        """Process CLIP state dict for refiner (G model only).

        Args:
            state_dict: Raw state dictionary

        Returns:
            Processed state dictionary
        """
        replace_prefix = {}
        replace_prefix["conditioner.embedders.0.model."] = "clip_g."
        state_dict = util.state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )

        state_dict = util.clip_text_transformers_convert(
            state_dict, "clip_g.", "clip_g.transformer."
        )
        return state_dict

    def clip_target(self, state_dict=None):
        """Return the CLIP target for refiner.

        Args:
            state_dict: Optional state dictionary

        Returns:
            ClipTarget for SDXL Refiner (G model only)
        """
        return Clip.ClipTarget(SDXLClip.SDXLTokenizer, SDXLClip.SDXLRefinerClipModel)


class SDXL(ModelBase.BASE):
    """SDXL model configuration."""

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = Latent.SDXL
    memory_usage_factor = 0.8

    def process_vae_state_dict(self, state_dict):
        """Process VAE state dict for SDXL.
        
        Detects if the VAE is a 'flux-style' VAE (missing post_quant_conv)
        and sets the flag for decoding logic.
        """
        if "post_quant_conv.weight" not in state_dict:
            # If missing post_quant_conv, it's a Flux-style VAE
            self.vae_config = {"flux": True}
        return state_dict

    def model_type(self, state_dict, prefix=""):
        """Detect the model type from state dict.

        Args:
            state_dict: Model state dictionary
            prefix: Key prefix

        Returns:
            ModelType enum value
        """
        # Check for Playground V2.5
        if "edm_mean" in state_dict and "edm_std" in state_dict:
            self.latent_format = Latent.SDXL_Playground_2_5()
            self.sampling_settings["sigma_data"] = 0.5
            self.sampling_settings["sigma_max"] = 80.0
            self.sampling_settings["sigma_min"] = 0.002
            return sampling.ModelType.EDM
        # Check for V-prediction EDM variant
        elif "edm_vpred.sigma_max" in state_dict:
            self.sampling_settings["sigma_max"] = float(
                state_dict["edm_vpred.sigma_max"].item()
            )
            if "edm_vpred.sigma_min" in state_dict:
                self.sampling_settings["sigma_min"] = float(
                    state_dict["edm_vpred.sigma_min"].item()
                )
            return sampling.ModelType.V_PREDICTION_EDM
        # Check for V-prediction
        elif "v_pred" in state_dict:
            if "ztsnr" in state_dict:  # Some zsnr anime checkpoints
                self.sampling_settings["zsnr"] = True
            return sampling.ModelType.V_PREDICTION
        else:
            return sampling.ModelType.EPS

    def get_model(self, state_dict, prefix="", device=None):
        """Get the SDXL model instance.

        Args:
            state_dict: Model state dictionary
            prefix: Key prefix for state dict
            device: Device to load model on

        Returns:
            SDXL model instance
        """
        out = ModelBase.SDXL(
            self, model_type=self.model_type(state_dict, prefix), device=device
        )
        if self.inpaint_model():
            out.set_inpaint()
        return out

    def process_clip_state_dict(self, state_dict):
        """Process CLIP state dict for SDXL (dual L+G models).

        Args:
            state_dict: Raw state dictionary

        Returns:
            Processed state dictionary
        """
        replace_prefix = {}
        replace_prefix[
            "conditioner.embedders.0.transformer.text_model"
        ] = "clip_l.transformer.text_model"
        replace_prefix["conditioner.embedders.1.model."] = "clip_g."
        state_dict = util.state_dict_prefix_replace(
            state_dict, replace_prefix, filter_keys=True
        )

        state_dict = util.clip_text_transformers_convert(
            state_dict, "clip_g.", "clip_g.transformer."
        )
        return state_dict

    def clip_target(self, state_dict=None):
        """Return the CLIP target for SDXL.

        Args:
            state_dict: Optional state dictionary

        Returns:
            ClipTarget for SDXL (dual L+G models)
        """
        return Clip.ClipTarget(SDXLClip.SDXLTokenizer, SDXLClip.SDXLClipModel)


class SSD1B(SDXL):
    """SSD-1B model configuration (SDXL variant with fewer transformer blocks)."""

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 4, 4],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class Segmind_Vega(SDXL):
    """Segmind Vega model configuration (SDXL variant)."""

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 1, 1, 2, 2],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class KOALA_700M(SDXL):
    """KOALA 700M model configuration (SDXL variant)."""

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 5],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }


class KOALA_1B(SDXL):
    """KOALA 1B model configuration (SDXL variant)."""

    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 2, 6],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }
