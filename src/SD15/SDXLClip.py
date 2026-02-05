"""SDXL CLIP model implementation for LightDiffusion.

This module provides SDXL-specific CLIP tokenizers and models, adapted from
ComfyUI's implementation but using local LightDiffusion modules.
"""

import torch
from src.SD15 import SDClip, SDToken


class SDXLClipG(SDClip.SDClipModel):
    """SDXL ClipG model - uses the larger G model with different layer settings."""

    def __init__(
        self,
        device="cpu",
        max_length=77,
        freeze=True,
        layer="penultimate",
        layer_idx=None,
        dtype=None,
        model_options=None,
    ):
        """Initialize SDXLClipG model.

        Args:
            device: Device to load model on
            max_length: Maximum token length
            freeze: Whether to freeze weights
            layer: Layer type ('penultimate' maps to 'hidden')
            layer_idx: Specific layer index (defaults to -2 for penultimate)
            dtype: Data type for model weights
            model_options: Additional model options (optional)
        """
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2

        # Use the bigg config for SDXL's G model (in include/clip directory)
        textmodel_json_config = "./include/clip/clip_config_bigg.json"

        super().__init__(
            device=device,
            freeze=freeze,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config=textmodel_json_config,
            dtype=dtype,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
            return_projected_pooled=True,
        )

    def load_sd(self, sd):
        """Load state dict into model."""
        return super().load_sd(sd)


class SDXLClipGTokenizer(SDToken.SDTokenizer):
    """Tokenizer for SDXL ClipG model."""

    def __init__(
        self, tokenizer_path=None, embedding_directory=None, tokenizer_data=None
    ):
        """Initialize SDXLClipGTokenizer.

        Args:
            tokenizer_path: Path to tokenizer config
            embedding_directory: Directory containing embeddings
            tokenizer_data: Pre-loaded tokenizer data dict (ignored for compatibility)
        """
        # Note: tokenizer_data is accepted for compatibility but not used
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=1280,
            embedding_key="clip_g",
        )


class SDXLTokenizer:
    """Dual tokenizer for SDXL (combines L and G models)."""

    def __init__(self, embedding_directory=None, tokenizer_data=None):
        """Initialize SDXLTokenizer with both L and G tokenizers.

        Args:
            embedding_directory: Directory containing embeddings
            tokenizer_data: Pre-loaded tokenizer data dict (ignored for compatibility)
        """
        # Note: tokenizer_data is accepted for compatibility but not used
        self.clip_l = SDToken.SDTokenizer(embedding_directory=embedding_directory)
        self.clip_g = SDXLClipGTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        """Tokenize text with both L and G tokenizers.

        Args:
            text: Input text to tokenize
            return_word_ids: Whether to return word IDs
            **kwargs: Additional arguments

        Returns:
            Dict with 'g' and 'l' tokenization results
        """
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        """Convert tokens back to text using G tokenizer."""
        return self.clip_g.untokenize(token_weight_pair)

    def state_dict(self):
        """Return empty state dict (tokenizer has no trainable params)."""
        return {}


class SDXLClipModel(torch.nn.Module):
    """SDXL CLIP model combining both L and G encoders."""

    def __init__(self, device="cpu", dtype=None, model_options=None):
        """Initialize SDXL CLIP model with both L and G components.

        Args:
            device: Device to load models on
            dtype: Data type for model weights
            model_options: Additional model options (optional)
        """
        super().__init__()
        if model_options is None:
            model_options = {}
        self.clip_l = SDClip.SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device=device,
            dtype=dtype,
            layer_norm_hidden_state=False,
        )
        self.clip_g = SDXLClipG(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = {dtype} if dtype else set()

    def set_clip_options(self, options):
        """Set options for both CLIP models."""
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        """Reset options for both CLIP models."""
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle shape mismatches.

        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching

        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        # Filter out keys with shape mismatches
        filtered_sd = {}
        for k, v in state_dict.items():
            # Handle logit_scale shape mismatch (scalar vs 1D)
            if "logit_scale" in k and k in self.state_dict():
                expected_shape = self.state_dict()[k].shape
                if v.shape != expected_shape and v.numel() == 1 and len(expected_shape) == 1:
                    # Reshape scalar to 1D
                    filtered_sd[k] = v.reshape(expected_shape)
                    continue
            filtered_sd[k] = v
        
        # Call parent load_state_dict with filtered dict
        return super().load_state_dict(filtered_sd, strict=strict)

    def encode_token_weights(self, token_weight_pairs):
        """Encode tokens from both L and G models and concatenate.

        Args:
            token_weight_pairs: Dict with 'g' and 'l' token weight pairs

        Returns:
            Tuple of (concatenated embeddings, pooled output from G)
        """
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        # Cut to minimum length and concatenate
        cut_to = min(l_out.shape[1], g_out.shape[1])
        return torch.cat([l_out[:, :cut_to], g_out[:, :cut_to]], dim=-1), g_pooled

    def load_sd(self, sd):
        """Load state dict - routes to G or L model based on keys present.

        Args:
            sd: State dict to load

        Returns:
            Tuple of (missing keys, unexpected keys)
        """
        # Filter out problematic keys that have shape mismatches
        # logit_scale can be a scalar in checkpoint but 1D in model
        filtered_sd = {}
        for k, v in sd.items():
            # Skip logit_scale if it has the wrong shape
            if "logit_scale" in k:
                # Check expected shape from the model
                if k in self.state_dict():
                    expected_shape = self.state_dict()[k].shape
                    if v.shape != expected_shape:
                        # Try to reshape or skip
                        if v.numel() == 1 and len(expected_shape) == 1:
                            # Reshape scalar to 1D
                            filtered_sd[k] = v.reshape(expected_shape)
                            continue
                        else:
                            # Skip if can't resolve
                            continue
            filtered_sd[k] = v
        
        # Check if this is a G model state dict (has layer 30)
        if "text_model.encoder.layers.30.mlp.fc1.weight" in filtered_sd:
            return self.clip_g.load_sd(filtered_sd)
        else:
            return self.clip_l.load_sd(filtered_sd)


class SDXLRefinerClipModel(SDClip.SD1ClipModel):
    """SDXL Refiner CLIP model (G only)."""

    def __init__(self, device="cpu", dtype=None, model_options=None):
        """Initialize SDXL Refiner CLIP model.

        Args:
            device: Device to load model on
            dtype: Data type for model weights
            model_options: Additional model options (optional)
        """
        if model_options is None:
            model_options = {}
        super().__init__(
            device=device, dtype=dtype, clip_name="g", clip_model=SDXLClipG, model_options=model_options
        )

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle common SDXL mismatches."""
        filtered_sd = {}
        for k, v in state_dict.items():
            # Handle logit_scale shape mismatch
            if "logit_scale" in k and k in self.state_dict():
                expected_shape = self.state_dict()[k].shape
                if v.shape != expected_shape and v.numel() == 1 and len(expected_shape) == 1:
                    filtered_sd[k] = v.reshape(expected_shape)
                    continue
            # Skip position_ids if they cause mismatches (Refiner often doesn't need them if embeddings.weight is present)
            if "position_ids" in k:
                continue
            filtered_sd[k] = v
        
        return super().load_state_dict(filtered_sd, strict=strict)

    def load_sd(self, sd):
        """Load state dict and route to G model."""
        return self.clip_g.load_sd(sd)
