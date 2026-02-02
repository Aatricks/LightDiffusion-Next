"""Flux2 Klein model adapter for LightDiffusion-Next.

Provides a clean interface to the Flux2 Klein 4B model that inherits from
AbstractModel and integrates with the LightDiffusion-Next model factory.

This implementation uses ONLY native LightDiffusion-Next components,
without any ComfyUI imports.

File structure expected:
    - include/diffusion_model/flux-2-klein-4b.safetensors  (or similar)
    - include/text_encoder/qwen_3_4b.safetensors
    - include/vae/ae.safetensors (Flux VAE)
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import torch

from src.Core.AbstractModel import AbstractModel, ModelCapabilities
from src.Utilities import util
from src.Device import Device

if TYPE_CHECKING:
    from src.Core.Context import Context


logger = logging.getLogger(__name__)

# Default paths for Flux2 Klein components
DEFAULT_DIFFUSION_MODEL_DIR = "./include/diffusion_model"
DEFAULT_TEXT_ENCODER_DIR = "./include/text_encoder"  
DEFAULT_VAE_DIR = "./include/vae"


class Flux2KleinModel(AbstractModel):
    """Flux2 Klein 4B model implementation.
    
    Wraps the Flux2 Klein model with the clean AbstractModel interface
    for use with the LightDiffusion-Next pipeline system.
    
    The Flux2 Klein model is a distilled version of the Flux2 architecture
    using the Klein (Qwen3 4B) text encoder.
    
    Unlike SD1.5/SDXL which use combined checkpoints, Flux2 Klein loads
    components separately:
    - Diffusion model from include/diffusion_model/
    - Text encoder (Qwen3 4B) from include/text_encoder/
    - VAE from include/vae/
    """
    
    def __init__(
        self, 
        model_path: str = None,
        text_encoder_path: str = None,
        vae_path: str = None,
    ):
        """Initialize the Flux2 Klein model adapter.
        
        Args:
            model_path: Path to diffusion model (safetensors)
            text_encoder_path: Path to Qwen3 text encoder (optional, auto-detected)
            vae_path: Path to VAE (optional, auto-detected)
        """
        super().__init__(model_path)
        self._text_encoder = None
        self._tokenizer = None
        self._model_config = None
        self._text_encoder_path = text_encoder_path
        self._vae_path = vae_path
        self._raw_model = None  # The raw Flux2 nn.Module
    
    def _create_capabilities(self) -> ModelCapabilities:
        """Create capabilities for Flux2 Klein model."""
        return ModelCapabilities(
            min_resolution=256,
            max_resolution=2048,
            preferred_resolution=1024,
            requires_resolution_multiple=16,  # Flux2 uses 16-pixel patches
            supports_hires_fix=True,
            supports_img2img=True,
            supports_inpainting=False,  # Not yet implemented for Flux2
            supports_controlnet=False,  # ControlNet support pending
            supports_stable_fast=False,  # May need special handling
            supports_deepcache=False,  # Architecture differs from UNet
            supports_tome=False,  # Token merging needs special implementation
            supports_lora=False,  # Flux2 LoRA format differs from SD
            uses_dual_clip=False,  # Uses single Klein (Qwen3) encoder
            requires_size_conditioning=False,
        )

    def _find_diffusion_model(self) -> Optional[str]:
        """Auto-detect Flux2 diffusion model in default directory."""
        if os.path.exists(DEFAULT_DIFFUSION_MODEL_DIR):
            for f in os.listdir(DEFAULT_DIFFUSION_MODEL_DIR):
                f_lower = f.lower()
                if ("flux" in f_lower or "klein" in f_lower) and f.endswith((".safetensors", ".pt", ".pth")):
                    return os.path.join(DEFAULT_DIFFUSION_MODEL_DIR, f)
        return None

    def _find_text_encoder(self) -> Optional[str]:
        """Auto-detect Qwen3 text encoder in default directory."""
        if os.path.exists(DEFAULT_TEXT_ENCODER_DIR):
            for f in os.listdir(DEFAULT_TEXT_ENCODER_DIR):
                f_lower = f.lower()
                if ("qwen" in f_lower or "klein" in f_lower) and f.endswith((".safetensors", ".pt", ".pth")):
                    return os.path.join(DEFAULT_TEXT_ENCODER_DIR, f)
        return None

    def _find_vae(self) -> Optional[str]:
        """Auto-detect VAE in default directory."""
        if os.path.exists(DEFAULT_VAE_DIR):
            # Look for Flux-compatible VAE (ae.safetensors)
            for f in os.listdir(DEFAULT_VAE_DIR):
                if f.endswith((".safetensors", ".pt", ".pth")):
                    return os.path.join(DEFAULT_VAE_DIR, f)
        return None
    
    def load(self, model_path: str = None) -> "Flux2KleinModel":
        """Load the Flux2 Klein model components from disk.
        
        Components are loaded separately:
        - Diffusion model (Flux2 transformer)
        - Text encoder (Qwen3 4B via Klein tokenizer)
        - VAE
        
        Args:
            model_path: Optional override for the diffusion model path
            
        Returns:
            Self for method chaining
        """
        # Resolve paths
        diffusion_path = model_path or self.model_path or self._find_diffusion_model()
        text_encoder_path = self._text_encoder_path or self._find_text_encoder()
        vae_path = self._vae_path or self._find_vae()
        
        if diffusion_path is None:
            raise ValueError(
                "No Flux2 diffusion model found. Please place the model in "
                f"{DEFAULT_DIFFUSION_MODEL_DIR}/ with 'flux' or 'klein' in the filename."
            )
        
        self.model_path = diffusion_path
        
        logger.info(f"Flux2KleinModel: Loading components...")
        logger.info(f"  Diffusion model: {diffusion_path}")
        logger.info(f"  Text encoder: {text_encoder_path}")
        logger.info(f"  VAE: {vae_path}")
        
        try:
            # Load diffusion model
            self.model = self._load_diffusion_model(diffusion_path)
            
            # Load text encoder (Qwen3 via Klein)
            if text_encoder_path:
                self.clip = self._load_klein_text_encoder(text_encoder_path)
            else:
                logger.warning("No Qwen3 text encoder found - prompt encoding may fail")
                self.clip = None
            
            # Load VAE
            if vae_path:
                self.vae = self._load_vae(vae_path)
            else:
                logger.warning("No VAE found - image decoding may fail")
                self.vae = None
            
            # Store config for sampling
            self._model_config = self._create_model_config()
            
            self._loaded = True
            logger.info(f"Flux2KleinModel: Successfully loaded all components")
            
        except Exception as e:
            logger.exception(f"Flux2KleinModel: Failed to load: {e}")
            raise
        
        return self

    def _load_diffusion_model(self, path: str):
        """Load the Flux2 diffusion model using native LightDiffusion-Next.
        
        Args:
            path: Path to diffusion model safetensors
            
        Returns:
            ModelPatcher wrapping the Flux2 model
        """
        from src.NeuralNetwork.flux2.model import Flux2, Flux2Params
        from src.Model import ModelPatcher
        
        logger.info(f"Loading Flux2 diffusion model: {path}")
        
        # Load state dict using native utility
        sd = util.load_torch_file(path)
        
        # Sanitize NaN values in weights (some Flux2 checkpoints have NaN biases)
        nan_keys = []
        for key, value in sd.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                nan_keys.append(key)
                sd[key] = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        if nan_keys:
            logger.warning(f"Sanitized NaN values in {len(nan_keys)} keys: {nan_keys[:5]}...")
        
        # Detect model configuration from state dict
        config = self._detect_flux2_config(sd)
        
        # Determine dtype and device
        load_device = Device.get_torch_device()
        offload_device = Device.unet_offload_device()
        
        # Infer dtype from weights
        dtype = torch.bfloat16
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                dtype = v.dtype
                break
        
        logger.info(f"Flux2 model dtype: {dtype}")
        
        # Create model with detected config
        params = Flux2Params(**config)
        model = Flux2(params=params, dtype=dtype, device="cpu")
        
        # Load weights
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.debug(f"Missing keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Unexpected keys: {len(unexpected)}")
        
        self._raw_model = model
        
        # Wrap in ModelPatcher for compatibility with sampling infrastructure
        model_patcher = ModelPatcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=offload_device,
            current_device=torch.device("cpu"),
        )
        
        return model_patcher

    def _detect_flux2_config(self, sd: dict) -> dict:
        """Detect Flux2 model configuration from state dict.
        
        Args:
            sd: Model state dictionary
            
        Returns:
            Configuration dict for Flux2Params
        """
        # Detect if this is Flux2 (has double_stream_modulation) or Flux1
        is_flux2 = any("double_stream_modulation" in k for k in sd.keys())
        
        if is_flux2:
            # Flux2 / Klein defaults (patch_size=1 unlike Flux1!)
            config = {
                "patch_size": 1,  # CRITICAL: Flux2 uses patch_size=1 (no spatial patchification)
                "in_channels": 128,  # Direct channel input (no patch_size division)
                "out_channels": 128,  # Direct channel output
                "vec_in_dim": 768,
                "context_in_dim": 7680,  # Klein uses concatenated multi-layer output
                "hidden_size": 3072,
                "mlp_ratio": 3.0,  # Klein uses 3.0 with gated MLP
                "num_heads": 48,  # Flux2 uses 48 heads (hidden/axes_sum = 3072/64)
                "depth": 19,
                "depth_single_blocks": 38,
                "axes_dim": [32, 32, 32, 32],  # Flux2 specific
                "theta": 2000,  # Flux2 uses lower theta
                "qkv_bias": False,
                "guidance_embed": False,
                "gated_mlp": True,  # Klein uses gated MLP (SwiGLU)
                "global_modulation": True,  # Flux2 feature
                "mlp_silu_act": True,  # Flux2 feature
                "ops_bias": False,  # Flux2 feature
                "use_vector_in": False,  # Flux2/Klein doesn't use pooled conditioning
            }
            logger.info("Detected Flux2 model (has double_stream_modulation)")
        else:
            # Flux1 defaults
            config = {
                "in_channels": 16,
                "out_channels": 16,
                "vec_in_dim": 768,
                "context_in_dim": 4096,
                "hidden_size": 3072,
                "mlp_ratio": 4.0,
                "num_heads": 24,
                "depth": 19,
                "depth_single_blocks": 38,
                "axes_dim": [16, 56, 56],  # Flux1 specific
                "theta": 10000,
                "qkv_bias": True,
                "guidance_embed": True,
                "gated_mlp": False,
            }
            logger.info("Detected Flux1 model")
        
        # Detect depth from double_blocks
        double_blocks = [k for k in sd.keys() if "double_blocks" in k]
        if double_blocks:
            max_block = max(
                int(k.split("double_blocks.")[1].split(".")[0])
                for k in double_blocks
                if "double_blocks." in k
            )
            config["depth"] = max_block + 1
        
        # Detect single blocks depth
        single_blocks = [k for k in sd.keys() if "single_blocks" in k]
        if single_blocks:
            max_single = max(
                int(k.split("single_blocks.")[1].split(".")[0])
                for k in single_blocks
                if "single_blocks." in k
            )
            config["depth_single_blocks"] = max_single + 1
        
        # Detect hidden size and in_channels from img_in
        if "img_in.weight" in sd:
            config["hidden_size"] = sd["img_in.weight"].shape[0]
            # img_in input dim = in_channels * patch_size^2
            # For Flux2 with patch_size=1: in_channels = img_in_dim directly
            img_in_dim = sd["img_in.weight"].shape[1]
            patch_size = config.get("patch_size", 2)
            config["in_channels"] = img_in_dim // (patch_size ** 2)
            logger.info(f"Detected in_channels={config['in_channels']} from img_in (patch_size={patch_size})")
        
        # Detect out_channels from final_layer
        if "final_layer.linear.weight" in sd:
            # final_layer.linear maps hidden -> patch_size * patch_size * out_channels
            # For Flux2 with patch_size=1: out_channels = final.shape[0] directly
            final_out = sd["final_layer.linear.weight"].shape[0]
            patch_size = config.get("patch_size", 2)
            config["out_channels"] = final_out // (patch_size ** 2)
            logger.info(f"Detected out_channels={config['out_channels']} from final_layer")
        
        # Detect mlp_ratio and gated_mlp from double_blocks MLP weights
        # For gated MLP: img_mlp.0 maps hidden -> 2*intermediate (gate+up)
        #                img_mlp.2 maps intermediate -> hidden
        # So: mlp_0_out = 2 * intermediate, intermediate = mlp_2_in
        # mlp_ratio = intermediate / hidden
        if "double_blocks.0.img_mlp.0.weight" in sd and "double_blocks.0.img_mlp.2.weight" in sd:
            mlp_0_out = sd["double_blocks.0.img_mlp.0.weight"].shape[0]
            mlp_2_in = sd["double_blocks.0.img_mlp.2.weight"].shape[1]
            hidden = config["hidden_size"]
            
            # Check if it's gated MLP: mlp_0_out should be 2 * mlp_2_in
            if abs(mlp_0_out - 2 * mlp_2_in) < 10:  # Small tolerance
                # Gated MLP detected
                config["gated_mlp"] = True
                intermediate = mlp_2_in
                config["mlp_ratio"] = intermediate / hidden
                logger.info(f"Detected gated MLP: intermediate={intermediate}, mlp_ratio={config['mlp_ratio']}")
            else:
                # Standard MLP: mlp_0_out = mlp_2_in = hidden * mlp_ratio
                config["gated_mlp"] = False
                config["mlp_ratio"] = mlp_0_out / hidden
        
        # Calculate num_heads from hidden_size and axes_dim (ComfyUI approach)
        # num_heads = hidden_size // sum(axes_dim)
        axes_sum = sum(config["axes_dim"])
        config["num_heads"] = config["hidden_size"] // axes_sum
        logger.info(f"Calculated num_heads={config['num_heads']} from hidden_size={config['hidden_size']} / axes_sum={axes_sum}")
        
        # Detect context_in_dim from txt_in
        if "txt_in.weight" in sd:
            config["context_in_dim"] = sd["txt_in.weight"].shape[1]
        
        # Detect vec_in_dim from vector_in
        if "vector_in.in_layer.weight" in sd:
            config["vec_in_dim"] = sd["vector_in.in_layer.weight"].shape[1]
        
        # Detect guidance embedding
        if any("guidance_in" in k for k in sd.keys()):
            config["guidance_embed"] = True
        
        logger.info(f"Detected Flux2 config: depth={config['depth']}, "
                   f"single_blocks={config['depth_single_blocks']}, "
                   f"hidden={config['hidden_size']}, mlp_ratio={config['mlp_ratio']}, "
                   f"gated_mlp={config.get('gated_mlp', False)}")
        
        return config

    def _load_klein_text_encoder(self, path: str):
        """Load the Klein (Qwen3 4B) text encoder.
        
        Args:
            path: Path to Qwen3 text encoder safetensors
            
        Returns:
            CLIP-compatible text encoder wrapper
        """
        from src.clip.KleinEncoder import KleinCLIP, Qwen3_4BModel
        
        logger.info(f"Loading Klein text encoder (Qwen3): {path}")
        
        # Load state dict
        sd = util.load_torch_file(path)
        
        # Determine dtype
        dtype = torch.float16
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.bfloat16):
                dtype = v.dtype
                break
        
        # Create model and load weights
        load_device = Device.get_torch_device()
        model = Qwen3_4BModel(dtype=dtype, device="cpu")
        
        # Load state dict (handle potential key prefixes)
        model_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                model_sd[k[6:]] = v
            else:
                model_sd[k] = v
        
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing:
            logger.debug(f"Klein encoder missing keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Klein encoder unexpected keys: {len(unexpected)}")
        
        # IMPORTANT: Keep model on CPU to save VRAM for diffusion model
        # KleinCLIP will move it to GPU only during encoding
        # This follows ComfyUI's approach of lazy model loading
        offload_device = Device.text_encoder_offload_device()  # CPU
        model = model.to(offload_device).to(dtype)
        
        # Create CLIP wrapper - pass load_device so it knows where to move when encoding
        clip = KleinCLIP(model=model, dtype=dtype, device=load_device, offload_device=offload_device)
        
        # Ensure embeddings directory exists
        os.makedirs("./include/embeddings", exist_ok=True)
        
        return clip

    def _load_vae(self, path: str):
        """Load the VAE for decoding latents using native LightDiffusion-Next.
        
        Following ComfyUI's VAE loading approach:
        - Detects z_channels from decoder.conv_in.weight.shape[1]
        - Uses post_quant_conv/quant_conv (flux=False) for standard VAE structure
        
        Args:
            path: Path to VAE safetensors
            
        Returns:
            VAE model
        """
        from src.AutoEncoders import VariationalAE
        
        logger.info(f"Loading VAE: {path}")
        
        # Load state dict
        sd = util.load_torch_file(path)
        
        # Check for diffusers format and convert if needed (ComfyUI approach)
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd:
            logger.info("Converting diffusers VAE format to SD format")
            sd = self._convert_diffusers_vae(sd)
        
        # Log VAE structure
        if 'decoder.conv_in.weight' in sd:
            z_ch = sd['decoder.conv_in.weight'].shape[1]
            logger.info(f"VAE z_channels: {z_ch}")
        if 'post_quant_conv.weight' in sd:
            embed_dim = sd['post_quant_conv.weight'].shape[1]
            logger.info(f"VAE embed_dim: {embed_dim}")
        
        # Create VAE using native implementation
        # flux=False because Flux2 VAE uses standard post_quant_conv/quant_conv structure
        vae = VariationalAE.VAE(sd=sd, flux=False)
        
        return vae
    
    def _convert_diffusers_vae(self, sd: dict) -> dict:
        """Convert diffusers VAE format to SD format (ComfyUI approach)."""
        # VAE conversion map from ComfyUI's diffusers_convert.py
        vae_conversion_map = [
            ("nin_shortcut", "conv_shortcut"),
            ("norm_out", "conv_norm_out"),
            ("mid.attn_1.", "mid_block.attentions.0."),
        ]
        
        for i in range(4):
            for j in range(2):
                hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
                sd_down_prefix = f"encoder.down.{i}.block.{j}."
                vae_conversion_map.append((sd_down_prefix, hf_down_prefix))
            
            if i < 3:
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
                sd_downsample_prefix = f"down.{i}.downsample."
                vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))
                
                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"up.{3 - i}.upsample."
                vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))
            
            for j in range(3):
                hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
                sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
                vae_conversion_map.append((sd_up_prefix, hf_up_prefix))
        
        for i in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{i}."
            sd_mid_res_prefix = f"mid.block_{i + 1}."
            vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))
        
        vae_conversion_map_attn = [
            ("norm.", "group_norm."),
            ("q.", "query."), ("k.", "key."), ("v.", "value."),
            ("q.", "to_q."), ("k.", "to_k."), ("v.", "to_v."),
            ("proj_out.", "to_out.0."), ("proj_out.", "proj_attn."),
        ]
        
        mapping = {k: k for k in sd.keys()}
        for k, v in mapping.items():
            for sd_part, hf_part in vae_conversion_map:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
        
        for k, v in mapping.items():
            if "attentions" in k:
                for sd_part, hf_part in vae_conversion_map_attn:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
        
        new_state_dict = {v: sd[k] for k, v in mapping.items()}
        
        # Reshape attention weights
        weights_to_convert = ["q", "k", "v", "proj_out"]
        for k, v in new_state_dict.items():
            for weight_name in weights_to_convert:
                if f"mid.attn_1.{weight_name}.weight" in k:
                    new_state_dict[k] = v.reshape(*v.shape, 1, 1)
        
        return new_state_dict
    
    def _create_model_config(self):
        """Create a model config object for sampling."""
        class Flux2KleinConfig:
            """Configuration for Flux2 Klein sampling."""
            sampling_settings = {
                "shift": 2.02,  # Flux2 default shift (different from Flux1's 1.15)
            }
            latent_format = Flux2LatentFormat()
        
        return Flux2KleinConfig()
    
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] = "",
        clip_skip: int = None,
    ) -> tuple[Any, Any]:
        """Encode text prompts into conditioning tensors.
        
        For Flux2 Klein, this uses the Qwen3-based Klein text encoder
        which does not use traditional CLIP skip.
        
        Args:
            prompt: Positive prompt(s) to encode
            negative_prompt: Negative prompt(s) (may be ignored for Flux2)
            clip_skip: Not used for Klein encoder
            
        Returns:
            Tuple of (positive_conditioning, negative_conditioning)
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before encoding prompts")
        
        if self.clip is None:
            raise RuntimeError("No text encoder loaded")
        
        try:
            import torch
            
            # Use Klein encoder directly
            if isinstance(prompt, list):
                prompt = prompt[0]  # Handle batch
            
            # Tokenize and encode positive
            tokens = self.clip.tokenizer.tokenize_with_weights(prompt)
            hidden_states, pooled, extra = self.clip.encode_token_weights(tokens)
            pos_mask = extra.get("attention_mask")
            
            # Encode negative (or empty)
            neg_prompt = negative_prompt
            if neg_prompt:
                if isinstance(neg_prompt, list):
                    neg_prompt = neg_prompt[0]
            else:
                neg_prompt = ""  # Empty string for negative
            
            neg_tokens = self.clip.tokenizer.tokenize_with_weights(neg_prompt)
            neg_hidden, neg_pooled, neg_extra = self.clip.encode_token_weights(neg_tokens)
            neg_mask = neg_extra.get("attention_mask")
            
            # Pad to same sequence length (required for batching in sampling)
            pos_len = hidden_states.shape[1]
            neg_len = neg_hidden.shape[1]
            max_len = max(pos_len, neg_len)
            
            if pos_len < max_len:
                # Pad positive (right padding with zeros)
                pad_size = max_len - pos_len
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_size), value=0)
                if pos_mask is not None:
                    pos_mask = torch.nn.functional.pad(pos_mask, (0, pad_size), value=0)
            
            if neg_len < max_len:
                # Pad negative (right padding with zeros)
                pad_size = max_len - neg_len
                neg_hidden = torch.nn.functional.pad(neg_hidden, (0, 0, 0, pad_size), value=0)
                if neg_mask is not None:
                    neg_mask = torch.nn.functional.pad(neg_mask, (0, pad_size), value=0)
            
            # Format as conditioning - include attention_mask for the diffusion model
            cond_dict = {"pooled_output": pooled}
            if pos_mask is not None:
                cond_dict["attention_mask"] = pos_mask
            positive = [[hidden_states, cond_dict]]
            
            neg_cond_dict = {"pooled_output": neg_pooled}
            if neg_mask is not None:
                neg_cond_dict["attention_mask"] = neg_mask
            negative = [[neg_hidden, neg_cond_dict]]
            
            return positive, negative
            
        except Exception as e:
            logger.exception(f"Prompt encoding failed: {e}")
            raise
    
    def generate(
        self,
        ctx: "Context",
        positive: Any,
        negative: Any,
    ) -> dict:
        """Generate latents using the Flux2 sampler.
        
        Args:
            ctx: Context with generation parameters
            positive: Positive conditioning
            negative: Negative conditioning (may be ignored)
            
        Returns:
            Dictionary with 'samples' key containing generated latents
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before generating")
        
        try:
            from src.sample import sampling
            
            # Create empty latent for Flux2
            latent = self._create_flux2_latent(
                ctx.width,
                ctx.height,
                ctx.generation.batch,
            )
            
            # Add seeds for deterministic noise
            latent["seeds"] = ctx.seeds[:ctx.generation.batch] if ctx.seeds else [ctx.seed]
            
            # CRITICAL: Force-disable multi-scale for Flux2 models
            # Multi-scale is designed for UNet architectures (SD1.5/SDXL) and
            # causes significant performance overhead for Flux2's DiT architecture
            enable_multiscale = False  # Always disable for Flux2
            if ctx.sampling.enable_multiscale:
                logger.info("Multi-scale disabled: not compatible with Flux2 architecture")
            
            # Run sampling with flux=True
            ksampler = sampling.KSampler()
            result = ksampler.sample(
                seed=ctx.seed,
                steps=ctx.sampling.steps,
                cfg=ctx.sampling.cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=ctx.sampling.scheduler,
                denoise=ctx.sampling.denoise,
                pipeline=True,
                model=self.model,
                positive=positive,
                negative=negative,
                latent_image=latent,
                flux=True,  # Enable Flux sampling mode
                enable_multiscale=enable_multiscale,  # Force disabled for Flux2
                multiscale_factor=ctx.sampling.multiscale_factor,
                multiscale_fullres_start=ctx.sampling.multiscale_fullres_start,
                multiscale_fullres_end=ctx.sampling.multiscale_fullres_end,
                multiscale_intermittent_fullres=ctx.sampling.multiscale_intermittent_fullres,
                cfg_free_enabled=ctx.sampling.cfg_free_enabled,
                cfg_free_start_percent=ctx.sampling.cfg_free_start_percent,
                batched_cfg=ctx.sampling.batched_cfg,
                dynamic_cfg_rescaling=ctx.sampling.dynamic_cfg_rescaling,
                dynamic_cfg_method=ctx.sampling.dynamic_cfg_method,
                dynamic_cfg_percentile=ctx.sampling.dynamic_cfg_percentile,
                dynamic_cfg_target_scale=ctx.sampling.dynamic_cfg_target_scale,
                adaptive_noise_enabled=ctx.sampling.adaptive_noise_enabled,
                adaptive_noise_method=ctx.sampling.adaptive_noise_method,
            )
            
            return result[0]
            
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            raise
    
    def _create_flux2_latent(self, width: int, height: int, batch_size: int) -> dict:
        """Create an empty latent tensor for Flux2.
        
        Flux2 uses variable latent channels based on model config.
        
        Args:
            width: Image width
            height: Image height
            batch_size: Batch size
            
        Returns:
            Dict with 'samples' key containing latent tensor
        """
        # Use detected in_channels or default
        in_channels = 64
        if self._raw_model is not None and hasattr(self._raw_model, 'in_channels'):
            in_channels = self._raw_model.in_channels
        
        # Flux2 uses 16x downscaling 
        latent_height = height // 16
        latent_width = width // 16
        
        latent = torch.zeros(
            batch_size,
            in_channels,
            latent_height,
            latent_width,
            dtype=torch.float32,
        )
        
        return {"samples": latent}
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space using the VAE.
        
        Following ComfyUI's approach:
        - Unpatchify 128-channel patchified latent to 32-channel VAE format
        - Use process_latent_out to undo scale/shift from sampling
        - Decode with VAE (flux=False for standard post_quant_conv path)
        
        Args:
            latents: Latent tensor or dict with 'samples' key
            
        Returns:
            Decoded image tensor in [0, 1] range
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before decoding")
        
        try:
            from src.AutoEncoders import VariationalAE
            from src.Utilities import Latent
            
            # Handle both raw tensor and dict input
            if isinstance(latents, dict):
                samples_tensor = latents["samples"]
            else:
                samples_tensor = latents
            
            # Flux2 latent is patchified: [B, 128, H/16, W/16]
            # VAE expects: [B, 32, H/8, W/8]
            # Use the Flux2 latent format's unpatchify_for_vae method
            flux2_latent_format = Latent.Flux2()
            samples_tensor = flux2_latent_format.unpatchify_for_vae(samples_tensor)
            logger.info(f"Unpatchified latent shape: {samples_tensor.shape}")
            
            # Apply process_latent_out (undo scale/shift from sampling)
            # For Flux2, this is identity (no scale/shift)
            samples_tensor = flux2_latent_format.process_out(samples_tensor)
            
            # Decode with VAE (flux=False for standard post_quant_conv structure)
            decoder = VariationalAE.VAEDecode()
            result = decoder.decode(
                vae=self.vae,
                samples={"samples": samples_tensor},
                flux=False,  # Use standard post_quant_conv path
            )
            
            return result[0]
            
        except Exception as e:
            logger.exception(f"Decoding failed: {e}")
            raise
    
    def apply_lora(
        self,
        lora_name: str,
        strength_model: float = 1.0,
        strength_clip: float = 1.0,
    ) -> "Flux2KleinModel":
        """Apply a LoRA to the Flux2 Klein model.
        
        Note: LoRA support for Flux2 may be limited.
        
        Args:
            lora_name: Name/path of the LoRA file
            strength_model: Strength to apply to the model
            strength_clip: Strength to apply to CLIP
            
        Returns:
            Self for method chaining
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before applying LoRA")
        
        try:
            from src.Model import LoRas
            loader = LoRas.LoraLoader()
            result = loader.load_lora(
                lora_name=lora_name,
                strength_model=strength_model,
                strength_clip=strength_clip,
                model=self.model,
                clip=self.clip,
            )
            self.model = result[0]
            self.clip = result[1]
            logger.info(f"Applied LoRA: {lora_name}")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA {lora_name}: {e}")
        
        return self


class Flux2LatentFormat:
    """Latent format specification for Flux2 models."""
    
    latent_channels = 64  # Flux2 Klein standard
    latent_rgb_factors = [
        # RGB mapping factors for preview generation (abbreviated)
        [0.0036, -0.0159, 0.0113],
        [0.0115, -0.0065, 0.0018],
        [0.0109, -0.0098, -0.0021],
        [0.0023, -0.0017, -0.0036],
    ]
    
    def __init__(self):
        self.scale_factor = 0.3611  # Flux2 specific scale factor
        self.shift_factor = 0.1159
    
    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Process latent input for the model."""
        return (latent - self.shift_factor) * self.scale_factor
    
    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Process latent output from the model."""
        return latent / self.scale_factor + self.shift_factor


def detect_flux2_klein(state_dict_keys: set) -> bool:
    """Detect if a checkpoint is a Flux2 Klein model.
    
    Args:
        state_dict_keys: Set of keys from the state dict
        
    Returns:
        True if this is a Flux2 Klein checkpoint
    """
    flux2_indicators = [
        "double_stream_modulation",
        "double_blocks.0.img_mod",
        "single_blocks.0.modulation",
        "img_in.weight",
    ]
    
    for indicator in flux2_indicators:
        for key in state_dict_keys:
            if indicator in key:
                return True
    
    return False
