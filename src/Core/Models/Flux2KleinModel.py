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
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from src.Core.AbstractModel import AbstractModel, ModelCapabilities
from src.Utilities import util
from src.Device import Device

# Import modules that were previously lazy-loaded inside methods
# This avoids KeyError: 'src' when running via uv run streamlit
from src.NeuralNetwork.flux2.model import Flux2, Flux2Params
from src.Model.ModelPatcher import ModelPatcher
from src.clip.KleinEncoder import KleinCLIP, Qwen3_4BModel
from src.AutoEncoders import VariationalAE
from src.sample import sampling
from src.Utilities import Latent
from src.Model import LoRas

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
        
        # Device management
        self.load_device = Device.get_torch_device()
        self.offload_device = torch.device("cpu")
    
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
            is_flux=True,
            is_flux2=True,
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
        
        # Guard: Don't reload if already loaded with same diffusion model
        if self._loaded and self.model_path == diffusion_path:
            logger.info("Flux2KleinModel: Already loaded, skipping redundant load")
            return self
        
        if diffusion_path is None:
            raise ValueError(
                "No Flux2 diffusion model found. Please place the model in "
                f"{DEFAULT_DIFFUSION_MODEL_DIR}/ with 'flux' or 'klein' in the filename."
            )
        
        self.model_path = diffusion_path
        
        # Resolve other paths only when loading is actually needed
        text_encoder_path = self._text_encoder_path or self._find_text_encoder()
        vae_path = self._vae_path or self._find_vae()
        
        logger.info(f"Flux2KleinModel: Loading components...")
        logger.info(f"  Diffusion model: {diffusion_path}")
        logger.info(f"  Text encoder: {text_encoder_path}")
        logger.info(f"  VAE: {vae_path}")
        
        try:
            # Load diffusion model
            # self.model = self._load_diffusion_model(diffusion_path) # Original line
            
            # New FP8 loading logic
            from src.NeuralNetwork.flux2.model import create_flux2_klein
            from src.Device import Device
            from src.FileManaging import Loader
            
            # Check for FP8 support and user preference/environment
            use_fp8 = Device.is_fp8_supported(self.load_device)
            # For 8GB cards, we force FP8 for Flux2 Klein 4B to avoid swapping
            total_vram = Device.get_total_memory(self.load_device) / (1024**3)
            if total_vram < 12.0: # If less than 12GB, FP8 is highly recommended for Flux
                use_fp8 = use_fp8 and True
            
            dtype = torch.bfloat16 # Base weight dtype
            
            # Create model with detected config
            config = self._detect_flux2_config(util.load_torch_file(diffusion_path, device=torch.device("cpu"))) # Load temporarily to detect config
            params = Flux2Params(**config)
            self.model = Flux2(params=params, dtype=dtype, device=torch.device("cpu")) # Create on CPU first
            self.model.eval()
            
            # Attach config for compatibility
            self._model_config = self._create_model_config() # Ensure _model_config is set
            
            # Load weights
            sd = util.load_torch_file(diffusion_path, device=self.offload_device)
            # Sanitize NaN values in weights (some Flux2 checkpoints have NaN biases)
            nan_keys = []
            for key, value in sd.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    nan_keys.append(key)
                    sd[key] = torch.where(torch.isnan(value), torch.zeros_like(value), value)
            if nan_keys:
                logger.warning(f"Sanitized NaN values in {len(nan_keys)} keys: {nan_keys[:5]}...")
            
            self.model.load_state_dict(sd, strict=False)
            del sd
            
            self._raw_model = self.model # Store raw model
            
            # Create ModelPatcher
            self.model = ModelPatcher(self.model, self.load_device, self.offload_device)
            
            # Apply FP8 quantization if supported and needed
            if use_fp8:
                logging.info("Flux2: Applying FP8 weight-only quantization to fit in VRAM")
                self.model.weight_only_quantize(torch.float8_e4m3fn)
                self.model.model_dtype = lambda: torch.float8_e4m3fn # Override
            
            # Load text encoder
            if text_encoder_path:
                self.clip = self._load_klein_text_encoder(text_encoder_path, quantize=use_fp8)
                self._text_encoder = self.clip # For internal reference
                self._tokenizer = self.clip.tokenizer
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
            
            # Attach model_sampling for sampler infrastructure
            from src.sample import sampling
            self.model.model_sampling = sampling.model_sampling(self._model_config, "flux2", flux=True, flux2=True)
            
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
        
        # Attach config for compatibility
        model.model_config = self._create_model_config()
        
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
                "num_heads": 24,  # Flux2: hidden_size/sum(axes_dim) = 3072/128 = 24
                "depth": 19,
                "depth_single_blocks": 38,
                "axes_dim": [32, 32, 32, 32],  # Flux2 specific - sum=128
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
                "context_in_dim": 7680,
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
            config["use_vector_in"] = True  # Enable vector_in if weights exist
            logger.info(f"Detected vector_in with dim {config['vec_in_dim']}")
        
        # Detect guidance embedding
        if any("guidance_in" in k for k in sd.keys()):
            config["guidance_embed"] = True
            
        # Detect txt_norm (critical for some Flux2 variants)
        if any("txt_norm.scale" in k for k in sd.keys()):
            config["txt_norm"] = True
            logger.info("Detected txt_norm in model weights")
        
        logger.info(f"Detected Flux2 config: depth={config['depth']}, "
                   f"single_blocks={config['depth_single_blocks']}, "
                   f"hidden={config['hidden_size']}, mlp_ratio={config['mlp_ratio']}, "
                   f"gated_mlp={config.get('gated_mlp', False)}")
        
        return config

    def _load_klein_text_encoder(self, path: str, quantize: bool = False):
        """Load the Klein (Qwen3-4B) text encoder.
        
        Args:
            path: Path to text encoder safetensors
            quantize: Whether to quantize weights to FP8
            
        Returns:
            KleinCLIP wrapper
        """
        logger.info(f"Loading Text Encoder: {path}")
        from src.clip.KleinEncoder import KleinCLIP, KleinTokenizer, Qwen3_4BModel, get_ops
        from src.Model.ModelPatcher import ModelPatcher
        
        # Determine paths
        sd_path = path
        tokenizer_path = os.path.join(os.path.dirname(path), "qwen25_tokenizer")
        if not os.path.exists(tokenizer_path):
             tokenizer_path = None # Let KleinTokenizer find its default
             
        # Load weights
        sd = util.load_torch_file(sd_path, device=torch.device("cpu"))
        
        # Create model structure
        # Base dtype is BF16
        dtype = torch.bfloat16
        model = Qwen3_4BModel(dtype=dtype, device="cpu")
        
        # Load state dict
        model_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                model_sd[k[6:]] = v
            else:
                model_sd[k] = v
        
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        
        # Apply quantization BEFORE moving to offload device if requested
        if quantize:
            logger.info("Flux2KleinModel: Quantizing Klein (Qwen3-4B) to FP8")
            # We must use ModelPatcher to correctly update comfy_cast_weights flags
            te_patcher = ModelPatcher(model, self.load_device, self.offload_device)
            te_patcher.weight_only_quantize(torch.float8_e4m3fn)
            model = te_patcher.model

        # IMPORTANT: Keep model on CPU to save VRAM for diffusion model
        offload_device = Device.text_encoder_offload_device()
        model = model.to(offload_device)
        
        # Create wrapper
        tokenizer = KleinTokenizer(tokenizer_path)
        clip = KleinCLIP(tokenizer=tokenizer, model=model, dtype=dtype, device=self.load_device, offload_device=offload_device)
        
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
        logger.info(f"Loading VAE: {path}")
        
        # Load state dict
        sd = util.load_torch_file(path)
        
        # Check for diffusers format and convert if needed (ComfyUI approach)
        if 'decoder.up_blocks.0.resnets.0.norm1.weight' in sd:
            logger.info("Converting diffusers VAE format to SD format")
            sd = self._convert_diffusers_vae(sd)
        
        # Log VAE structure
        is_flux_vae = False
        if 'decoder.conv_in.weight' in sd:
            z_ch = sd['decoder.conv_in.weight'].shape[1]
            logger.info(f"VAE z_channels: {z_ch}")
        
        if 'post_quant_conv.weight' in sd:
            embed_dim = sd['post_quant_conv.weight'].shape[1]
            logger.info(f"VAE embed_dim: {embed_dim} (Standard VAE)")
            is_flux_vae = False
        else:
            logger.info("VAE missing post_quant_conv (Flux VAE)")
            is_flux_vae = True
        
        # Create VAE using native implementation
        # Set flux=True if it's a Flux VAE (skips post_quant_conv)
        # Use bfloat16 for better precision/memory balance on modern GPUs
        vae = VariationalAE.VAE(sd=sd, flux=is_flux_vae, dtype=torch.bfloat16)
        
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
            latent_format = Latent.Flux2()
            recommended_steps = 4
            recommended_cfg = 1.0
        
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
        
        CRITICAL: ComfyUI LEFT-PADS text embeddings to 512 tokens before passing
        to the diffusion model. This is essential for matching image quality because:
        1. The positional encoding (RoPE) depends on sequence length
        2. The model was trained with fixed 512-token text sequences
        
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
                # Encode each prompt in the batch
                all_hidden = []
                all_pooled = []
                for p in prompt:
                    tokens = self.clip.tokenizer.tokenize_with_weights(p)
                    h, pol, _ = self.clip.encode_token_weights(tokens)
                    all_hidden.append(h)
                    # Handle cases where pooled output might be None (common in Klein/Qwen encoders)
                    if pol is not None:
                        all_pooled.append(pol)
                
                hidden_states = torch.cat(all_hidden, dim=0)
                pooled = torch.cat(all_pooled, dim=0) if all_pooled else None
            else:
                # Single prompt
                tokens = self.clip.tokenizer.tokenize_with_weights(prompt)
                hidden_states, pooled, extra = self.clip.encode_token_weights(tokens)
            
            # Encode negative (or empty)
            neg_prompt = negative_prompt
            if neg_prompt:
                if isinstance(neg_prompt, list):
                    # We usually only need one negative for the whole batch or match batch size
                    if len(neg_prompt) == 1:
                        neg_prompt = neg_prompt[0]
                    else:
                        # Encode all negatives
                        all_neg_hidden = []
                        all_neg_pooled = []
                        for np in neg_prompt:
                            ntokens = self.clip.tokenizer.tokenize_with_weights(np)
                            nh, npol, _ = self.clip.encode_token_weights(ntokens)
                            all_neg_hidden.append(nh)
                            if npol is not None:
                                all_neg_pooled.append(npol)
                        neg_hidden = torch.cat(all_neg_hidden, dim=0)
                        neg_pooled = torch.cat(all_neg_pooled, dim=0) if all_neg_pooled else None
                        neg_prompt = None # Mark as processed
            
            if neg_prompt is not None:
                neg_tokens = self.clip.tokenizer.tokenize_with_weights(neg_prompt or "")
                neg_hidden, neg_pooled, neg_extra = self.clip.encode_token_weights(neg_tokens)
            
            # Embeddings are already padded to 512 tokens by the tokenizer
            # Format as conditioning
            # Note: ComfyUI does NOT pass attention_mask to diffusion model for Flux2
            # The zero-padded tokens don't contribute meaningfully to cross-attention
            cond_dict = {"pooled_output": pooled}
            positive = [[hidden_states, cond_dict]]
            
            neg_cond_dict = {"pooled_output": neg_pooled}
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
        latent_image: Optional[Any] = None,
        start_step: Optional[int] = None,
        last_step: Optional[int] = None,
        disable_noise: bool = False,
        callback: Optional[Callable] = None,
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
        
        # Log recommendation if CFG is high for this distilled model
        if ctx.sampling.cfg > 2.0:
            logger.info(f"Tip: Flux2 Klein works best with CFG 1.0. "
                       f"You are currently using CFG {ctx.sampling.cfg}.")
        
        try:
            # Use provided latent or create empty one for Flux2
            if latent_image is not None:
                latent = latent_image
            else:
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
            
            # Run sampling with flux=True AND flux2=True for resolution-aware scheduler
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
                start_step=start_step,
                last_step=last_step,
                disable_noise=disable_noise,
                callback=callback or ctx.callback,
                flux=True,  # Enable Flux sampling mode
                flux2=True,  # Enable Flux2-specific resolution-aware scheduler (matches ComfyUI Flux2Scheduler)
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
        
        Flux2 uses 32-channel VAE-shaped latents in the pipeline.
        
        Args:
            width: Image width
            height: Image height
            batch_size: Batch size
            
        Returns:
            Dict with 'samples' key containing latent tensor
        """
        # Flux VAE uses 8x downscaling 
        latent_height = height // 8
        latent_width = width // 8
        
        latent = torch.zeros(
            batch_size,
            32,
            latent_height,
            latent_width,
            dtype=torch.float32,
        )
        
        return {"samples": latent}
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space using the VAE.
        
        Args:
            latents: Latent tensor or dict with 'samples' key
            
        Returns:
            Decoded image tensor in [0, 1] range
        """
        if not self._loaded:
            raise RuntimeError("Model must be loaded before decoding")
        
        try:
            # Handle both raw tensor and dict input
            if isinstance(latents, dict):
                samples_tensor = latents["samples"]
            else:
                samples_tensor = latents
            
            # Use the Flux2 latent format
            # Apply process_latent_out (undo scale/shift from sampling) is now handled by KSAMPLER
            
            # Decode with VAE
            decoder = VariationalAE.VAEDecode()
            result = decoder.decode(
                vae=self.vae,
                samples={"samples": samples_tensor},
            )
            
            return result[0]
            
        except Exception as e:
            logger.exception(f"Decoding failed: {e}")
            raise
    
    def get_model_object(self, name):
        """Get an attribute from the model or its patcher."""
        if name == "latent_format":
            return self._model_config.latent_format
        if self.model:
            return self.model.get_model_object(name)
        return None

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



