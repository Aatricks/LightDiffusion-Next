"""Flux2 main model implementation for LightDiffusion-Next.

This module contains the main Flux2 model class that orchestrates
the double-stream and single-stream transformer blocks for image generation.

Adapted from ComfyUI's Flux implementation.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.cond import cast as ops_module
from src.NeuralNetwork.flux2.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    LastLayer,
    MLPEmbedder,
    EmbedND,
    Modulation,
)


def get_ops():
    """Get the operations module for weight initialization."""
    return ops_module.disable_weight_init


@dataclass
class Flux2Params:
    """Configuration parameters for Flux2 model.
    
    Attributes:
        in_channels: Input channels (latent space)
        out_channels: Output channels (for prediction)
        vec_in_dim: Dimension of vectorized conditioning input
        context_in_dim: Dimension of text context input
        hidden_size: Transformer hidden dimension
        mlp_ratio: MLP hidden dim multiplier
        num_heads: Number of attention heads  
        depth: Number of transformer layers
        depth_single_blocks: Number of single-stream blocks
        axes_dim: Dimensions for positional encoding axes
        theta: Base frequency for RoPE
        qkv_bias: Whether to use bias in QKV projections
        guidance_embed: Whether to use guidance embedding
        global_modulation: Use global modulation (Flux2/Klein style)
        mlp_silu_act: Use SiLU activation in MLPs
        gated_mlp: Use gated MLP (SwiGLU) structure for Klein models
        ops_bias: Use bias in final projection
        patch_size: Size of image patches (1 for Flux2, 2 for Flux1)
        use_vector_in: Whether to use vector conditioning (pooled text embedding)
        txt_ids_dims: Which axes to give text tokens positional IDs (critical for conditioning)
    """
    in_channels: int = 128  # Flux2 default (128 for patch_size=1)
    out_channels: int = 128  # Flux2 default
    vec_in_dim: int = 768
    context_in_dim: int = 7680
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24  # Flux2 default: hidden_size/sum(axes_dim) = 3072/128 = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)  # Flux2 default - sum=128
    theta: int = 2000  # Flux2 default
    qkv_bias: bool = False  # Flux2 default
    guidance_embed: bool = False
    global_modulation: bool = True  # Flux2 feature
    mlp_silu_act: bool = True  # Flux2 feature
    gated_mlp: bool = True  # Flux2/Klein feature
    ops_bias: bool = False  # Flux2 default
    patch_size: int = 1  # CRITICAL: Flux2 uses patch_size=1
    use_vector_in: bool = False  # Flux2/Klein doesn't use pooled conditioning
    txt_ids_dims: tuple[int, ...] = (3,)  # Flux2/Klein: text gets position IDs in axis 3
    txt_norm: bool = False  # Flux2/Klein may use text normalization


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class Flux2(nn.Module):
    """Flux2 transformer model for image generation.
    
    This model uses a dual-stream architecture where image and text
    are processed through joint attention in double-stream blocks,
    then merged into a single stream for final processing.
    """

    def __init__(self, params: Flux2Params = None, dtype=None, device=None, operations=None):
        super().__init__()
        
        if params is None:
            params = Flux2Params()
        self.params = params
        
        if operations is None:
            operations = get_ops()
        
        # Validation: hidden_size must be divisible by num_heads (ComfyUI check)
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        
        # Validation: pe_dim must equal sum(axes_dim) for RoPE to work correctly
        pe_dim = params.hidden_size // params.num_heads
        axes_sum = sum(params.axes_dim)
        if axes_sum != pe_dim:
            raise ValueError(
                f"sum(axes_dim)={axes_sum} must equal hidden_size/num_heads={pe_dim}. "
                f"For hidden_size={params.hidden_size}, axes_dim={params.axes_dim}, "
                f"num_heads should be {params.hidden_size // axes_sum}"
            )
        
        self.dtype = dtype
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.patch_size = params.patch_size
        
        # Latent format for sampling infrastructure
        from src.Utilities.Latent import Flux2 as Flux2LatentFormat
        self.latent_format = Flux2LatentFormat()
        
        # Model sampling for sigma calculations
        from src.sample.sampling import model_sampling
        self.model_sampling = model_sampling(None, None, flux2=True)
        
        # Memory management
        self.memory_usage_factor = 2.0
        
        # Patch embedding
        # After patchifying, each patch has in_channels * patch_size^2 features
        patch_dim = params.in_channels * (params.patch_size ** 2)
        self.img_in = operations.Linear(
            patch_dim, 
            params.hidden_size, 
            bias=params.ops_bias,  # Flux2 checkpoints often have no bias
            dtype=dtype, 
            device=device
        )
        
        # Conditioning embeddings
        self.txt_in = operations.Linear(
            params.context_in_dim,
            params.hidden_size,
            bias=params.ops_bias,  # Flux2 checkpoints often have no bias
            dtype=dtype,
            device=device
        )

        if params.txt_norm:
            self.txt_norm = RMSNorm(params.context_in_dim, dtype=dtype, device=device)
        else:
            self.txt_norm = None
        
        # Time/vector embedding
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=params.hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
            ops_bias=params.ops_bias,
        )
        
        # Optional vector conditioning (pooled text embedding) - not used in Flux2/Klein
        self.use_vector_in = params.use_vector_in
        if params.use_vector_in:
            self.vector_in = MLPEmbedder(
                in_dim=params.vec_in_dim,
                hidden_dim=params.hidden_size,
                dtype=dtype,
                device=device,
                operations=operations,
                ops_bias=params.ops_bias,
            )
        else:
            self.vector_in = None
        
        # Optional guidance embedding
        self.guidance_embed = params.guidance_embed
        if self.guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=256,
                hidden_dim=params.hidden_size,
                dtype=dtype,
                device=device,
                operations=operations,
                ops_bias=params.ops_bias,
            )
        
        # Global modulation for Flux2 (Klein) - shared across all blocks
        # These are at model level, not per-block, to match checkpoint naming
        if params.global_modulation:
            self.double_stream_modulation_img = Modulation(
                params.hidden_size, double=True, dtype=dtype, device=device, 
                operations=operations, ops_bias=params.ops_bias
            )
            self.double_stream_modulation_txt = Modulation(
                params.hidden_size, double=True, dtype=dtype, device=device,
                operations=operations, ops_bias=params.ops_bias
            )
            self.single_stream_modulation = Modulation(
                params.hidden_size, double=False, dtype=dtype, device=device,
                operations=operations, ops_bias=params.ops_bias
            )
        else:
            self.double_stream_modulation_img = None
            self.double_stream_modulation_txt = None
            self.single_stream_modulation = None
        
        # Positional embedding
        self.pe_embedder = EmbedND(
            dim=params.hidden_size // params.num_heads,
            theta=params.theta,
            axes_dim=list(params.axes_dim),
        )
        
        # Double-stream transformer blocks (joint image-text attention)
        # When global_modulation is True, blocks don't have their own modulation
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                hidden_size=params.hidden_size,
                num_heads=params.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
                global_modulation=params.global_modulation,
                dtype=dtype,
                device=device,
                operations=operations,
                silu_mlp=params.mlp_silu_act,
                gated_mlp=params.gated_mlp,
                ops_bias=params.ops_bias,
            )
            for _ in range(params.depth)
        ])
        
        # Single-stream transformer blocks (merged image-text)
        # When global_modulation is True, blocks don't have their own modulation
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                hidden_size=params.hidden_size,
                num_heads=params.num_heads,
                mlp_ratio=params.mlp_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
                silu_mlp=params.mlp_silu_act,
                gated_mlp=params.gated_mlp,
                ops_bias=params.ops_bias,
                global_modulation=params.global_modulation,
            )
            for _ in range(params.depth_single_blocks)
        ])
        
        # Output layer
        self.final_layer = LastLayer(
            hidden_size=params.hidden_size,
            patch_size=params.patch_size,
            out_channels=params.out_channels,
            dtype=dtype,
            device=device,
            operations=operations,
            ops_bias=params.ops_bias,
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        control=None,
        transformer_options={},
        attn_mask=None,
    ) -> torch.Tensor:
        """Forward pass through the Flux2 model.
        
        Args:
            img: Image latent tensor [B, C, H, W] or already patchified
            txt: Text embeddings [B, L, D]
            timesteps: Timestep tensor [B]
            y: Vector conditioning (pooled text embedding) [B, D]
            guidance: Optional guidance scale tensor [B]
            control: Optional control signals
            transformer_options: Dict with additional options
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input img
        """
        # Get original image dimensions for unpatchifying
        patches_replace = transformer_options.get("patches_replace", {})
        initial_shape = img.shape
        
        # Track if we converted from VAE format (32ch 8x -> 128ch 16x)
        converted_from_vae = False
        
        # Handle input dimensions
        if img.ndim == 4:
            # Input is [B, C, H, W]
            b, c, h_orig, w_orig = img.shape
            
            # Auto-convert from VAE format if needed
            if c == 32 and self.in_channels == 128:
                img = self.latent_format.patchify_from_vae(img)
                converted_from_vae = True
                b, c, h, w = img.shape
            else:
                h, w = h_orig, w_orig
            
            # Pad to patch size (matches ComfyUI's pad_to_patch_size)
            img = self._pad_to_patch_size(img, self.patch_size)
            _, _, h, w = img.shape
            
            img = self._patchify(img)
        else:
            # Assume already patchified [B, L, C]
            b = img.shape[0]
            # Approximate H, W for img_ids
            h = w = int(math.sqrt(img.shape[1] * self.patch_size * self.patch_size / self.in_channels))
            h_orig = w_orig = h
        
        # Create position IDs for RoPE (number of axes matches axes_dim)
        # CRITICAL: Position IDs must ALWAYS be float32 for precision (matches ComfyUI)
        num_axes = len(self.params.axes_dim)
        
        # Support positional offsets for tiling (from UltimateSDUpscale)
        # Offsets are provided in pixels, convert to latent patches
        offset_y = transformer_options.get("top", 0) // 16
        offset_x = transformer_options.get("left", 0) // 16
        
        img_ids = self._create_img_ids(b, h, w, img.device, torch.float32, num_axes, 
                                       offset_y=offset_y, offset_x=offset_x)
        
        # Create text position IDs - CRITICAL: text tokens need positional IDs in txt_ids_dims
        txt_ids = torch.zeros(b, txt.shape[1], num_axes, device=txt.device, dtype=torch.float32)
        if len(self.params.txt_ids_dims) > 0:
            # Give text tokens positional IDs in specified dimensions
            txt_seq_len = txt.shape[1]
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, txt_seq_len - 1, steps=txt_seq_len, 
                                                    device=txt.device, dtype=torch.float32)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        
        # Embed inputs
        img = self.img_in(img)
        
        # Apply text norm if enabled (matches ComfyUI)
        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)
        
        # Time embedding
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        
        # Add vector conditioning (if available)
        if y is not None and self.vector_in is not None:
            vec = vec + self.vector_in(y)
        
        # Add guidance embedding
        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        
        # Compute global modulation (for Flux2/Klein)
        if self.double_stream_modulation_img is not None:
            img_mod1, img_mod2 = self.double_stream_modulation_img(vec)
            txt_mod1, txt_mod2 = self.double_stream_modulation_txt(vec)
            single_mod, _ = self.single_stream_modulation(vec)
        else:
            img_mod1 = img_mod2 = txt_mod1 = txt_mod2 = single_mod = None
        
        # Run double-stream blocks
        for i, block in enumerate(self.double_blocks):
            block_replace = patches_replace.get(f"double_block{i}", {})
            img, txt = block(img, txt, vec, pe, attn_mask, 
                           img_mod=(img_mod1, img_mod2), txt_mod=(txt_mod1, txt_mod2))
            
            # Handle control signals if provided
            if control is not None:
                control_out_i = control.get("output", {}).get(f"double_block{i}")
                if control_out_i is not None:
                    img = img + control_out_i
        
        # Handle fp16 numerical issues (matches ComfyUI exactly)
        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)
        
        # Merge streams
        x = torch.cat((txt, img), dim=1)
        
        # Run single-stream blocks
        for i, block in enumerate(self.single_blocks):
            block_replace = patches_replace.get(f"single_block{i}", {})
            x = block(x, vec, pe, attn_mask, modulation=single_mod)
            
            # Handle control signals
            if control is not None:
                control_out_i = control.get("output", {}).get(f"single_block{i}")
                if control_out_i is not None:
                    x = x + control_out_i
        
        # Extract image portion (remove text tokens)
        img = x[:, txt.shape[1]:, :]
        
        # Final layer
        img = self.final_layer(img, vec)
        
        # Unpatchify back to image shape
        img = self._unpatchify(img, h // self.patch_size, w // self.patch_size)
        
        # If we converted from VAE format, convert back
        if converted_from_vae:
            img = self.latent_format.unpatchify_for_vae(img)
            img = img[:, :, :initial_shape[2], :initial_shape[3]]
        else:
            # Crop back to original size (remove padding - matches ComfyUI)
            img = img[:, :, :h_orig, :w_orig]
        
        return img

    def _pad_to_patch_size(self, img: torch.Tensor, patch_size: int, mode: str = "circular") -> torch.Tensor:
        """Pad image to be divisible by patch size.
        
        Matches ComfyUI's pad_to_patch_size function exactly.
        
        Args:
            img: Image tensor [B, C, H, W]
            patch_size: Patch size to pad to
            mode: Padding mode ("circular", "reflect", etc.)
            
        Returns:
            Padded image tensor
        """
        if mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
            mode = "reflect"
        
        _, _, h, w = img.shape
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            # PyTorch pad format: (left, right, top, bottom)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=mode)
        
        return img

    def _patchify(self, img: torch.Tensor) -> torch.Tensor:
        """Convert image to patch sequence.
        
        Args:
            img: Image tensor [B, C, H, W]
            
        Returns:
            Patch sequence [B, N_patches, patch_dim]
        """
        p = self.patch_size
        b, c, h, w = img.shape
        
        # Reshape into patches
        img = rearrange(img, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        return img

    def _unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert patch sequence back to image.
        
        Args:
            x: Patch sequence [B, N, patch_dim]
            h: Height in patches
            w: Width in patches
            
        Returns:
            Image tensor [B, C, H*patch, W*patch]
        """
        p = self.patch_size
        c = self.out_channels
        
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c)
        return x

    def _create_img_ids(self, batch: int, h: int, w: int, device, dtype, num_axes: int = 3, 
                        offset_y: int = 0, offset_x: int = 0) -> torch.Tensor:
        """Create image position IDs for RoPE.
        
        Matches ComfyUI's img_ids creation exactly for numerical precision.
        
        Returns tensor of shape [B, H*W/patch^2, num_axes] with indices.
        For Flux1: [time=0, row, col] (3 axes)
        For Flux2: [index=0, row, col, extra=0] (4 axes)
        """
        nh = h // self.patch_size
        nw = w // self.patch_size
        
        # Create img_ids matching ComfyUI's format: [h, w, num_axes] then reshape
        img_ids = torch.zeros((nh, nw, num_axes), device=device, dtype=torch.float32)
        
        # Axis 0: index (time/frame), always 0 for single images (like ComfyUI)
        img_ids[:, :, 0] = 0

        # Axis 1: row position using linspace (matches ComfyUI exactly) + offset
        img_ids[:, :, 1] = torch.linspace(offset_y, offset_y + nh - 1, steps=nh, device=device, dtype=torch.float32).unsqueeze(1)
        
        # Axis 2: col position using linspace (matches ComfyUI exactly) + offset
        img_ids[:, :, 2] = torch.linspace(offset_x, offset_x + nw - 1, steps=nw, device=device, dtype=torch.float32).unsqueeze(0)
        
        # Additional axes are zeros (for Flux2 which has 4 axes)
        # Already initialized to zeros
        
        # Reshape to [batch, seq_len, num_axes] and expand
        img_ids = img_ids.reshape(1, -1, num_axes).expand(batch, -1, -1)
        return img_ids

    def get_dtype(self):
        """Get the model dtype."""
        return self.dtype
    
    def process_latent_in(self, latent):
        """Process latent input before sampling (latent format conversion)."""
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        """Process latent output after sampling (latent format conversion)."""
        return self.latent_format.process_out(latent)
    
    def memory_required(self, input_shape):
        """Calculate memory required for given input shape.
        
        Args:
            input_shape: Input tensor shape [B, C, H, W]
            
        Returns:
            Memory required in bytes
        """
        from src.Device import Device
        dtype = self.dtype or torch.bfloat16
        area = input_shape[0] * math.prod(input_shape[2:])
        return area * Device.dtype_size(dtype) * 0.01 * self.memory_usage_factor * 1024 * 1024

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, 
                    transformer_options={}, **kwargs):
        """Apply model to input tensor - interface for sampler.
        
        Args:
            x: Input latent tensor [B, C, H, W]
            t: Timestep/sigma tensor [B]
            c_concat: Optional concat conditioning (unused for Flux2)
            c_crossattn: Text embeddings [B, L, D] from Klein encoder
            control: Optional control signals
            transformer_options: Additional transformer options
            **kwargs: Additional arguments (y/pooled, etc.)
        
        Returns:
            Model output (noise prediction) [B, C, H, W]
        """
        # Get derived values from model_sampling
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        timestep = self.model_sampling.timestep(t).float()
        
        # Cast to model dtype - use non_blocking for async transfer
        dtype = self.dtype or torch.bfloat16
        xc = xc.to(dtype, non_blocking=True)
        
        # Get text conditioning
        txt = c_crossattn.to(dtype, non_blocking=True) if c_crossattn is not None else None
        
        # Get pooled text embedding
        y = kwargs.get("y")
        if y is None:
            y = kwargs.get("pooled_output")
            
        if y is not None:
            y = y.to(dtype, non_blocking=True)
        else:
            # Create dummy pooled if not provided
            batch_size = x.shape[0]
            y = torch.zeros(batch_size, self.params.vec_in_dim, device=x.device, dtype=dtype)
        
        # Guidance (Inject default 3.5 for Flux if missing)
        guidance = kwargs.get("guidance")
        if guidance is None and self.guidance_embed:
            guidance = torch.full((x.shape[0],), 3.5, device=x.device, dtype=dtype)
        
        # Get attention mask for text conditioning (CRITICAL for padding masking)
        attention_mask = kwargs.get("attention_mask")
        
        # Call forward
        output = self.forward(
            img=xc,
            txt=txt,
            timesteps=timestep,
            y=y,
            guidance=guidance,
            control=control,
            transformer_options=transformer_options,
            attn_mask=attention_mask,
        )
        
        return self.model_sampling.calculate_denoised(sigma, output.float(), x)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.
    
    Args:
        t: Timestep tensor [B]
        dim: Embedding dimension
        max_period: Maximum period for frequencies
        time_factor: Scaling factor for timestep (default 1000.0 as in ComfyUI)
        
    Returns:
        Embeddings [B, dim]
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding


def get_flux2_klein_params() -> Flux2Params:
    """Get default parameters for Flux2 Klein 4B model."""
    return Flux2Params(
        in_channels=128,           # Different from standard Flux (16)
        out_channels=128,          # Different from standard Flux (16)
        vec_in_dim=768,            # Unchanged
        context_in_dim=7680,       # From Klein/Qwen3 text encoder (3 layers × 2560)
        hidden_size=3072,          # Model hidden size
        mlp_ratio=3.0,             # Different from standard (4.0)
        num_heads=24,              # hidden_size/sum(axes_dim) = 3072/128 = 24
        depth=5,                   # Klein 4B has 5 double blocks (NOT 19!)
        depth_single_blocks=20,    # Klein 4B has 20 single blocks (NOT 38!)
        axes_dim=(32, 32, 32, 32), # Different from standard (16, 56, 56) - sum=128
        theta=2000,                # Different from standard (10000)
        qkv_bias=False,            # Different from standard (True)
        guidance_embed=False,      # No guidance embedding needed
        global_modulation=True,    # Klein uses global modulation
        mlp_silu_act=True,         # Klein uses SiLU in MLPs
        ops_bias=False,            # No bias in final ops
        patch_size=1,              # Different from standard (2)
    )


def create_flux2_klein(dtype=None, device=None) -> Flux2:
    """Create a Flux2 Klein 4B model instance."""
    params = get_flux2_klein_params()
    return Flux2(params=params, dtype=dtype, device=device)
