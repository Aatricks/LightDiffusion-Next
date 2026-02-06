"""Klein text encoder for Flux2 models in LightDiffusion-Next.

This module provides the Klein (Qwen3-4B based) text encoder used by
Flux2 Klein models, including:
- KleinTokenizer: Qwen3-based tokenizer with special formatting
- Qwen3Model: Transformer-based language model for text encoding

Adapted from ComfyUI's Klein implementation.
"""

import logging
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.cond import cast as ops_module
from src.Device import Device


logger = logging.getLogger(__name__)


def get_ops():
    """Get the operations module for weight initialization."""
    return ops_module.disable_weight_init


class QwenRMSNorm(nn.Module):
    """RMS Normalization for Qwen3."""
    
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        if operations is None:
            operations = get_ops()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization - compute in float32 for precision, cast back to input dtype
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.float()).to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    """Rotary position embeddings for Qwen3."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for RoPE."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QwenAttention(nn.Module):
    """Multi-head attention for Qwen3 with Grouped Query Attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        head_dim: int = 128,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        
        # Qwen3 uses separate projections with different output sizes
        self.q_proj = operations.Linear(hidden_size, num_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(hidden_size, self.num_kv_heads * head_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(num_heads * head_dim, hidden_size, bias=False, dtype=dtype, device=device)
        
        # Normalize Q and K
        self.q_norm = QwenRMSNorm(head_dim, dtype=dtype, device=device, operations=operations)
        self.k_norm = QwenRMSNorm(head_dim, dtype=dtype, device=device, operations=operations)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Grouped query attention - repeat K,V for each group
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        # Ensure all tensors have same dtype for SDPA
        attn_dtype = q.dtype
        k = k.to(attn_dtype)
        v = v.to(attn_dtype)
        
        # Scaled dot-product attention with causal masking
        # Use is_causal=True for efficiency, or attn_mask for custom masks
        if attention_mask is None:
            # Pure causal masking
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Custom mask (includes causal + padding) - ensure mask dtype matches
            attention_mask = attention_mask.to(attn_dtype)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        
        # Reshape back and ensure output dtype matches input for o_proj
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = attn_output.to(hidden_states.dtype)  # Match input dtype for o_proj
        return self.o_proj(attn_output)


class QwenMLP(nn.Module):
    """MLP (Gate-Up-Down) for Qwen3."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
        
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenDecoderLayer(nn.Module):
    """Single transformer decoder layer for Qwen3."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: int = None,
        head_dim: int = 128,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
        
        self.self_attn = QwenAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.mlp = QwenMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.input_layernorm = QwenRMSNorm(hidden_size, dtype=dtype, device=device, operations=operations)
        self.post_attention_layernorm = QwenRMSNorm(hidden_size, dtype=dtype, device=device, operations=operations)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Qwen3_4BModel(nn.Module):
    """Qwen3 4B model for Klein text encoding.
    
    This is a decoder-only transformer used as a text encoder
    for the Flux2 Klein model.
    """
    
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,  # Matches checkpoint
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,   # Matches checkpoint (4096/128)
        num_key_value_heads: int = 8,    # Matches checkpoint (1024/128)
        head_dim: int = 128,
        max_position_embeddings: int = 32768,
        layer_indices: tuple = (9, 18, 27),  # Layers to extract embeddings from
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        if operations is None:
            operations = get_ops()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_indices = layer_indices
        
        # Token embeddings
        self.embed_tokens = operations.Embedding(vocab_size, hidden_size, dtype=dtype, device=device)
        
        # Rotary embeddings
        self.rotary_emb = QwenRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            QwenDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final norm
        self.norm = QwenRMSNorm(hidden_size, dtype=dtype, device=device, operations=operations)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass returning hidden states from specified layers.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            Dict with 'hidden_states' from specified layers (concatenated)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens - ensure output is at least bfloat16 for subsequent math
        hidden_states = self.embed_tokens(input_ids).to(torch.bfloat16)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        position_embeddings = (cos, sin)
        
        # Prepare attention mask
        # If we have a padding mask, create a combined causal + padding mask
        # Otherwise, pass None and let the attention layer use is_causal=True
        final_mask = None
        if attention_mask is not None:
            # Create mask matching ComfyUI's approach:
            # 1. Convert padding mask from [B, L] to [B, 1, L, L] with expansion
            # 2. Set padded positions (where mask=0) to -inf
            # 3. Add causal mask
            
            # Reshape and expand: [B, L] -> [B, 1, L, L]
            mask = 1.0 - attention_mask.to(hidden_states.dtype)  # 0 = valid, 1 = padding
            mask = mask.reshape(mask.shape[0], 1, -1, mask.shape[-1])  # [B, 1, 1, L]
            mask = mask.expand(mask.shape[0], 1, seq_len, seq_len)  # [B, 1, L, L]
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))
            
            # Create causal mask [L, L]
            causal_mask = torch.empty(seq_len, seq_len, dtype=hidden_states.dtype, device=input_ids.device).fill_(float("-inf")).triu_(1)
            
            # Combine
            final_mask = mask + causal_mask
        
        # Collect outputs from specified layers
        # NOTE: ComfyUI captures the INPUT to layers (before the layer runs),
        # so we capture before applying each layer
        layer_outputs = []
        
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, final_mask, position_embeddings)
            
            # Capture AFTER the layer (output of layer i)
            if i in self.layer_indices:
                layer_outputs.append(hidden_states.clone())
        
        # Apply final norm
        hidden_states = self.norm(hidden_states)
        
        # Concatenate layer outputs matching ComfyUI's interleaving pattern
        # This is critical for Flux2/Klein cross-attention
        if layer_outputs:
            # layer_outputs is a list of [B, L, D] tensors
            # stack: (B, 3, L, D)
            stacked = torch.stack(layer_outputs, dim=1)
            # permute: (B, L, 3, D) - interleave the 3 layers at each sequence position
            permuted = stacked.permute(0, 2, 1, 3)
            # reshape: (B, L, 3*D)
            concatenated = permuted.reshape(batch_size, seq_len, -1)
        else:
            concatenated = hidden_states
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": concatenated,
            "pooled_output": None,  # Match ComfyUI: No pooling for Qwen -> Flux2 uses zeros
        }


class KleinTokenizer:
    """Tokenizer for Klein (Qwen3-based) text encoder.
    
    Uses Qwen2Tokenizer from Hugging Face transformers with
    Klein-specific formatting template.
    """
    
    # Klein template for prompt formatting
    TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def __init__(
        self,
        tokenizer_path: str = None,
        max_length: int = 99999999,  # ComfyUI uses essentially unlimited
        min_length: int = 512,  # ComfyUI uses min_length=512 for Klein
        padding: str = "do_not_pad",  # ComfyUI uses pad_to_max_length=False
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.padding = padding
        
        # Klein special tokens
        self.pad_token_id = 151643  # <|endoftext|>
        self.bos_token_id = 151644  # <|im_start|>
        self.eos_token_id = 151645  # <|im_end|>
        
        # Load the real tokenizer
        if tokenizer_path is None:
            # Default path relative to include folder
            import os
            # Try multiple locations
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "include", "text_encoder", "qwen25_tokenizer"),
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    tokenizer_path = path
                    break
            else:
                tokenizer_path = possible_paths[0]  # Use first as default
        
        try:
            from transformers import Qwen2Tokenizer
            self._tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            # Use right padding for content-first alignment, matching ComfyUI default
            self._tokenizer.padding_side = "right"
            logger.info(f"Loaded Qwen2Tokenizer from {tokenizer_path}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise RuntimeError(f"Could not load Klein tokenizer from {tokenizer_path}") from e
    
    def apply_template(self, text: str) -> str:
        """Apply Klein's prompt template to input text."""
        return self.TEMPLATE.format(text)
    
    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict:
        """Tokenize text with Klein template formatting.
        
        Args:
            text: Input text to tokenize
            return_word_ids: Whether to return word IDs
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        # Apply template
        formatted_text = self.apply_template(text)
        
        # Tokenize with the real tokenizer - pad to min_length (512)
        # Matches ComfyUI Qwen3Tokenizer behavior
        encoded = self._tokenizer(
            formatted_text,
            padding="max_length",
            max_length=self.min_length,
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        
        if return_word_ids:
            # Word IDs from the tokenizer's encoding
            word_ids = encoded.word_ids() if hasattr(encoded, 'word_ids') else list(range(encoded["input_ids"].shape[1]))
            result["word_ids"] = word_ids
        
        return result
    
    def state_dict(self) -> dict:
        """Return tokenizer state for serialization."""
        return {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "padding": self.padding,
        }


class KleinCLIP:
    """Klein text encoder wrapper compatible with CLIP interface.
    
    This provides the same interface as other CLIP models while
    using the Qwen3-based Klein encoder internally.
    
    VRAM Optimization: Model stays on CPU until encoding, then moves to GPU
    and back to CPU. This follows ComfyUI's lazy loading approach.
    """
    
    def __init__(
        self,
        tokenizer: KleinTokenizer = None,
        model: Qwen3_4BModel = None,
        dtype=None,
        device=None,
        offload_device=None,
    ):
        self.tokenizer = tokenizer or KleinTokenizer()
        self.dtype = dtype
        self.device = device  # Device to use for encoding (GPU)
        self.offload_device = offload_device or torch.device("cpu")  # Device when idle (CPU)
        
        if model is None:
            self.model = Qwen3_4BModel(dtype=dtype, device=self.offload_device)
        else:
            self.model = model
        
        self.clip_options = {}
    
    def reset_clip_options(self):
        """Reset clip options to defaults."""
        self.clip_options = {}
    
    def set_clip_options(self, options: dict):
        """Set clip options (for API compatibility)."""
        self.clip_options.update(options)
    
    def encode_token_weights(self, tokens: dict) -> tuple:
        """Encode token weights returning (embeddings, pooled, extra).
        
        VRAM Optimization: Moves model to GPU only during encoding,
        then offloads back to CPU to free VRAM for diffusion model.
        
        Args:
            tokens: Dict with 'input_ids' and 'attention_mask' tensors
            
        Returns:
            Tuple of (hidden_states, pooled_output, extra_dict)
            where extra_dict contains 'attention_mask' for the diffusion model
        """
        input_ids = tokens.get("input_ids")
        if input_ids is None:
            raise ValueError("tokens dict must contain 'input_ids'")
        
        # Move model to GPU for encoding
        logger.info(f"Moving text encoder to {self.device} for encoding...")
        self.model = self.model.to(self.device)
        
        input_ids = input_ids.to(self.device)
        
        # Get attention mask if present - CRITICAL for proper masking of padding tokens
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Return concatenated hidden states, pooled output, and extra with attention_mask
            hidden_states = outputs["hidden_states"].clone()  # Clone to keep on GPU
            pooled_out = outputs["pooled_output"]
            pooled = pooled_out.clone() if pooled_out is not None else None  # Clone if exists
        finally:
            # Offload model back to CPU to free VRAM for diffusion model
            logger.info(f"Offloading text encoder to {self.offload_device}...")
            self.model = self.model.to(self.offload_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Return attention mask in extra dict for the diffusion model to use
        extra = {}
        if attention_mask is not None:
            extra["attention_mask"] = attention_mask
        
        return hidden_states, pooled, extra
    
    def tokenize(self, text, return_word_ids=False):
        """Tokenize text (CLIP interface compatibility for Adetailer).
        
        Args:
            text: Text to tokenize
            return_word_ids: Whether to return word IDs
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)
    
    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        """Encode from tokens (CLIP interface compatibility for Adetailer).
        
        Args:
            tokens: Dict with 'input_ids' and 'attention_mask'
            return_pooled: Whether to return pooled output
            return_dict: Whether to return as dict
            
        Returns:
            Embeddings, or (embeddings, pooled) if return_pooled, or dict if return_dict
        """
        cond, pooled, extra = self.encode_token_weights(tokens)
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            out.update(extra)
            return out
        return (cond, pooled) if return_pooled else cond
    
    def load_model(self):
        """Load model to GPU (CLIP interface compatibility).
        
        Returns:
            Self for compatibility
        """
        # Move model to device if not already there
        if self.device is not None:
            self.model = self.model.to(self.device)
        return self
    
    def load_sd(self, state_dict: dict) -> tuple:
        """Load state dictionary into model.
        
        Args:
            state_dict: Model weights
            
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        # Filter and map state dict keys for Qwen3 model
        model_sd = {}
        for k, v in state_dict.items():
            # Map state dict keys to model structure
            if k.startswith("model."):
                model_sd[k[6:]] = v  # Remove "model." prefix
            else:
                model_sd[k] = v
        
        return self.model.load_state_dict(model_sd, strict=False)


def klein_clip(dtype=None) -> dict:
    """Create Klein CLIP configuration.
    
    Returns:
        Dict with 'clip' and 'tokenizer' classes
    """
    class Target:
        clip = KleinCLIP
        tokenizer = KleinTokenizer
        params = {"dtype": dtype}
    
    return Target


# Convenience function to detect Klein model from state dict
def detect_klein_model(state_dict: dict) -> bool:
    """Detect if state dict is from a Klein text encoder.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        True if this appears to be a Klein model
    """
    klein_indicators = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "embed_tokens.weight",
    ]
    
    keys = set(state_dict.keys())
    for indicator in klein_indicators:
        for key in keys:
            if indicator in key:
                return True
    
    return False
