import torch
import logging
import math

logger = logging.getLogger(__name__)

# NVFP4 (E2M1) Table
# exp=0, mant=0 -> 0.0
# exp=0, mant=1 -> 0.5
# exp=1, mant=0 -> 1.0
# exp=1, mant=1 -> 1.5
# exp=2, mant=0 -> 2.0
# exp=2, mant=1 -> 3.0
# exp=3, mant=0 -> 4.0
# exp=3, mant=1 -> 6.0
NVFP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

def stochastic_float_to_fp4_e2m1(x, generator=None):
    """Convert float tensor to packed 4-bit E2M1 format with stochastic rounding."""
    device = x.device
    
    # Ensure the last dimension is even for packing
    orig_last_dim = x.shape[-1]
    if orig_last_dim % 2 != 0:
        x = torch.nn.functional.pad(x, (0, 1))
    
    orig_shape = x.shape
    
    # Calculate exponent for stochastic noise scaling
    # x.abs() log2 + 1 gives a rough exponent
    exp = torch.floor(torch.log2(x.abs() + 1e-8) + 1.0).clamp(0, 3)
    
    # Add stochastic noise scaled by exponent if generator is provided
    if generator is not None:
        noise = (torch.rand(x.size(), dtype=x.dtype, device=device, generator=generator) - 0.5)
        x = x + noise * (2 ** (exp - 2.0)) * 1.25

    sign = torch.signbit(x).to(torch.uint8)
    x = x.abs()
    
    # Recalculate exponent after noise
    exp = torch.floor(torch.log2(x + 1e-8) + 1.1925).clamp(0, 3)

    # Calculate mantissa
    # If exp > 0: val = (1 + m/2) * 2^(exp-1) => m = (val / 2^(exp-1) - 1) * 2
    # If exp = 0: val = m/2 => m = val * 2
    mantissa = torch.where(
        exp > 0,
        (x / (2.0 ** (exp - 1)) - 1.0) * 2.0,
        (x * 2.0)
    ).round().clamp(0, 1).to(torch.uint8)

    # Pack into 4 bits: [sign:1, exp:2, mantissa:1]
    fp4 = (sign << 3) | (exp.to(torch.uint8) << 1) | mantissa
    
    # Pack two 4-bit values into one uint8
    fp4_flat = fp4.view(-1)
    # We already padded x to be even, so fp4_flat.numel() is even
        
    packed = (fp4_flat[0::2] << 4) | fp4_flat[1::2]
    
    new_shape = list(orig_shape)
    new_shape[-1] = new_shape[-1] // 2
    return packed.reshape(new_shape)

def to_blocked(input_matrix, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange a matrix by breaking it into blocks and applying the rearrangement pattern.
    Matches NVIDIA's block scaling factors layout.
    """
    def ceil_div(a, b):
        return (a + b - 1) // b

    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix
    else:
        padded = input_matrix

    # Rearrange the blocks: [n_row_blocks, 128, n_col_blocks, 4] -> [n_row_blocks, n_col_blocks, 128, 4]
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    
    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)

def from_blocked(blocked_matrix, original_rows, original_cols):
    """Inverse of to_blocked."""
    def ceil_div(a, b):
        return (a + b - 1) // b

    n_row_blocks = ceil_div(original_rows, 128)
    n_col_blocks = ceil_div(original_cols, 4)
    
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    
    # [Total_Blocks, 32, 16]
    rearranged = blocked_matrix.reshape(-1, 32, 16)
    # [Total_Blocks, 4, 32, 4] -> [Total_Blocks, 128, 4]
    blocks = rearranged.reshape(-1, 32, 4, 4).transpose(1, 2).reshape(n_row_blocks, n_col_blocks, 128, 4)
    # [n_row_blocks, 128, n_col_blocks, 4]
    padded = blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    
    return padded[:original_rows, :original_cols]

def quantize_nvfp4(tensor, stochastic_rounding=0):
    """Quantize tensor to NVFP4 format."""
    if tensor.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

    F4_E2M1_MAX = 6.0
    F8_E4M3_MAX = 448.0
    
    orig_shape = tensor.shape
    device = tensor.device
    
    # Calculate per-tensor scale
    # We want max(abs(x)) / (tensor_scale * block_scale) <= F4_MAX
    # And block_scale <= F8_MAX
    # So tensor_scale ~ max(abs(x)) / (F8_MAX * F4_MAX)
    tensor_scale = torch.amax(tensor.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
    if tensor_scale == 0:
        tensor_scale = torch.tensor(1.0, device=device)
    
    # Block size is 16 elements along the last dimension
    block_size = 16
    rows, cols = tensor.shape
    padded_cols = (cols + block_size - 1) // block_size * block_size
    if cols != padded_cols:
        x = torch.nn.functional.pad(tensor, (0, padded_cols - cols))
    else:
        x = tensor
    x = x.reshape(rows, -1, block_size)
    
    # Calculate per-block scales (FP8 E4M3)
    # block_scale = max(abs(block)) / (tensor_scale * F4_MAX)
    block_scales = (torch.amax(torch.abs(x), dim=-1) / (tensor_scale * F4_E2M1_MAX)).clamp(max=F8_E4M3_MAX)
    
    # Normalize by scales
    # x_norm = x / (tensor_scale * block_scale)
    x = x / (tensor_scale * block_scales.unsqueeze(-1) + 1e-12)
    x = x.view(rows, padded_cols)[:, :cols].reshape(orig_shape).nan_to_num()
    
    generator = None
    if stochastic_rounding > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(stochastic_rounding)
        
    qdata = stochastic_float_to_fp4_e2m1(x, generator=generator)
    
    # ComfyUI expects block_scales in a specific "blocked" layout
    blocked_scales = to_blocked(block_scales, flatten=False)
    
    return qdata, tensor_scale, blocked_scales

def dequantize_nvfp4(qdata, tensor_scale, blocked_scales, original_shape):
    """Dequantize NVFP4 data back to float."""
    device = qdata.device
    
    # Ensure scales are on the correct device
    if isinstance(tensor_scale, torch.Tensor):
        tensor_scale = tensor_scale.to(device)
    else:
        tensor_scale = torch.tensor(tensor_scale, device=device)
        
    blocked_scales = blocked_scales.to(device)
    
    # 1. Unpack uint8 to two 4-bit values
    high = (qdata >> 4) & 0x0F
    low = qdata & 0x0F
    
    rows, cols = original_shape
    # Each row in qdata has (cols + 1) // 2 elements
    # So we stack them and reshape to (rows, -1) to get the padded width
    fp4 = torch.stack([high, low], dim=-1).reshape(rows, -1)[:, :cols].reshape(original_shape)
    
    # 2. Map indices to values
    # sign: bit 3, index: bits 0-2
    sign = (fp4 >> 3).to(torch.float32)
    sign = 1.0 - 2.0 * sign # 0 -> 1.0, 1 -> -1.0
    
    indices = fp4 & 0x07
    values = NVFP4_TABLE.to(device)[indices.long()]
    
    x = sign * values
    
    # 3. Undo block scaling
    # blocked_scales shape: [padded_rows, padded_cols]
    rows, cols = original_shape
    block_cols = (cols + 15) // 16
    
    if blocked_scales.shape == (rows, block_cols):
        block_scales = blocked_scales
    else:
        block_scales = from_blocked(blocked_scales, rows, block_cols)
    
    # block_scales is [rows, block_cols], each scale covers 16 elements
    
    padded_cols = block_cols * 16
    if cols != padded_cols:
        x_padded = torch.nn.functional.pad(x.view(rows, cols), (0, padded_cols - cols))
    else:
        x_padded = x.view(rows, cols)
    
    x_padded = x_padded.reshape(rows, -1, 16)
    x_padded = x_padded * block_scales.to(x.dtype).unsqueeze(-1)
    x = x_padded.view(rows, padded_cols)[:, :cols].reshape(original_shape)
    
    # 4. Apply per-tensor scale
    x = x * tensor_scale.to(x.dtype)
    
    return x
