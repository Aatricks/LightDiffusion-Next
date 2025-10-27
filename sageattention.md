# SageAttention & SpargeAttn

## Overview

SageAttention and SpargeAttn are drop-in replacements for PyTorch's scaled dot-product attention that can provide significant speedup with zero to minimal quality loss. They work by optimizing the compute-heavy attention mechanism used throughout diffusion models (UNet, VAE, Flux Transformers).

- **SageAttention**: Uses INT8 quantization for key/value tensors while maintaining FP16 query precision
- **SpargeAttn**: Adds dynamic sparsity pruning on top of SageAttention, skipping redundant attention computations

Both are **training-free**, **hardware-accelerated** CUDA kernels that integrate transparently into LightDiffusion-Next.

## How It Works

### SageAttention

Standard attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

SageAttention accelerates this by:

1. **Quantizing K and V** to INT8 before the matrix multiplication
2. **Keeping Q in FP16** to preserve attention score precision
3. **Fusing operations** (softmax, scaling, matmul) in hand-tuned CUDA kernels
4. **Dequantizing** output back to FP16 after final matmul

This reduces memory bandwidth (K/V use half the space) and leverages Tensor Cores more efficiently.

### SpargeAttn

SpargeAttn extends SageAttention with **sparse attention masking**:

1. Computes a similarity metric between query and key patches
2. Prunes attention connections below a learned threshold (default: 60% similarity)
3. Applies cumulative distribution filtering to keep only the top 97% of attention scores
4. Uses partial vector thresholding to skip redundant computations

The result: 40-60% total speedup over baseline PyTorch attention with minimal impact on output quality.

## Installation

### SageAttention (All Platforms)

**Prerequisites:**
- CUDA Toolkit 11.8+ (must match your PyTorch CUDA version)
- Python 3.8+
- PyTorch with CUDA support

**Install:**

```bash
# Clone repository
git clone https://github.com/thu-ml/SageAttention
cd SageAttention

# Install from source (no build isolation to respect existing CUDA setup)
pip install -e . --no-build-isolation

# Verify installation
python -c "import sageattention; print('SageAttention installed successfully')"
```

### SpargeAttn (Linux/WSL2 Only)

**Prerequisites:**
- Same as SageAttention
- Linux or WSL2 environment (Windows native builds fail due to linker path limits)
- GPU with compute capability 8.0-9.0 (RTX 30xx, 40xx, A100, H100)

**Install:**

```bash
# Clone repository
git clone https://github.com/thu-ml/SparseAttention
cd SpargeAttn

# Set GPU architecture (critical for performance)
export TORCH_CUDA_ARCH_LIST="9.0"  # Or your GPU: 8.0, 8.6, 8.9, 9.0

# Install from source
pip install -e . --no-build-isolation

# Verify installation
python -c "import spas_sage_attn; print('SpargeAttn installed successfully')"
```

**GPU Architecture Reference:**

| GPU Model | Compute Capability | TORCH_CUDA_ARCH_LIST |
|-----------|-------------------|----------------------|
| RTX 3060/3070/3080/3090 | 8.6 | `"8.6"` |
| RTX 4060/4070/4080/4090 | 8.9 | `"8.9"` |
| A100 | 8.0 | `"8.0"` |
| H100 | 9.0 | `"9.0"` |
| RTX 5060/5070/5080/5090 | 12.0 | Not supported yet |

### Docker Installation

Both kernels are automatically built during the Docker image creation if the architecture is supported:

```bash
# Build with SpargeAttn (compute 8.0-9.0)
docker-compose build --build-arg TORCH_CUDA_ARCH_LIST="8.9"

# RTX 50xx builds (SageAttention only, no SpargeAttn yet)
docker-compose build --build-arg TORCH_CUDA_ARCH_LIST="12.0"
```

## Usage

### Automatic Detection

LightDiffusion-Next automatically detects and enables the best available attention backend at startup:

```python
# Priority order (highest to lowest):
SpargeAttn > SageAttention > xformers > PyTorch SDPA
```

Check which backend is active in the server logs:

```bash
# SpargeAttn enabled
cat logs/server.log | grep "attention"
# Output: Using SpargeAttn (Sparse + SageAttention) cross attention

# SageAttention enabled
# Output: Using SageAttention cross attention

# Fallback
# Output: Using pytorch cross attention
```

### Streamlit UI

No configuration needed — SageAttention/SpargeAttn are always active if installed.

### REST API

Same as UI — the backend selection is transparent:

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a serene mountain lake at dawn",
        "width": 768,
        "height": 512,
        "num_images": 1
      }'
# Automatically uses SpargeAttn if available
```

### Manual Disable

Force PyTorch SDPA for debugging:

```bash
export LD_DISABLE_SAGE_ATTENTION=1
python streamlit_app.py
```

## Performance

Both SageAttention and SpargeAttn provide measurable speedup over PyTorch SDPA baseline:

- **SageAttention**: Moderate speedup with zero quality loss (reported ~15-20% in papers)
- **SpargeAttn**: Significant speedup with minimal quality loss (reported ~40-60% in papers)

Actual performance gains vary based on:
- GPU architecture and VRAM
- Model type (SD1.5, SDXL, Flux)
- Resolution and batch size
- Head dimensions and sequence lengths

**Note:** Benchmark your specific setup to measure real-world performance.## Technical Details

### Head Dimension Support

Both kernels natively support head dimensions of `[64, 96, 128]`. For other dimensions:

- **< 64**: Pad to 64, compute, then slice result
- **64-128**: Pad to 128, compute, then slice result
- **> 128**: Fallback to xformers or PyTorch SDPA

LightDiffusion-Next handles padding/slicing automatically.

### Tensor Layout

SageAttention expects tensors in `(batch_size, num_heads, seq_len, head_dim)` format. The pipeline reshapes inputs transparently:

```python
# Internal reshaping (handled automatically)
q, k, v = map(
    lambda t: t.reshape(b, -1, heads, dim_head).transpose(1, 2),
    (q, k, v),
)
out = sageattention.sageattn(q, k, v, tensor_layout="HND")
```

### SpargeAttn Thresholds

Default pruning parameters (tuned for quality/speed balance):

```python
out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
    q, k, v,
    simthreshd1=0.6,      # Similarity threshold (60%)
    cdfthreshd=0.97,      # Keep top 97% of attention scores
    pvthreshd=15,         # Partial vector threshold
    is_causal=False
)
```

Adjust `simthreshd1` for different trade-offs:
- `0.5`: More aggressive pruning, higher speedup, slight quality loss
- `0.7`: Conservative pruning, lower speedup, minimal quality loss

## Compatibility

### Compatible With

- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion 2.1
- ✅ SDXL
- ✅ Flux (both cross-attention and self-attention blocks)
- ✅ All samplers (Euler, DPM++, etc.)
- ✅ LoRA adapters
- ✅ Textual inversion embeddings
- ✅ HiresFix, ADetailer, Img2Img
- ✅ Stable-Fast (when stacked)
- ✅ WaveSpeed caching (when stacked)

### Known Limitations

- ❌ RTX 50xx (compute 12.0) does not support SpargeAttn yet (SageAttention works)
- ❌ CPU-only inference (CUDA required)
- ❌ AMD GPUs (ROCm port not available)
- ⚠️ Head dimensions > 128 fall back to slower backends

## Troubleshooting

### Import Error: `No module named 'sageattention'`

**Cause:** Not installed or installation failed.

**Fix:**
```bash
cd SageAttention
pip install -e . --no-build-isolation
```

Verify CUDA toolkit is accessible:
```bash
nvcc --version  # Should match PyTorch CUDA version
```

### Compilation Error: `nvcc fatal error`

**Cause:** CUDA toolkit not found or version mismatch.

**Fix:**
1. Install CUDA toolkit matching your PyTorch version
2. Add CUDA to PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. Reinstall SageAttention

### SpargeAttn Build Fails on Windows

**Cause:** Windows linker has path length limitations.

**Fix:** Use WSL2 or native Linux:
```bash
# In WSL2
cd SpargeAttn
export TORCH_CUDA_ARCH_LIST="8.9"
pip install -e . --no-build-isolation
```

### Slower Than Expected

**Cause:** Wrong GPU architecture compiled or kernel fallback.

**Fix:**
1. Check logs for "Using pytorch cross attention" (fallback indicator)
2. Rebuild with correct `TORCH_CUDA_ARCH_LIST`
3. Verify GPU compute capability:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

### Quality Degradation with SpargeAttn

**Cause:** Pruning thresholds too aggressive.

**Fix:** Currently not user-configurable in the UI, but you can modify `src/Attention/AttentionMethods.py`:
```python
# Line ~290
out = spas_sage_attn.spas_sage2_attn_meansim_cuda(
    q, k, v,
    simthreshd1=0.7,      # Increase from 0.6 for better quality
    cdfthreshd=0.98,      # Increase from 0.97
    pvthreshd=15,
    is_causal=False
)
```

## Citation

If you use SageAttention or SpargeAttn in your work:

```bibtex
@article{sageattention2024,
  title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration},
  author={Zhang, Jintao and Zhang, Jia and Zhai, Pengle and others},
  journal={arXiv preprint arXiv:2410.02367},
  year={2024}
}

@article{spargeattn2024,
  title={SpargeAttn: Sparsity-Aware Efficient Attention for Long Context LLMs},
  author={Zhang, Jintao and others},
  journal={arXiv preprint},
  year={2024}
}
```

## Resources

- [SageAttention Repository](https://github.com/thu-ml/SageAttention)
- [SpargeAttn Repository](https://github.com/thu-ml/SparseAttention)
- [SageAttention Paper](https://arxiv.org/abs/2410.02367)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) (related work)
