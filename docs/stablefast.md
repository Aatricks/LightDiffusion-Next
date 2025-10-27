# Stable-Fast Compilation

## Overview

Stable-Fast is a JIT compilation framework that optimizes Stable Diffusion UNet models by tracing execution, fusing operators and optionally capturing CUDA graphs. It can provide significant speedup for SD1.5/SDXL batch workflows with zero quality loss.

Unlike runtime attention optimizations (SageAttention, SpargeAttn), Stable-Fast performs **ahead-of-time compilation** on the first inference pass. The compiled model is cached and reused for subsequent generations with compatible shapes.

## How It Works

Stable-Fast applies three optimization layers:

### 1. TorchScript Tracing

The first forward pass through the UNet is recorded into a static computational graph:

```python
traced_model = torch.jit.trace(unet, example_inputs)
```

This eliminates Python interpreter overhead and enables downstream graph optimizations.

### 2. Operator Fusion

The traced graph undergoes pattern-based fusion:

- **Conv + BatchNorm fusion**: Merges normalization into convolution weights
- **Activation fusion**: Fuses ReLU/GELU/SiLU directly into linear/conv ops
- **Memory layout optimization**: Converts to channels-last format for faster conv execution
- **Triton kernels**: Replaces PyTorch ops with hand-tuned Triton implementations (if `enable_triton=True`)

Example fusion:

```python
# Before:
x = conv(input)
x = batch_norm(x)
x = relu(x)

# After:
x = fused_conv_bn_relu(input)  # Single kernel launch
```

### 3. CUDA Graph Capture (Optional)

When `enable_cuda_graph=True`, the entire forward pass is captured as a static CUDA graph:

- Kernel launches are recorded once and replayed on subsequent runs
- Eliminates CPU launch overhead (~10-15% speedup)
- Requires fixed input shapes and batch sizes

**Trade-off:** Higher VRAM usage (~500MB for graph buffers) and less flexibility.

## Installation

### Windows/Linux (Manual)

Follow the [official guide](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation):

```bash
# Install from PyPI (recommended)
pip install stable-fast

# Or build from source for latest features
git clone https://github.com/chengzeyi/stable-fast
cd stable-fast
pip install -e .
```

**Prerequisites:**
- PyTorch 2.0+ with CUDA support
- xformers (optional but recommended)
- Triton (optional for Triton kernel fusion)

### Docker

Stable-Fast is included in the Docker image when `INSTALL_STABLE_FAST=1`:

```bash
docker-compose build --build-arg INSTALL_STABLE_FAST=1
```

Default is `0` (disabled) to reduce image size and build time.

## Usage

### Streamlit UI

Enable in the **Performance** section of the sidebar:

1. Check **Stable Fast**
2. Generate images — the first run compiles the model (30-60s delay)
3. Subsequent generations reuse the cached compiled model

**Visual indicator:** The first generation shows "Compiling model..." in the progress bar.

### REST API

Pass `stable_fast: true` in the request payload:

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a peaceful garden with cherry blossoms",
        "width": 768,
        "height": 512,
        "num_images": 1,
        "stable_fast": true
      }'
```

### Configuration

Stable-Fast behavior is controlled by `CompilationConfig`:

```python
from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig

config = CompilationConfig.Default()
config.enable_xformers = True           # Use xformers attention
config.enable_cuda_graph = False        # CUDA graphs (set True for max speed)
config.enable_jit_freeze = True         # Freeze traced graph
config.enable_cnn_optimization = True   # Conv fusion
config.enable_triton = False            # Triton kernels (experimental)
config.memory_format = torch.channels_last  # Optimize memory layout
```

LightDiffusion-Next uses sensible defaults (CUDA graphs disabled by default for flexibility). To override:

```python
# In src/StableFast/StableFast.py
def gen_stable_fast_config(enable_cuda_graph=False):
    config = CompilationConfig.Default()
    config.enable_cuda_graph = enable_cuda_graph  # Pass True for max speed
    # ... rest of config
```

## Performance

### Speedup Benchmarks

Stable-Fast provides speedup through:
- **JIT compilation**: Eliminates Python overhead
- **Operator fusion**: Reduces kernel launches
- **CUDA graphs** (optional): Further reduces CPU overhead

Speedup varies significantly based on:
- GPU architecture
- Batch size and generation count
- Model size (SD1.5 vs SDXL)
- Whether CUDA graphs are enabled

**Note:** Performance benefits are most noticeable for batch operations (50+ images). For single 20-step generations, compilation overhead may exceed speedup gains.

### Compilation Time

First-run compilation overhead:

- **SD1.5 UNet**: ~30s (traced once per resolution/batch size)
- **SDXL UNet**: ~60s (larger model)
- **Subsequent runs**: <1s (cached)

Cached compiled models persist in `~/.cache/torch_extensions/`. Clear this directory to force recompilation.

## Stacking with Other Optimizations

Stable-Fast is **fully compatible** with SageAttention, SpargeAttn and WaveSpeed:

### Stable-Fast + SageAttention

```yaml
stable_fast: true
# SageAttention auto-detected
```

**Result:** 70% (Stable-Fast) + 15% (SageAttention) = **~2x total speedup**

### Stable-Fast + SpargeAttn

```yaml
stable_fast: true
# SpargeAttn auto-detected
```

**Result:** 70% (Stable-Fast) + 40% (SpargeAttn) = **~2.4x total speedup**

### Stable-Fast + SpargeAttn + DeepCache

```yaml
stable_fast: true
deepcache:
  enabled: true
  interval: 3
  depth: 2
# SpargeAttn auto-detected
```

**Result:** 70% × 40% × 150% (DeepCache 2-3x) = **~4-5x total speedup**

## Compatibility

### Compatible With

- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion 2.1
- ✅ SDXL
- ✅ All samplers (Euler, DPM++, etc.)
- ✅ LoRA adapters
- ✅ Textual inversion embeddings
- ✅ HiresFix
- ✅ ADetailer
- ✅ Img2Img (with fixed denoise strength)
- ✅ SageAttention/SpargeAttn
- ✅ WaveSpeed caching

### Not Compatible With

- ❌ Flux models (different architecture, no UNet)
- ❌ Dynamic resolution changes after compilation
- ❌ Dynamic batch size changes after compilation (with CUDA graphs)
- ⚠️ Frequent model switching (recompiles each time)

## Troubleshooting

### Slow First Run / Repeated Recompilation

**Symptom:** Every generation triggers compilation, even with identical settings.

**Causes:**
1. Cache directory not writable
2. System clock incorrect (invalidates timestamps)
3. Different model loaded (each model is cached separately)

**Fixes:**
```bash
# Check cache permissions
ls -la ~/.cache/torch_extensions

# Ensure stable timestamps
date  # Should be correct

# Mount cache in Docker to persist across container restarts
docker run -v ~/.cache/torch_extensions:/root/.cache/torch_extensions ...
```

### CUDA Out of Memory During Compilation

**Symptom:** OOM error on first run but not subsequent runs.

**Cause:** Compilation allocates temporary buffers for tracing.

**Fixes:**
1. Disable CUDA graphs: `enable_cuda_graph=False` (saves ~500MB)
2. Reduce batch size temporarily for first run
3. Clear other VRAM consumers (close other apps, disable model caching)

### Compilation Hangs or Crashes

**Symptom:** Process freezes during "Compiling model..." step.

**Causes:**
1. Triton compilation error (if `enable_triton=True`)
2. Driver incompatibility
3. Insufficient CPU RAM for graph analysis

**Fixes:**
```bash
# Disable Triton
# In src/StableFast/StableFast.py:
config.enable_triton = False

# Update NVIDIA driver
nvidia-smi  # Check version, upgrade if < 525.x

# Increase Docker memory limit
# In docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 16G  # Increase from default
```

### Error: `torch.jit.trace` fails

**Symptom:** `RuntimeError: Could not trace model`

**Cause:** Dynamic control flow in model (if/else statements depending on runtime values).

**Fix:** This is rare with standard SD models. If it occurs:
1. Check for custom LoRA/embeddings with dynamic logic
2. Disable Stable-Fast for that specific generation
3. Report issue with model details

### Model Quality Degradation

**Symptom:** Compiled model produces different outputs than baseline.

**Cause:** Numeric precision differences from operator fusion (very rare).

**Fixes:**
```python
# Disable aggressive optimizations
config.enable_cnn_optimization = False
config.memory_format = None  # Use default layout
```

If issue persists, disable Stable-Fast and file a bug report.

## Advanced Configuration

### Custom Compilation Config

Override defaults in `src/StableFast/StableFast.py`:

```python
def gen_stable_fast_config(enable_cuda_graph=False):
    config = CompilationConfig.Default()
    
    # Maximum speed (higher VRAM usage)
    config.enable_cuda_graph = True
    config.enable_triton = True
    config.prefer_lowp_gemm = True  # Use FP16 matrix multiplies
    
    # Balanced (recommended)
    config.enable_cuda_graph = False
    config.enable_triton = False
    config.enable_cnn_optimization = True
    
    # Debug (no optimizations)
    config.enable_cuda_graph = False
    config.enable_jit_freeze = False
    config.enable_cnn_optimization = False
    
    return config
```

### Clear Cached Compilations

```bash
# Linux/Mac
rm -rf ~/.cache/torch_extensions

# Windows
del /s /q %USERPROFILE%\.cache\torch_extensions

# Docker (mount cache as volume)
docker run -v my_cache:/root/.cache/torch_extensions ...
docker volume rm my_cache  # Clear cache
```

### Profile Compilation

```bash
# Enable debug logging
export LD_SERVER_LOGLEVEL=DEBUG

# Run generation and check logs
cat logs/server.log | grep "Stable"
```

## Best Practices

### Production Deployments

1. **Pre-compile models** during startup with a warm-up request (only for batch/long-running services)
2. **Mount cache volume** to persist compilations across container restarts
3. **Disable CUDA graphs** if serving multiple batch sizes
4. **Enable CUDA graphs** for fixed-resolution APIs with consistent high-volume traffic
5. **Disable Stable-Fast entirely** for single-shot API endpoints (compilation overhead exceeds benefit)

Example warm-up:

```python
# In startup script
def warmup_stable_fast(model, width=768, height=512):
    """Pre-compile model with dummy input."""
    dummy_input = torch.randn(1, 4, height // 8, width // 8, device="cuda")
    dummy_timestep = torch.tensor([999], device="cuda")
    
    with torch.no_grad():
        model(dummy_input, dummy_timestep, c={})
    
    print("Stable-Fast compilation complete")
```

### Development Workflows

1. **Disable Stable-Fast** when experimenting with new models/LoRAs (avoids repeated recompilation)
2. **Enable for final testing** to verify production performance
3. **Clear cache** after upgrading PyTorch/CUDA drivers

## Citation

If you use Stable-Fast in your work:

```bibtex
@misc{stable-fast,
  author = {Cheng Zeyi},
  title = {stable-fast: Fast Inference for Stable Diffusion},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/chengzeyi/stable-fast}
}
```

## Resources

- [Stable-Fast Repository](https://github.com/chengzeyi/stable-fast)
- [Installation Guide](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [CUDA Graphs Guide](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
