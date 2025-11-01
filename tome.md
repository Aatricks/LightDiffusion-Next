# Token Merging (ToMe)

## Overview

Token Merging (ToMe) is a **performance optimization** that accelerates diffusion models by intelligently merging similar tokens in the attention mechanism. By identifying and combining redundant computations, ToMe achieves **20-60% speedup** with minimal quality impact.

Unlike feature caching (DeepCache, WaveSpeed), ToMe reduces the computational graph itself — fewer tokens means fewer attention operations, less memory bandwidth, and faster generation.

This is a **training-free**, **drop-in optimization** that works with all Stable Diffusion models (SD1.5, SDXL) and can be combined with other speedup techniques.

## How It Works

### The Token Redundancy Problem

Diffusion models process images as sequences of tokens (patches):

```
Input Image (512×512) → Tokenize → 4096 tokens (64×64 grid of 8×8 patches)
```

At each attention layer, **every token attends to every other token**:

$$
\text{Attention Cost} = O(N^2 \cdot D)
$$

Where:
- $N$ = number of tokens (e.g., 4096 for 512×512)
- $D$ = embedding dimension (e.g., 768 or 1024)

**Key insight:** Many tokens are highly similar (e.g., sky regions, uniform backgrounds, smooth gradients). Computing attention between nearly-identical tokens is redundant.

### The ToMe Solution

Token Merging reduces redundancy through **bipartite matching**:

```
Step 1: Split tokens into two sets
┌─────────────────────┬─────────────────────┐
│ Destination Set (dst)│ Source Set (src)    │
│ [Token 1, 3, 5, ...] │ [Token 2, 4, 6, ...] │
└─────────────────────┴─────────────────────┘

Step 2: Compute similarity (cosine distance)
   dst[0] ↔ src[0]: 0.92  (highly similar!)
   dst[0] ↔ src[1]: 0.34
   dst[0] ↔ src[2]: 0.18
   ...

Step 3: Merge most similar pairs
   merged_token[0] = (dst[0] + src[0]) / 2
   
Step 4: Continue with fewer tokens
   4096 tokens → 2048 tokens (50% merge ratio)
   Attention cost reduced by ~4x
```

This happens **per attention layer**, with merge ratio dynamically adjusting based on layer depth.

## Configuration

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `tome_enabled` | bool | `False` | - | Enable Token Merging |
| `tome_ratio` | float | `0.5` | 0.0-0.9 | Percentage of tokens to merge (higher = faster, lower quality) |
| `tome_max_downsample` | int | `1` | 1, 2, 4, 8 | Apply ToMe to layers with downsampling ≤ this value |

### Choosing `tome_max_downsample`

Controls which UNet layers apply ToMe:

| Value | Layers Affected | Speed vs Quality |
|-------|----------------|------------------|
| **1** | Only full-resolution layers (4/15) | Conservative, minimal quality impact |
| **2** | Half-resolution layers (8/15) | Balanced (recommended) |
| **4** | Quarter-resolution layers (12/15) | Aggressive |
| **8** | All layers (15/15) | Maximum speedup, noticeable quality loss |

**Recommendation:** Start with `max_downsample=1`. Only increase if you need more speedup and can tolerate quality reduction.

## Usage

### Streamlit UI

Enable in the **🔀 Token Merging (ToMe)** expander:

1. Check **Enable Token Merging**
2. Select a preset:
   - **Conservative** — 30% merge, max_downsample=2 (minimal impact)
   - **Balanced** — 50% merge, max_downsample=1 (recommended)
   - **Aggressive** — 70% merge, max_downsample=1 (maximum speed)
   - **Custom** — Manual slider control
3. Generate images — console confirms activation

**Visual feedback:**
```
✓ Token Merging ACTIVE: 50% merge ratio, max_downsample=1
```

### REST API

Include in your generation request:

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a cyberpunk cityscape at night, neon lights",
        "width": 1024,
        "height": 512,
        "steps": 25,
        "tome_enabled": true,
        "tome_ratio": 0.5,
        "tome_max_downsample": 1
      }'
```

### Python API

```python
from src.user.pipeline import pipeline

pipeline(
    prompt="a detailed fantasy castle on a cliff",
    w=768,
    h=1024,
    steps=30,
    sampler="dpmpp_sde_cfgpp",
    scheduler="ays",
    tome_enabled=True,
    tome_ratio=0.5,
    tome_max_downsample=1,
    number=4  # Generate multiple images faster
)
```

## Troubleshooting

### "No speedup detected"

**Possible causes:**
1. **tomesd not installed** — Install with `pip install tomesd`
2. **Other bottlenecks** — Enable only ToMe for isolated testing
3. **Very low resolution** — ToMe benefits are minimal below 512px

**Solutions:**
```bash
# Check installation
python -c "import tomesd; print('ToMe available')"

# Test in isolation at 1024×512 (ideal resolution for ToMe)
python quick_tome_test.py
```

### "Images look blurry or soft"

**Cause:** `tome_ratio` too high (>0.6) or `max_downsample` too aggressive (>2).

**Solutions:**
- Reduce `tome_ratio` to 0.4-0.5
- Lower `max_downsample` to 1
- Increase `steps` to 30-35 for better convergence
- Disable ToMe for final high-quality renders

### "Minimal speedup despite 70% merge"

**Cause:** Other optimizations (DeepCache, Multi-Scale) already bottlenecked elsewhere (VAE decode, sampling overhead).

**Solutions:**
- Profile with isolated tests (disable all other optimizations)
- Ensure GPU isn't memory-bound (reduce batch size)
- Check system monitoring for CPU/disk bottlenecks

### "Model fails to load / tomesd errors"

**Cause:** Outdated tomesd version or incompatible model architecture.

**Solutions:**
```bash
# Update tomesd
pip install --upgrade tomesd

# Check compatibility (ToMe only works with UNet-based models)
# Flux/Transformer models require different ToMe variant (not yet supported)
```

## Technical Details

### Implementation

ToMe is applied via the `ModelPatcher` class (`src/Model/ModelPatcher.py`):

```python
def apply_tome(self, ratio: float = 0.5, max_downsample: int = 1) -> bool:
    """Apply Token Merging to the diffusion model."""
    # Remove any existing patch (handles cached models)
    try:
        tomesd.remove_patch(self)
    except:
        pass
    
    # Apply ToMe patch
    tomesd.apply_patch(
        self,  # ModelPatcher with .model.diffusion_model structure
        ratio=ratio,
        max_downsample=max_downsample
    )
    self.tome_enabled = True
    return True
```

**Cache handling:** ToMe patches are removed after each generation and re-applied as needed, ensuring correct behavior with model caching.

### Bipartite Matching Algorithm

ToMe uses **proportional attention-based matching**:

1. **Partition tokens:**
   $$
   T_{\text{dst}}, T_{\text{src}} = \text{partition}(T, \text{stride}=(2,2))
   $$

2. **Compute similarity matrix:**
   $$
   S_{ij} = \frac{T_{\text{dst}}[i] \cdot T_{\text{src}}[j]}{||T_{\text{dst}}[i]|| \cdot ||T_{\text{src}}[j]||}
   $$

3. **Find top-k matches:**
   $$
   k = \lfloor \text{ratio} \times |T_{\text{src}}| \rfloor
   $$

4. **Merge tokens:**
   $$
   T'[i] = \frac{T_{\text{dst}}[i] + T_{\text{src}}[\text{match}(i)]}{2}
   $$

## Compatibility

| Feature | Compatible? | Notes |
|---------|-------------|-------|
| **SD1.5 models** | ✓ | Full support, tested extensively |
| **SDXL models** | ✓ | Full support, larger speedup |
| **Flux models** | ✗ | UNet-specific, Transformer variant TBD |
| **All samplers** | ✓ | ToMe patches attention, agnostic to sampler |
| **CFG-Free** | ✓ | No interaction, both apply independently |
| **DeepCache** | ✓ | Excellent combination, speedups multiply |
| **Multi-Scale** | ✓ | Compatible, benefits stack |
| **HiRes Fix** | ✓ | Applied to all upscaling passes |
| **ADetailer** | ✓ | Applied to detail-enhancement passes |
| **Stable-Fast** | ✓ | Can combine for maximum speedup |

## Limitations

1. **UNet-only:** Transformer architectures (Flux) use different attention patterns — dedicated Transformer-ToMe needed
2. **Detail sensitivity:** High-frequency textures (fabric weave, individual hairs) see most quality impact
3. **Diminishing returns:** Beyond 60% merge, quality degrades faster than speed improves
4. **One-time patch:** Doesn't adapt merge ratio dynamically during generation

## Related Optimizations

- **[DeepCache](wavespeed.md#deepcache)**: Feature caching — complements ToMe, speedups multiply (~2.8x combined)
- **[Multi-Scale Diffusion](optimizations.md#multi-scale)**: Resolution-based optimization — also reduces token count
- **[Stable-Fast](stablefast.md)**: Compilation-based speedup — can combine for maximum performance

## References & Further Reading

- **Original Paper:** [Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604) (Bolya & Hoffman, 2023)
- **tomesd Library:** https://github.com/dbolya/tomesd
- **ToMe for Vision Transformers:** https://github.com/facebookresearch/ToMe
