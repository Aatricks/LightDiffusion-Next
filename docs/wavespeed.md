# WaveSpeed Caching

## Overview

WaveSpeed is the project's caching-oriented optimization layer for reusing work across denoising steps. In the current codebase, the integrated path is DeepCache for UNet-based models, and the repository also contains groundwork for a Flux-oriented First Block Cache path.

LightDiffusion-Next contains two WaveSpeed-related implementations:

1. **DeepCache** — Integrated for UNet-based models (SD1.5, SDXL)
2. **First Block Cache (FBCache)** — Flux-oriented cache machinery present in the codebase

Both are training-free. DeepCache is the user-facing path today; First Block Cache is codebase groundwork for a more specialized transformer caching path.

## How It Works

### Core Insight

Diffusion models denoise images iteratively over 20-50 steps. Researchers observed that:

- **High-level features** (semantic structure, composition) change slowly across steps
- **Low-level features** (fine details, textures) require frequent updates

WaveSpeed aims to reduce repeated computation across nearby denoising steps by reusing information from earlier steps where practical.

### DeepCache (UNet Models) {#deepcache}

DeepCache is the integrated WaveSpeed path for UNet models.

**Cache step (every N steps):**
1. Run the full denoiser path
2. Store the output for later reuse

**Reuse step (intermediate steps):**
1. Reuse the cached denoiser output
2. Skip the full model recomputation for that step

**Speedup:** ~50-70% time saved per reuse step → 2-3x total speedup with `interval=3`

### First Block Cache (Flux Models)

Flux uses Transformer blocks instead of UNet convolutions. The repository includes a First Block Cache implementation for this architecture family:

```
┌─────────────────────────────────────────┐
│ First Transformer Block (always run)    │ ← Computes initial features
├─────────────────────────────────────────┤
│ Remaining Blocks (cached if similar)    │ ← FBCache caching zone
└─────────────────────────────────────────┘
```

**Cache decision logic:**
1. Run first Transformer block
2. Compare output to previous step's output
3. If difference < threshold: reuse cached remaining blocks
4. If difference ≥ threshold: run all blocks and update cache

In the current project structure, this cache path is implementation groundwork rather than a standard generation toggle like DeepCache.

## DeepCache Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_interval` | int | 3 | Steps between cache updates (higher = faster, lower quality) |
| `cache_depth` | int | 2 | UNet depth for caching (0-12, higher = more aggressive) |
| `start_step` | int | 0 | Timestep to start caching (0-1000) |
| `end_step` | int | 1000 | Timestep to stop caching (0-1000) |

### Streamlit UI

Enable in the **⚡ DeepCache Acceleration** expander:

1. Check **Enable DeepCache**
2. Adjust sliders:
   - **Cache Interval**: 1-10 (default: 3)
   - **Cache Depth**: 0-12 (default: 2)
   - **Start/End Steps**: 0-1000 (default: 0/1000)
3. Generate images — caching applies transparently

### REST API

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a misty forest at twilight",
        "width": 768,
        "height": 512,
        "deepcache_enabled": true,
        "deepcache_interval": 3,
        "deepcache_depth": 2
      }'
```

### Recommended Presets

#### Balanced (Default)
```yaml
cache_interval: 3
cache_depth: 2
start_step: 0
end_step: 1000
```
- **Speedup:** 2-2.3x
- **Quality loss:** Very slight (1-2%)
- **Use case:** Everyday generation

#### Maximum Speed
```yaml
cache_interval: 5
cache_depth: 3
start_step: 0
end_step: 1000
```
- **Speedup:** 2.5-3x
- **Quality loss:** Noticeable (5-7%)
- **Use case:** Rapid prototyping, batch jobs

#### Maximum Quality
```yaml
cache_interval: 2
cache_depth: 1
start_step: 0
end_step: 1000
```
- **Speedup:** 1.5-2x
- **Quality loss:** Minimal (<1%)
- **Use case:** Final renders, client work

#### Partial Caching (Critical Steps Only)
```yaml
cache_interval: 3
cache_depth: 2
start_step: 200
end_step: 800
```
- **Speedup:** 1.8-2.2x
- **Quality loss:** Minimal
- **Use case:** Preserve early structure, late details

## First Block Cache (FBCache) Configuration

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residual_diff_threshold` | float | 0.05 | Max feature difference to trigger cache reuse (0.0-1.0) |

### Usage

First Block Cache is not currently exposed as a standard per-generation toggle. The implementation is available in the codebase for specialized integration work:

```python
# In src/user/pipeline.py
from src.WaveSpeed import fbcache_nodes

# Create cache context
cache_context = fbcache_nodes.create_cache_context()

# Apply caching to a Flux-style model
with fbcache_nodes.cache_context(cache_context):
    patched_model = fbcache_nodes.create_patch_flux_forward_orig(
        flux_model,
        residual_diff_threshold=0.05,  # Lower = stricter caching
    )
    # Generate images...
```

### Tuning Threshold

- **Lower threshold (0.01-0.03)**: Stricter caching, recomputes more often, higher quality
- **Higher threshold (0.05-0.1)**: Looser caching, reuses more often, higher speedup
- **Recommended:** 0.05 (balances quality and speed)

## Performance

### Speedup Guidance

Speedup scales with cache interval and depth:

| Model | Cache Interval | Expected Behavior |
|-------|---------------|-------------------|
| SD1.5 | 2 | Moderate speedup, minimal quality loss |
| SD1.5 | 3 | Good speedup, slight quality loss |
| SD1.5 | 5 | High speedup, noticeable quality loss |
| SDXL | 3 | Good speedup, slight quality loss |
| Flux-style caching paths | implementation-specific | Depends on the integration path |

**Performance varies based on:**
- GPU architecture
- Model size
- Resolution
- Sampler choice
- Number of steps

**Recommendation:** Start with `interval=3` and adjust based on your quality requirements.### VRAM Impact

Caching increases VRAM usage slightly (50-200MB depending on resolution):

| Model | Baseline VRAM | + DeepCache | Increase |
|-------|--------------|-------------|----------|
| SD1.5 (768×512) | 3.2 GB | 3.4 GB | +200 MB |
| SDXL (1024×1024) | 6.8 GB | 7.0 GB | +200 MB |
| Flux (832×1216) | 12.5 GB | 12.6 GB | +100 MB |

## Stacking with Other Optimizations

WaveSpeed is **fully compatible** with SageAttention, SpargeAttn and Stable-Fast:

### DeepCache + SageAttention

```yaml
deepcache_enabled: true
deepcache_interval: 3
# SageAttention auto-detected
```

**Result:** 2.2x (DeepCache) × 1.15 (SageAttention) = **~2.5x total speedup**

### DeepCache + SpargeAttn

```yaml
deepcache_enabled: true
deepcache_interval: 3
# SpargeAttn auto-detected
```

**Result:** Enhanced speedup from caching and sparse attention

### DeepCache + Stable-Fast + SpargeAttn

```yaml
stable_fast: true
deepcache_enabled: true
deepcache_interval: 3
# SpargeAttn auto-detected
```

**Result:** Maximum combined speedup (all optimizations active, batch operations only)

## Compatibility

### DeepCache Compatible With

- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion 2.1
- ✅ SDXL
- ✅ All samplers (Euler, DPM++, etc.)
- ✅ LoRA adapters
- ✅ Textual inversion embeddings
- ✅ HiresFix
- ✅ ADetailer
- ✅ Multi-scale diffusion
- ✅ SageAttention/SpargeAttn
- ✅ Stable-Fast

### DeepCache NOT Compatible With

- ❌ Flux models (use FBCache instead)
- ❌ Img2Img mode (can cause artifacts)

### FBCache Compatible With

- ✅ Flux models
- ✅ SageAttention/SpargeAttn
- ✅ All Flux-compatible features

### FBCache NOT Compatible With

- ❌ SD1.5/SDXL (use DeepCache instead)
- ❌ Stable-Fast (Flux not supported by Stable-Fast)

## Troubleshooting

### No Speedup Observed

**Causes:**
1. DeepCache disabled or not applied to correct model type
2. Cache interval too low (interval=1 provides no caching)
3. Model loaded incorrectly

**Fixes:**
```bash
# Check logs for DeepCache activation
cat logs/server.log | grep -i "deepcache\|cache"

# Verify UI toggle is enabled
# Streamlit: Check "Enable DeepCache" checkbox
# API: Ensure "deepcache_enabled": true in payload

# Try higher interval
deepcache_interval: 3  # Instead of 1 or 2
```

### Quality Degradation

**Symptoms:**
- Blurry details
- Smoothed textures
- Loss of fine patterns

**Causes:**
1. Cache interval too high
2. Cache depth too aggressive
3. Wrong model type (Flux using DeepCache)

**Fixes:**
```yaml
# Reduce cache interval
deepcache_interval: 2  # Down from 5

# Reduce cache depth
deepcache_depth: 1  # Down from 3

# Disable caching for critical phases
deepcache_start_step: 200  # Skip early structure formation
deepcache_end_step: 800    # Skip late detail refinement
```

### Artifacts in Img2Img

**Symptom:** Visible seams, inconsistent styles when using DeepCache with Img2Img.

**Cause:** Img2Img starts from a noisy input image, which violates DeepCache's assumptions about feature consistency.

**Fix:** Disable DeepCache for Img2Img:
```yaml
deepcache_enabled: false  # When img2img_enabled: true
```

### VRAM Increase

**Symptom:** OOM errors after enabling DeepCache.

**Cause:** Cached features consume additional VRAM.

**Fixes:**
1. Reduce batch size
2. Lower resolution
3. Disable other VRAM-heavy features (Stable-Fast CUDA graphs)
4. Use lower cache depth:
   ```yaml
   deepcache_depth: 1  # Minimal caching
   ```

### Flux FBCache Not Working

**Symptom:** No speedup with Flux generation.

**Cause:** FBCache implementation is more subtle — check logs for cache hit rate.

**Debugging:**
```bash
# Enable debug logging
export LD_SERVER_LOGLEVEL=DEBUG

# Check cache statistics
cat logs/server.log | grep "cache"
```

If no cache hits, try adjusting threshold:
```python
# In pipeline.py
residual_diff_threshold=0.1  # Increase from 0.05 for more cache reuse
```

## Quality Comparison

Visual impact of different cache intervals:

| Interval | Speed | Visual Difference |
|----------|-------|-------------------|
| Disabled | Baseline | Baseline (100% quality) |
| 2 | Faster | Virtually identical |
| 3 | Much faster | Very subtle smoothing |
| 5 | Very fast | Noticeable detail loss |
| 7+ | Fastest | Obvious quality degradation |

**Recommendation:** Start with `interval=3` and adjust based on visual results.

## Technical Details

### DeepCache Implementation

Simplified pseudocode:

```python
class DeepCacheWrapper:
    def __init__(self, model, interval, depth):
        self.model = model
        self.interval = interval
        self.cached_output = None
        self.current_step = 0
    
    def forward(self, x, timestep):
        is_cache_step = (self.current_step % self.interval == 0)
        
        if is_cache_step:
            # Run full model, cache output
            output = self.model(x, timestep)
            self.cached_output = output.clone()
        else:
            # Reuse cached output (skip expensive computation)
            output = self.cached_output
        
        self.current_step += 1
        return output
```

Actual implementation in `src/WaveSpeed/deepcache_nodes.py` includes:
- Proper timestep tracking
- Cache invalidation on batch changes
- Error handling and fallback to full forward

### FBCache Residual Comparison

```python
# Compute first block output
first_output = first_transformer_block(hidden_states)

# Compare to previous step
residual = first_output - previous_first_output
residual_norm = residual.abs().mean() / first_output.abs().mean()

if residual_norm < threshold:
    # Feature change is small — reuse cached blocks
    hidden_states = apply_cached_residual(first_output)
else:
    # Feature change is large — recompute all blocks
    hidden_states = run_remaining_blocks(first_output)
    cache_residual(hidden_states)
```

## Best Practices

### For Everyday Use

1. **Enable DeepCache** with default settings (`interval=3`, `depth=2`)
2. **Stack with SageAttention** for 2.5x+ total speedup
3. **Disable for final client renders** if absolute quality is critical

### For Batch Processing

1. **Use aggressive caching** (`interval=5`, `depth=3`)
2. **Pre-generate previews** at high speed, re-render winners at full quality
3. **Disable TAESD previews** to avoid overhead (set `enable_preview=false`)

### For Low VRAM

1. **Use conservative caching** (`interval=2`, `depth=1`)
2. **Avoid stacking** with Stable-Fast CUDA graphs
3. **Monitor VRAM** via `/api/telemetry` endpoint

## Citation

If you use WaveSpeed/DeepCache in your work:

```bibtex
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={CVPR},
  year={2024}
}
```

## Resources

- [DeepCache Paper](https://arxiv.org/abs/2312.00858)
- [DeepCache Repository](https://github.com/horseee/DeepCache)
- [ComfyUI DeepCache Implementation](https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86) (reference for LightDiffusion-Next)
- [First Block Cache Discussion](https://github.com/comfyanonymous/ComfyUI/discussions/3491)
