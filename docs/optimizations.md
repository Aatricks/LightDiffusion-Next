# Performance Optimizations

LightDiffusion-Next achieves its industry-leading inference speed through a layered stack of training-free optimizations that can be selectively enabled based on your hardware and quality requirements. This page provides an overview of each acceleration technique and links to detailed guides.

For a detailed source-based report on what is implemented today, including server-side throughput optimizations and practical implementation notes, see the [Implemented Optimizations Report](implemented-optimizations-report.md).

## Optimization Stack Overview

The pipeline orchestrates six primary acceleration paths:

| Technique | Type | Speedup | Quality Impact | Requirements |
|-----------|------|---------|----------------|---------------|
| [AYS Scheduler](#ays-scheduler) | Sampling schedule | ~2x | None/Better | All models |
| [Prompt Caching](#prompt-caching) | Embedding cache | 5-15% | None | All models |
| [SageAttention](#sageattention--spargeattn) | Attention kernel | Moderate | None | All CUDA GPUs |
| [SpargeAttn](#sageattention--spargeattn) | Sparse attention | Significant | Minimal | Compute 8.0-9.0 |
| [Stable-Fast](#stable-fast) | Graph compilation | Significant* | None | >8GB VRAM, batch jobs |
| [WaveSpeed](#wavespeed-caching) | Feature caching | High | Tunable | All models |

*Speedup depends heavily on batch size and generation count

These optimizations **work together** — enabling multiple techniques simultaneously can provide substantial cumulative speedup with tunable quality trade-offs.

## Quick Comparison

### AYS Scheduler

**What it does:** Uses research-backed optimal timestep distributions that allow equivalent quality in approximately half the steps. Instead of uniform sigma spacing, AYS concentrates samples on noise levels that contribute most to image formation.

**When to use:**
- Always recommended for SD1.5, SDXL, and Flux models
- Txt2Img generation
- Production workflows where speed matters
- Any scenario where you'd normally use 20+ steps

**Trade-offs:** Images will differ slightly from standard schedulers (different sampling path), but quality is equivalent or better. Not ideal when exact reproduction of old results is required.

[→ Full AYS Scheduler guide](ays-scheduler.md)

---

### Prompt Caching

**What it does:** Caches CLIP text embeddings for prompts that have been encoded before. When generating multiple images with the same or similar prompts, embeddings are retrieved from cache instead of being recomputed.

**When to use:**
- Batch generation with same prompt
- Testing different seeds or settings
- Iterative prompt refinement
- Any workflow with repeated prompts

**Trade-offs:** None — minimal memory overhead (~50-200MB), negligible CPU cost, automatically enabled by default.

[→ Full Prompt Caching guide](prompt-caching.md)

---

### SageAttention & SpargeAttn {#sageattention--spargeattn}

**What it does:** Replaces PyTorch's default scaled dot-product attention with highly optimized CUDA kernels. SageAttention uses INT8 quantization for key/value tensors while maintaining FP16 query precision. SpargeAttn extends this with dynamic sparsity pruning, skipping redundant attention computations.

**When to use:**
- Always enable SageAttention if available (no quality loss, pure speed gain)
- SpargeAttn for maximum speed on supported hardware (RTX 30xx/40xx, A100, H100)
- Both work seamlessly with all samplers, LoRAs and post-processing stages

**Trade-offs:** None for SageAttention. SpargeAttn may introduce subtle texture variations at very high sparsity thresholds (default is conservative).

[→ Full SageAttention/SpargeAttn guide](sageattention.md)

---

### CFG Samplers {#cfg-samplers}

CFG++ Samplers are advanced sampling algorithms that incorporate Classifier-Free Guidance directly into the sampling process, providing better quality and stability compared to standard CFG.

---

### Multi-Scale Diffusion {#multi-scale}

Multi-Scale Diffusion optimizes performance by processing images at multiple resolutions during generation, reducing computation for high-resolution areas.

**When to use:**
- High-resolution generation (>1024px)
- When memory is limited
- For faster previews

**Trade-offs:** May reduce detail in fine areas.

**Note:** In most cases, Multi-Scale Diffusion in quality mode gives better results than standard diffusion while giving a small speedup (this is explained by the upsampling process).

---

### Stable-Fast

**What it does:** JIT-compiles the UNet diffusion model into optimized TorchScript with optional CUDA graphs. The first forward pass traces execution, caches kernel launches and fuses operators for reduced overhead.

**When to use:**
- **Systems with >8GB VRAM** (preferably 12GB+)
- Batch jobs or workflows generating 50+ images with identical settings
- Long-running operations where 30-60s compilation amortizes over time
- Fixed resolutions and batch sizes

**When NOT to use:**
- Normal 20-step single image generation (compilation overhead > speedup gains)
- Systems with <8GB VRAM
- Flux workflows (different architecture)
- Quick prototyping or frequent model/resolution changes

**Trade-offs:** Compilation time on first run (30-60s), VRAM overhead (~500MB), reduced flexibility for dynamic shapes.

[→ Full Stable-Fast guide](stablefast.md)

---

### WaveSpeed Caching

**What it does:** Exploits temporal redundancy in diffusion processes by reusing work across denoising steps. In the current project stack this primarily means DeepCache on supported UNet models, with additional Flux-oriented cache groundwork present in the codebase.

1. **DeepCache** — Reuses prior denoiser outputs on selected steps in UNet models (SD1.5, SDXL)
2. **First Block Cache (FBCache)** — Flux-oriented cache machinery available for specialized integration work

**When to use:**
- Any workflow where you can tolerate slight smoothing in exchange for 2-3x speedup
- Combine with conservative cache intervals (2-3) for minimal quality loss
- Works alongside SageAttention and Stable-Fast

**Trade-offs:** Reduced fine detail if interval is too high, slight VRAM increase for cached tensors.

[→ Full WaveSpeed guide](wavespeed.md)

---

## Priority & Fallback System

LightDiffusion-Next automatically selects the best available attention backend at runtime:

```
SpargeAttn > SageAttention > xformers > PyTorch SDPA
```

If a kernel fails (e.g., unsupported head dimension), the system gracefully falls back to the next option. You can force PyTorch SDPA by setting `LD_DISABLE_SAGE_ATTENTION=1` for debugging.

Stable-Fast and WaveSpeed are opt-in toggles controlled via the UI or REST API.

## Recommended Configurations

### Maximum Speed - Batch Jobs (SD1.5, >8GB VRAM, 50+ images)
```yaml
stable_fast: true  # Only for batch operations
sageattention: auto  # or spargeattn if available
deepCache:
  enabled: true
  interval: 3
  depth: 2
```
**Expected:** Maximum speedup for batch operations, some quality loss
**Note:** Disable stable_fast for single 20-step generations

### Balanced - Quick Generation (SD1.5, any VRAM)
```yaml
scheduler: ays  # NEW: Use AYS for 2x speedup
steps: 10  # Reduced from 20 (same quality with AYS)
stable_fast: false  # Disabled for normal generations
sageattention: auto
prompt_cache_enabled: true  # Enabled by default
deepcache:
  enabled: true
  interval: 2
  depth: 1
```
**Expected:** ~2-3x speedup with minimal quality loss
**Note:** AYS scheduler provides the main speedup; enable stable_fast only for batch jobs (50+ images)

### Quality-First (Flux)
```yaml
scheduler: ays_flux  # NEW: Optimized for Flux models
steps: 10  # Reduced from 15 (same quality with AYS)
stable_fast: false  # not supported
sageattention: auto
prompt_cache_enabled: true
deepcache:
  enabled: true
  interval: 2
```
**Expected:** ~2x speedup with minimal quality impact

### Production API - High Volume (>8GB VRAM)
```yaml
stable_fast: true  # Only for sustained high-volume APIs
sageattention: auto
deepCache:
  enabled: false  # avoid variability across batch sizes
keep_models_loaded: true
```
**Expected:** Consistent latency for repeated identical requests
**Note:** For low-volume or single-shot APIs, use `stable_fast: false`

## Hardware-Specific Tips

### RTX 30xx / 40xx (Ampere/Ada)
- Enable SpargeAttn for best results
- Stable-Fast only for batch jobs (disable for quick 20-step generations)
- Stable-Fast + SpargeAttn + DeepCache stacks well for long operations
- Watch VRAM — Stable-Fast graphs consume ~500MB

### RTX 50xx (Blackwell)
- SageAttention only (SpargeAttn support pending)
- Stable-Fast works but recompiles for new CUDA arch
- DeepCache is your best additional speedup

### A100 / H100 (Datacenter)
- SpargeAttn + Stable-Fast + aggressive WaveSpeed
- Prefer larger batch sizes to amortize kernel overhead
- Use CUDA graphs (`enable_cuda_graph=True` in Stable-Fast config)

### Low VRAM (<8GB)
- **Always disable Stable-Fast** (requires >8GB VRAM)
- Use SageAttention (minimal overhead)
- Enable DeepCache with conservative intervals
- Set `vae_on_cpu=True` for HiRes workflows

## Debugging & Profiling

Check which optimizations are active:

```bash
# View startup logs
cat logs/server.log | grep -i "using\|enabled"

# Sample output:
# Using SpargeAttn (Sparse + SageAttention) cross attention
# Using SpargeAttn (Sparse + SageAttention) in VAE
# Stable-Fast compilation enabled
# DeepCache active: interval=3, depth=2
```

Monitor telemetry:

```bash
curl http://localhost:7861/api/telemetry | jq '.vram_usage_mb, .average_latency_ms'
```

Disable individual optimizations to isolate issues:

```bash
export LD_DISABLE_SAGE_ATTENTION=1      # Forces PyTorch SDPA
export LD_DISABLE_STABLE_FAST=1         # Skips compilation
export LD_DISABLE_WAVESPEED=1           # Disables all caching
```

## Further Reading
- [AYS Scheduler Deep Dive](ays-scheduler.md) — Theory, implementation, quality tuning
- [Prompt Caching Deep Dive](prompt-caching.md) — Implementation details, cache management, performance impact
- [SageAttention & SpargeAttn Deep Dive](sageattention.md) — Installation, technical details, head dimension handling
- [Stable-Fast Compilation Guide](stablefast.md) — Configuration, CUDA graphs, troubleshooting
- [WaveSpeed Caching Strategies](wavespeed.md) — DeepCache vs FBCache, tuning parameters, compatibility matrix
- [Performance Tuning](quirks.md) — VRAM management, slow first runs, recompilation fixes

---

Armed with this overview, dive into the technique-specific guides or experiment directly in the UI to find your optimal speed/quality balance.
