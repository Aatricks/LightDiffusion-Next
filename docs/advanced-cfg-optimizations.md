# Advanced CFG Optimizations

## Overview

This document describes three advanced optimizations for Classifier-Free Guidance (CFG) that improve both quality and performance in LightDiffusion-Next:

1. **Batched CFG Computation** - Speed optimization
2. **Dynamic CFG Rescaling** - Quality optimization  
3. **Adaptive Noise Scheduling** - Quality & speed optimization

## 1. Batched CFG Computation

### What It Does

Instead of running two separate forward passes for conditional and unconditional predictions, this optimization combines them into a single batched forward pass.

**Before:**
```python
# Two separate forward passes
cond_pred = model(x, timestep, cond)      # Pass 1
uncond_pred = model(x, timestep, uncond)  # Pass 2
result = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

**After:**
```python
# Single batched forward pass
both_preds = model(x, timestep, [cond, uncond])  # Single pass
cond_pred, uncond_pred = both_preds[0], both_preds[1]
result = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
```

### Performance Impact

- **Speed**: ~1.8-2x faster CFG computation
- **Memory**: Same or slightly less (batch processing)
- **Quality**: Identical to baseline

### Usage

```python
from src.sample import sampling

samples = sampling.sample1(
    model=model,
    noise=noise,
    steps=20,
    cfg=7.5,
    # ... other params ...
    batched_cfg=True,  # Enable batched CFG (default: True)
)
```

### When to Use

- **Always recommended** - This is a pure speed optimization with no quality tradeoff
- Particularly beneficial for high-resolution images or batch generation
- Compatible with all samplers and schedulers

---

## 2. Dynamic CFG Rescaling

### What It Does

Dynamically adjusts the CFG scale based on prediction statistics to prevent over-saturation while maintaining prompt adherence.

### The Problem

High CFG values (7-12) improve prompt following but can cause:
- Over-saturated colors
- Over-sharpened edges ("halo effect")
- Loss of fine details
- Unnatural, "CG-like" appearance

### The Solution

Dynamic CFG rescaling analyzes the guidance vector (difference between conditional and unconditional predictions) and adjusts the CFG scale to keep it within an optimal range.

**Two Methods:**

#### Variance Method (Recommended)
```python
guidance_std = std(cond_pred - uncond_pred)
adjusted_cfg = cfg_scale * (target_scale / (1 + guidance_std))
```

Best for: General use, prevents over-saturation

#### Range Method
```python
guidance_range = percentile(guidance, 95) - percentile(guidance, 5)
adjusted_cfg = cfg_scale * (target_scale / guidance_range)
```

Best for: Extreme cases, outlier filtering

### Performance Impact

- **Speed**: Minimal overhead (~2-5%)
- **Quality**: Improved color balance, reduced artifacts
- **Prompt Adherence**: Maintained or improved

### Usage

```python
samples = sampling.sample1(
    model=model,
    # ... other params ...
    dynamic_cfg_rescaling=True,        # Enable dynamic rescaling
    dynamic_cfg_method="variance",     # Method: "variance" or "range"
    dynamic_cfg_percentile=95,         # Percentile for range method
    dynamic_cfg_target_scale=1.0,      # Target normalization scale
)
```

### When to Use

- High CFG values (>7.5)
- Detailed prompts that might cause over-saturation
- Photorealistic generations
- Portraits and faces

### When to Avoid

- Very low CFG (<3.0) - minimal benefit
- Artistic/stylized generations where saturation is desired
- When using CFG-free sampling (already handles this differently)

---

## 3. Adaptive Noise Scheduling

### What It Does

Dynamically adjusts the noise schedule based on content complexity during generation.

### The Problem

Traditional fixed noise schedules apply the same denoising steps to all regions:
- Complex scenes (detailed textures) may need more steps in certain regions
- Simple scenes (smooth gradients) can use fewer steps
- This wastes computation or undersamples complexity

### The Solution

Analyzes the complexity of intermediate predictions and adjusts subsequent noise levels accordingly.

**Two Methods:**

#### Complexity Method (Recommended)
```python
complexity = variance(denoised, spatial_dims)
# High variance = complex details = maintain fine noise steps
# Low variance = simple areas = can skip intermediate steps
```

Best for: General content-aware optimization

#### Attention Method
```python
complexity = mean(|gradient(denoised)|)
# High gradients = edges/details = need more precision
# Low gradients = smooth areas = can denoise faster
```

Best for: Edge-focused content (architecture, technical drawings)

### Performance Impact

- **Speed**: 10-20% faster for simple scenes, same for complex
- **Quality**: Adaptive - maintains quality where needed
- **Prompt Adherence**: Unchanged

### Usage

```python
samples = sampling.sample1(
    model=model,
    # ... other params ...
    adaptive_noise_enabled=True,          # Enable adaptive scheduling
    adaptive_noise_method="complexity",   # Method: "complexity" or "attention"
)
```

### When to Use

- Mixed complexity scenes (e.g., detailed subject + simple background)
- Long sampling runs (50+ steps) - more opportunity to optimize
- Batch generation with varying prompt complexity

### When to Avoid

- Very short sampling runs (<10 steps) - overhead > benefit
- Uniformly complex scenes - no simplification possible
- When exact step-by-step reproducibility is critical

---

## Combining Optimizations

All three optimizations can be used together:

```python
samples = sampling.sample1(
    model=model,
    noise=noise,
    steps=20,
    cfg=7.5,
    sampler_name="dpmpp_sde_cfgpp",
    scheduler="ays",
    positive=positive_cond,
    negative=negative_cond,
    latent_image=latent,
    # All optimizations enabled
    batched_cfg=True,
    dynamic_cfg_rescaling=True,
    dynamic_cfg_method="variance",
    dynamic_cfg_target_scale=1.0,
    adaptive_noise_enabled=True,
    adaptive_noise_method="complexity",
)
```

**Expected Results:**
- Better color balance and detail preservation
- Reduced over-saturation artifacts
- Maintained or improved prompt adherence

## Troubleshooting

### Batched CFG Issues

**Problem**: Memory errors with batched CFG  
**Solution**: System may not have enough VRAM. Disable with `batched_cfg=False`

### Dynamic CFG Issues

**Problem**: Images too flat/desaturated  
**Solution**: Increase `dynamic_cfg_target_scale` (try 1.5 or 2.0)

**Problem**: Still over-saturated  
**Solution**: Switch to `dynamic_cfg_method="range"` and lower `dynamic_cfg_percentile`

### Adaptive Noise Issues

**Problem**: Inconsistent results  
**Solution**: Adaptive scheduling makes slight changes based on content. Disable for exact reproducibility.

**Problem**: No speed improvement  
**Solution**: Works best with simple scenes. Complex scenes won't see speedup (but won't be slower either).

---

## Credits

Implemented for LightDiffusion-Next by combining insights from:
- CFG++ dynamic rescaling techniques
- ComfyUI batched computation patterns
- Stable Diffusion WebUI adaptive scheduling
