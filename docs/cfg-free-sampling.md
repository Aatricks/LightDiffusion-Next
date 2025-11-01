# CFG-Free Sampling

## Overview

CFG-Free Sampling is a **quality optimization technique** that gradually reduces Classifier-Free Guidance (CFG) to zero during the final stages of image generation. This approach leverages the observation that high CFG strength is most beneficial early in the denoising process, while later steps benefit from reduced guidance for more natural, detailed outputs.

By intelligently transitioning from high-guidance to low-guidance sampling, CFG-Free achieves:

- **Improved fine detail** and texture quality
- **More natural color saturation** and tonal balance
- **Reduced artifacts** from over-guidance (halos, oversaturation, unnatural sharpness)
- **Better prompt adherence** while maintaining photorealism

This is a **training-free** technique that works with any sampler and can be combined with other optimizations.

## How It Works

### The CFG Problem

Classifier-Free Guidance strengthens prompt adherence by amplifying the difference between conditional and unconditional predictions:

$$
\text{output} = \text{uncond\_pred} + \text{cfg\_scale} \times (\text{cond\_pred} - \text{uncond\_pred})
$$

**Benefits of high CFG (7-12):**
- Strong prompt following
- Clear compositional structure
- Distinct subjects and backgrounds

**Drawbacks of high CFG throughout generation:**
- Over-sharpened edges ("halo effect")
- Oversaturated colors
- Loss of fine detail and texture
- Unnatural, "CG-like" appearance
- Potential anatomical distortions

### The CFG-Free Solution

Research shows that CFG importance varies by denoising stage:

```
┌─────────────────────────────────────────────────────────┐
│ Early Steps (0-70%)                                     │
│ High CFG is crucial:                                    │
│   • Establishes composition                             │
│   • Defines subject placement                           │
│   • Interprets prompt semantics                         │
│                                                         │
│ CFG = 7.0 (user-configured)                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Late Steps (70-100%)                                    │
│ High CFG becomes detrimental:                           │
│   • Composition already locked in                       │
│   • Fine details being refined                          │
│   • Oversaturation and artifacts emerge                 │
│                                                         │
│ CFG = 7.0 → 0.0 (linear reduction)                      │
└─────────────────────────────────────────────────────────┘
```

CFG-Free gradually reduces guidance from your configured value (e.g., 7.0) to 0.0 over the final portion of generation. This preserves strong prompt adherence while allowing the model to naturally refine details without over-guidance.

## Configuration

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `cfg_free_enabled` | bool | `False` | - | Enable CFG-Free sampling |
| `cfg_free_start_percent` | float | `70.0` | 0-100 | Percentage of steps at which to start reducing CFG |

### How to Choose `cfg_free_start_percent`

The optimal starting point depends on your aesthetic goals:

| Start % | Behavior | Best For |
|---------|----------|----------|
| **60-65%** | Aggressive reduction, maximum detail preservation | Photorealistic portraits, product photography, architectural renders |
| **70-75%** | Balanced approach (recommended) | General purpose, landscapes, character art, concept art |
| **80-85%** | Conservative reduction, maintains stronger guidance | Abstract art, heavily stylized content, complex compositions |
| **90%+** | Minimal effect, mostly for testing | Debugging, comparing with full-CFG baseline |

**Rule of thumb:** Start with 70% for most use cases. If images appear oversaturated or have unnatural sharpness, lower it to 65%. If prompt adherence weakens, raise it to 75-80%.

## Usage

### Streamlit UI

Enable in the **🎨 CFG-Free Sampling** expander:

1. Check **Enable CFG-Free Sampling**
2. Adjust the **Start Percentage** slider (0-100%, default: 70%)
3. The info panel shows exactly when CFG reduction begins
4. Generate images — you'll see console logging confirming activation

**Visual feedback:**
```
✓ CFG-Free sampling ACTIVE: CFG will gradually reduce to 0 starting at 70% of steps
```

### REST API

Include in your generation request:

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a portrait of a woman with flowing hair, soft lighting",
        "negative_prompt": "blurry, low quality",
        "width": 768,
        "height": 1024,
        "steps": 25,
        "cfg_scale": 7.5,
        "cfg_free_enabled": true,
        "cfg_free_start_percent": 70.0
      }'
```

### Python API

```python
from src.user.pipeline import pipeline

pipeline(
    prompt="a serene mountain landscape at sunset",
    negative_prompt="blurry, distorted",
    w=1024,
    h=768,
    steps=30,
    sampler="dpmpp_sde_cfgpp",
    scheduler="ays",
    cfg_free_enabled=True,
    cfg_free_start_percent=70.0,
    number=1
)
```

## Quality Impact Analysis

### Visual Improvements

CFG-Free sampling produces subtle but meaningful quality improvements:

**Before (Standard CFG=7.5):**
- Sharper edges, sometimes with halos
- More saturated colors (can appear "painted")
- Higher contrast, more dramatic lighting
- Occasionally oversimplified textures

**After (CFG-Free from 70%):**
- Softer, more natural edge transitions
- Improved color accuracy and tonal range
- Better fine detail in hair, fabric, skin textures
- More photorealistic lighting and shadow falloff
- Reduced artifacts around high-contrast boundaries

**Key insight:** Prompt adherence is determined in the first 60-70% of steps. Reducing CFG afterward doesn't weaken composition, it enhances natural detail refinement.

## Troubleshooting

### "Images look washed out or less vibrant"

**Cause:** CFG-Free starting too early (e.g., 50-60%) can over-reduce guidance.

**Solutions:**
- Increase `cfg_free_start_percent` to 70-75%
- Slightly increase base `cfg_scale` to 8.0-8.5
- Use a different sampler (try `dpmpp_sde_cfgpp` or `dpmpp_2m_cfgpp`)

### "No visible difference from standard CFG"

**Cause:** Differences are subtle and may be masked by:
- Very simple prompts (single subject, plain background)
- Low resolution (<512px in any dimension)
- Aggressive other optimizations obscuring quality gains

**Solutions:**
- Test with complex prompts (portraits, detailed scenes)
- Use higher resolutions (768px+ recommended)
- Generate comparison images side-by-side with CFG-Free on/off
- Try lower `cfg_free_start_percent` (60-65%) for more noticeable effect

### "Prompt adherence weakened"

**Cause:** CFG-Free starting too early for your particular prompt complexity.

**Solutions:**
- Increase `cfg_free_start_percent` to 75-80%
- Use stronger base `cfg_scale` (8.0-9.0)
- Increase step count to 30-35 for better convergence

## Technical Details

### Implementation

CFG-Free is implemented in the `CFGGuider` class (`src/sample/CFG.py`):

```python
def _update_cfg_for_sigma(self, sigma):
    """Update CFG value based on current sigma and CFG-free parameters."""
    if not self.cfg_free_enabled:
        return
    
    # Find current step position in schedule
    current_step = find_closest_sigma_index(sigma, self.sigmas)
    total_steps = len(self.sigmas) - 1
    progress_percent = (current_step / total_steps) * 100.0
    
    if progress_percent >= self.cfg_free_start_percent:
        # Linear interpolation from original CFG to 0
        cfg_free_progress = (
            (progress_percent - self.cfg_free_start_percent) /
            (100.0 - self.cfg_free_start_percent)
        )
        self.cfg = self.original_cfg * (1.0 - cfg_free_progress)
        self.cfg = max(0.0, self.cfg)  # Clamp to [0, original_cfg]
```

**Schedule visualization:**

```
CFG Scaling Over Time (start_percent=70%, original_cfg=7.5)

Step:  0    5    10   15   20   25   30   35   40   45   50
       ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
CFG:   7.5  7.5  7.5  7.5  7.5  7.5  7.5  5.6  3.8  1.9  0.0
       │■■■■■■■■■■■■■■■■■■■■■■■■│▓▓▓▓▓▓▓▓▓▓│░░░░│    │    │
       └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
       ←──── Full CFG ────────→←─── Gradual Reduction ───→
```

### Mathematical Formulation

Standard CFG at every step:

$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \text{cfg\_scale} \times (\mathbf{cond} - \mathbf{uncond})
$$

CFG-Free with schedule:

$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \text{cfg}(t) \times (\mathbf{cond} - \mathbf{uncond})
$$

Where:

$$
\text{cfg}(t) = \begin{cases}
\text{cfg\_scale} & \text{if } t < t_{\text{start}} \\
\text{cfg\_scale} \times \left(1 - \frac{t - t_{\text{start}}}{t_{\text{total}} - t_{\text{start}}}\right) & \text{if } t \geq t_{\text{start}}
\end{cases}
$$

## Related Optimizations

- **[CFG++ Samplers](optimizations.md#cfg-samplers)**: Advanced CFG implementation with momentum and multi-scale — CFG-Free complements these
- **[Multi-Scale Diffusion](optimizations.md#multi-scale)**: Resolution-based optimization — works independently of CFG-Free
- **[DeepCache](wavespeed.md#deepcache)**: Feature caching for speedup — no quality interaction with CFG-Free

## References & Further Reading

- Original research: CFG-Free sampling builds on insights from [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
- Implementation inspired by community experiments with dynamic CFG schedules
- Mathematical framework adapted from diffusion model literature
