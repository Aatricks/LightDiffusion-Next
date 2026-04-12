## 2. AYS (Align Your Steps) Scheduler

### What It Does

Uses optimized timestep distributions that allow **fewer sampling steps** with **same or better quality** compared to uniform schedulers.

### Key Insight

Not all timesteps contribute equally to image formation. AYS pre-computes optimal sigma schedules that focus more steps on critical noise levels.

### Research Background

Based on "Align Your Steps: Optimizing Sampling Schedules in Diffusion Models" (2024)
- https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
- Developed by NVIDIA researchers
- Validated across SD1.5, SDXL, and other models

### Performance

| Model | Normal Scheduler | AYS Scheduler | Quality |
|-------|-----------------|---------------|---------|
| SD1.5 | 20 steps | **10 steps** | Same/Better |
| SDXL  | 20 steps | **10 steps** | Same/Better |
| Flux  | 15 steps | **8 steps** | Same |

### Usage

#### Via UI (Streamlit)

1. Open Settings → Sampling
2. Select scheduler: "AYS (Align Your Steps)"
3. Reduce steps to 10 (SD1.5/SDXL) or 8 (Flux)
4. Generate - same quality, 2x faster!

#### Programmatically

```python
from src.sample import ksampler_util

# Using AYS scheduler
sigmas = ksampler_util.calculate_sigmas(
    model_sampling, 
    scheduler_name="ays",  # or "ays_sd15", "ays_sdxl", "ays_flux"
    steps=10
)
```

### Scheduler Variants

- `"ays"` or `"ays_sd15"` - SD1.5 optimized (default)
- `"ays_sdxl"` - SDXL optimized
- `"ays_flux"` - Flux optimized (experimental)

### Optimal Step Counts

Pre-computed optimal schedules exist for:

**SD1.5**: 4, 6, 8, 10, 12, 15, 20, 25 steps  
**SDXL**: 4, 6, 8, 10, 12, 15, 20 steps  
**Flux**: 4, 8, 10, 15, 20 steps  

Other step counts use interpolation (slightly less optimal but still better than uniform).

### Recommended Settings

#### SD1.5 Quick Generation
```yaml
scheduler: "ays"
steps: 10          # instead of 20
sampler: "euler" or "dpmpp_2m_cfgpp"
cfg: 7.0
```

#### SDXL High Quality
```yaml
scheduler: "ays_sdxl"
steps: 12          # instead of 20-25
sampler: "dpmpp_2m_cfgpp"
cfg: 6.0
```

#### Flux Fast Mode
```yaml
scheduler: "ays_flux"
steps: 8           # instead of 15
sampler: "euler"
cfg: 3.5
```

### Comparison: Uniform vs AYS

**Uniform Distribution (normal scheduler)**:
```
Steps: 0  4  8  12  16  20
Sigmas evenly spaced → wastes compute on low-impact timesteps
```

**AYS Distribution**:
```
Steps: 0  2  5  8  12  17  20
Sigmas concentrated on critical noise levels → better efficiency
```

### Technical Details

AYS schedules are pre-computed using optimization to minimize reconstruction error:

```python
# Example SD1.5 10-step schedule
AYS_SD15_10 = [
    14.6146,  # High noise (early steps - image structure)
    10.4708,
    7.3688,
    4.9651,   # Mid noise (detail formation)
    3.2924,
    2.1391,
    1.3633,   # Low noise (fine details)
    0.8437,
    0.4898,
    0.2279,
    0.0       # Final step
]
```

Compare to uniform schedule:
```python
# Normal scheduler @ 10 steps
NORMAL_10 = [14.6146, 11.3, 8.7, 6.7, 5.1, 3.9, 3.0, 2.3, 1.7, 1.2, 0.0]
# More evenly spaced → less efficient
```

### Troubleshooting

**Q: Images look different with AYS?**  
A: Yes, they will differ slightly (different paths through noise space). Quality should be same or better. Adjust CFG if needed.

**Q: AYS + multiscale?**  
A: Works great together! AYS optimizes step distribution, multiscale optimizes spatial resolution.

**Q: Can I use AYS with euler_ancestral?**  
A: Yes! Works with all samplers (euler, euler_ancestral, dpmpp_2m_cfgpp, dpmpp_sde_cfgpp, etc.)

**Q: How to verify it's active?**  
A: Check logs for "Using AYS optimal schedule" message.

### References

- Original paper: https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
- Implementation: `src/sample/ays_scheduler.py`
- Integration: `src/sample/ksampler_util.py`
