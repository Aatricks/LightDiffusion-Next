# DeepCache Integration

## Overview

DeepCache is a training-free acceleration technique for diffusion models that can provide **2-3x speedup** with minimal quality loss. It works by reusing high-level features in the U-Net architecture while updating low-level features in a cheap way.

## How It Works

DeepCache exploits the property of U-Net architectures used in Stable Diffusion models:
- **High-level features** (deeper layers) change slowly during the denoising process
- **Low-level features** (shallow layers) need frequent updates for detail

By caching the expensive middle and output blocks and reusing them for several steps, DeepCache significantly reduces computation while maintaining image quality.

## Usage

### Streamlit UI

1. Open the **⚡ DeepCache Acceleration** expander in the sidebar
2. Check **Enable DeepCache**
3. Configure parameters:
   - **Cache Interval** (1-10): Steps between cache updates. Higher = faster but may reduce quality. Default: 3
   - **Cache Depth** (0-12): U-Net depth for caching. Higher = more aggressive caching. Default: 2
   - **Start Step** (0-1000): When to start applying DeepCache. Default: 0
   - **End Step** (0-1000): When to stop applying DeepCache. Default: 1000

### Command Line

```bash
python pipeline.py "your prompt" 512 512 1 1 --deepcache --deepcache-interval 3 --deepcache-depth 2
```

Available flags:
- `--deepcache`: Enable DeepCache
- `--deepcache-interval <int>`: Cache interval (default: 3)
- `--deepcache-depth <int>`: Cache depth (default: 2)
- `--deepcache-start-step <int>`: Start step (default: 0)
- `--deepcache-end-step <int>`: End step (default: 1000)

## Recommended Settings

### Balanced (Default)
- **Cache Interval**: 3
- **Cache Depth**: 2
- Best balance between speed and quality

### Maximum Speed
- **Cache Interval**: 5-7
- **Cache Depth**: 3-4
- 2.5-3x speedup, slight quality reduction

### Maximum Quality
- **Cache Interval**: 2
- **Cache Depth**: 1
- 1.5-2x speedup, minimal quality loss

## Compatibility

### Compatible With:
- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion 2.1
- ✅ SDXL (Stable Diffusion XL)
- ✅ HiRes Fix
- ✅ ADetailer
- ✅ Multi-scale diffusion
- ✅ Most samplers (DPM++, Euler, etc.)

### Not Compatible With:
- ❌ Flux models (different architecture)
- ❌ Img2Img mode (can cause artifacts)

## Performance

Expected speedup on typical hardware:

| Configuration | Speedup | Quality Impact |
|--------------|---------|----------------|
| interval=2, depth=1 | 1.5-2x | Minimal |
| interval=3, depth=2 | 2-2.3x | Very slight |
| interval=5, depth=3 | 2.5-3x | Noticeable |

*Actual speedup may vary based on hardware, resolution, and sampler.*

## Technical Details

DeepCache is implemented as a model wrapper that:
1. Intercepts the U-Net forward pass
2. On cache steps: Runs full forward, stores intermediate features
3. On reuse steps: Skips expensive blocks, reuses cached features
4. Maintains quality by running full forward at regular intervals

The implementation is based on:
- [DeepCache Paper](https://arxiv.org/abs/2312.00858)
- [Official Repository](https://github.com/horseee/DeepCache)
- [ComfyUI Implementation](https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86)

## Troubleshooting

**Issue**: No speedup or errors
- Solution: DeepCache only works with U-Net models (SD1.5, SD2.1, SDXL). Disable for Flux.

**Issue**: Quality degradation
- Solution: Reduce cache interval (2-3) or cache depth (1-2)

**Issue**: Artifacts in images
- Solution: Disable DeepCache for img2img, or adjust start/end steps to skip critical phases

**Issue**: VRAM usage increased
- Solution: DeepCache stores cached features, which may increase VRAM slightly

## Citation

If you use DeepCache in your work, please cite:

```bibtex
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
