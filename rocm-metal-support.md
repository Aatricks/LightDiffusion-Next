# ROCm and Metal/MPS Support

LightDiffusion-Next includes comprehensive support for AMD GPUs with ROCm and Apple Silicon Macs with Metal Performance Shaders (MPS). This guide covers the platform-specific considerations and optimizations available for non-NVIDIA hardware.

## ROCm Support (AMD GPUs)

### Overview

ROCm (Radeon Open Compute) is AMD's open-source platform for GPU computing. LightDiffusion-Next automatically detects and utilizes ROCm-compatible AMD GPUs through PyTorch's HIP backend.

### Supported Hardware

- **RDNA Architecture:**

  - RDNA 2 (RX 6000 series) - FP16 support
  - RDNA 3 (RX 7000 series) - FP16 and BF16 support

- **CDNA Architecture:**

  - CDNA (MI100)
  - CDNA 2 (MI200 series) - FP16 and BF16 support
  - CDNA 3 (MI300 series) - FP16 and BF16 support

### Installation

1. **Install ROCm drivers and runtime:**

   Follow the official [ROCm installation guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) for your Linux distribution.

```bash
   # Example for Ubuntu 22.04
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_latest_all.deb
   sudo apt-get install ./amdgpu-install_latest_all.deb
   sudo amdgpu-install --usecase=rocm
```

2. **Verify ROCm installation:**

```bash
   rocm-smi
   /opt/rocm/bin/rocminfo
```

3. **Install PyTorch with ROCm support:**
   
```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.0
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip uv

   # Install PyTorch with ROCm 6.0 support (adjust version as needed)
   uv pip install --index-url https://download.pytorch.org/whl/rocm6.0 torch torchvision
   
   # Install project dependencies
   uv pip install -r requirements.txt
```

4. **Launch LightDiffusion-Next:**

```bash
   streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

### ROCm-Specific Features

#### Automatic Detection

LightDiffusion-Next automatically detects ROCm GPUs at startup and reports them in the logs:

```
Device: cuda:0 AMD Radeon RX 7900 XTX (ROCm) : 
```

#### Memory Management

- **Cache Management:** ROCm uses a more conservative cache clearing strategy compared to CUDA. Cache is only cleared when explicitly forced to prevent memory fragmentation issues.
- **Memory Statistics:** Full memory statistics are available through the standard PyTorch CUDA API (which works transparently with ROCm).

#### Precision Support

- **FP16:** Fully supported on all RDNA and CDNA architectures
- **BF16:** Supported on RDNA 3+ and CDNA 2+ GPUs (automatically detected)
- **FP32:** Always available as fallback

#### Attention Mechanisms

| Feature | ROCm Support | Notes |
|---------|--------------|-------|
| PyTorch Scaled Dot-Product Attention (SDPA) | ✅ Yes | Default and recommended |
| PyTorch Flash Attention | ✅ Yes | Available on RDNA 3 and CDNA 2+ |
| xformers | ✅ Yes | Works with ROCm builds of xformers |
| SageAttention | ❌ No | CUDA-only kernels |
| SpargeAttn | ❌ No | CUDA-only kernels |

**Recommendation:** Use PyTorch's built-in attention (SDPA) on ROCm for best compatibility. Install xformers ROCm build for additional optimizations.

### Performance Tips

1. **Use BF16 on supported GPUs:**

   - RDNA 3 (RX 7000 series) and CDNA 2+ support BF16 natively
   - BF16 provides better numerical stability than FP16

2. **Enable PyTorch attention:**

   - Automatically enabled for PyTorch 2.0+
   - Provides good performance without CUDA-specific optimizations

3. **Install ROCm-compatible xformers:**
   
```bash
   # Build xformers from source for ROCm
   git clone https://github.com/facebookresearch/xformers.git
   cd xformers
   git submodule update --init --recursive
   pip install -e . --no-build-isolation
```

4. **Monitor GPU utilization:**

```bash
   watch -n 1 rocm-smi
```

### Known Limitations

- **SageAttention and SpargeAttn:** These optimizations use CUDA-specific kernels and are not available on ROCm. The system automatically falls back to PyTorch SDPA.
- **Stable-Fast:** May have limited support depending on ROCm version. Test compilation before relying on it.
- **Driver Maturity:** Ensure you're using the latest ROCm version for best stability and performance.

---

## Metal/MPS Support (Apple Silicon)

### Overview

Metal Performance Shaders (MPS) provides GPU acceleration on Apple Silicon Macs (M1, M2, M3 series). LightDiffusion-Next automatically detects and utilizes MPS when running on macOS.

### Supported Hardware

- **Apple Silicon:**

  - M1, M1 Pro, M1 Max, M1 Ultra
  - M2, M2 Pro, M2 Max, M2 Ultra
  - M3, M3 Pro, M3 Max
  - All future M-series chips

### Installation

1. **Ensure macOS is up to date:**

   - macOS 12.3 (Monterey) or later required
   - macOS 13+ (Ventura) recommended for best performance

2. **Install Python 3.10:**

```bash
   # Using Homebrew
   brew install python@3.10
```

3. **Create virtual environment and install dependencies:**

```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip

   # Install PyTorch with MPS support
   pip install torch torchvision torchaudio
   
   # Install project dependencies
   pip install -r requirements.txt
```

4. **Launch LightDiffusion-Next:**

```bash
   streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

### MPS-Specific Features

#### Automatic Detection

MPS is automatically detected and enabled on compatible hardware:

```
Device: mps
VAE dtype: torch.float16
Set vram state to: SHARED
```

#### Memory Management

- **Unified Memory:** Apple Silicon uses unified memory shared between CPU and GPU
- **VRAM State:** Automatically set to `SHARED` mode
- **Cache Management:** Uses `torch.mps.empty_cache()` for memory cleanup

#### Precision Support

- **FP16:** Fully supported and recommended (default)
- **FP32:** Supported but slower
- **BF16:** Not supported on MPS backend

#### Attention Mechanisms

| Feature | MPS Support | Notes |
|---------|-------------|-------|
| PyTorch Scaled Dot-Product Attention (SDPA) | ✅ Yes | Default and recommended |
| PyTorch Flash Attention | ❌ No | Not available on MPS |
| xformers | ❌ No | MPS backend not supported |
| SageAttention | ❌ No | CUDA/MPS incompatible |
| SpargeAttn | ❌ No | CUDA-only kernels |

**Recommendation:** Use PyTorch's built-in attention (SDPA) on MPS. It's well-optimized for Apple Silicon.

### Performance Tips

- **Use FP16 precision:**

MPS works best with FP16  
Automatically enabled by LightDiffusion-Next

- **Optimize batch sizes:**

Start with smaller batch sizes and increase gradually  
Monitor memory usage through Activity Monitor

- **Keep macOS updated:**

Apple regularly improves MPS performance in system updates  

- **Close unnecessary applications:**

Unified memory is shared with system processes  
Free up RAM for better GPU performance

- **Monitor GPU usage:**

```bash
   # Use Activity Monitor -> GPU tab
   # Or use powermetrics (requires sudo):
   sudo powermetrics --samplers gpu_power -i 1000
```

### Known Limitations

- **Non-blocking transfers:** Not supported; MPS operations are blocking
- **Advanced optimizations:** SageAttention, SpargeAttn, and xformers are not available
- **BF16:** Not supported on MPS backend
- **Memory pressure:** System may swap under high memory load due to unified architecture

### Unified Memory Considerations

Apple Silicon's unified memory architecture means:

- GPU and CPU share the same physical memory pool
- Less memory copying between devices
- System processes compete for the same memory
- Available VRAM depends on total system RAM and current usage

**Recommended RAM:**

- 16 GB: SD1.5 models at moderate resolutions
- 32 GB: Comfortable for most workflows including Flux (with quantization)
- 64 GB+: Professional workflows with large batch sizes

---

## Comparison Table

| Feature | NVIDIA (CUDA) | AMD (ROCm) | Apple (MPS) |
|---------|---------------|------------|-------------|
| FP16 | ✅ Full | ✅ Full | ✅ Full |
| BF16 | ✅ Full | ✅ RDNA3+/CDNA2+ | ❌ No |
| PyTorch SDPA | ✅ Yes | ✅ Yes | ✅ Yes |
| Flash Attention | ✅ Yes | ✅ RDNA3+/CDNA2+ | ❌ No |
| xformers | ✅ Yes | ✅ Build from source | ❌ No |
| SageAttention | ✅ Yes | ❌ No | ❌ No |
| SpargeAttn | ✅ Yes (CC 8.0-9.0) | ❌ No | ❌ No |
| Stable-Fast | ✅ Yes | ⚠️ Limited | ❌ No |
| Memory Management | ✅ Dedicated VRAM | ✅ Dedicated VRAM | ⚠️ Unified Memory |

---

## Troubleshooting

### ROCm Issues

**Problem:** PyTorch doesn't detect ROCm GPU

```bash
# Check ROCm installation
rocm-smi
rocminfo | grep "Name:"

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
```

**Problem:** Out of memory errors

- Reduce batch size
- Enable lower VRAM mode in settings
- Close other GPU-using applications
- Check with `rocm-smi` for memory usage

**Problem:** Slow performance

- Verify you're using the correct ROCm-optimized PyTorch build
- Check GPU utilization with `rocm-smi`
- Ensure FP16 or BF16 is enabled (check logs)

### MPS Issues

**Problem:** MPS not detected

```bash
# Verify MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```
- Ensure macOS 12.3+
- Update to latest macOS version
- Reinstall PyTorch

**Problem:** Memory warnings or crashes

- Reduce batch size
- Close other applications to free unified memory
- Check Activity Monitor for memory pressure

**Problem:** Slower than expected performance

- Verify FP16 is being used (check logs)
- Close background applications
- Update to latest macOS version for performance improvements
- Some models may be CPU-bound on older M1 chips

---

## Getting Help

For platform-specific issues:

1. Check the [FAQ](faq.md) for common questions
2. Review PyTorch's platform-specific documentation:
   - [ROCm installation](https://pytorch.org/get-started/locally/#linux-rocm)
   - [MPS backend](https://pytorch.org/docs/stable/notes/mps.html)
3. Open an issue on GitHub with:
   - Platform details (GPU model, driver version, OS)
   - LightDiffusion-Next startup logs
   - Output of `python -c "import torch; print(torch.__version__); print(torch.version.hip if hasattr(torch.version, 'hip') else 'CUDA'); print(torch.cuda.is_available())"`

---

**Note:** This documentation reflects the current state of ROCm and MPS support in PyTorch and LightDiffusion-Next. As these platforms mature, more optimizations and features may become available.
