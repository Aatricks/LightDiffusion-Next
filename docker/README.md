# Docker Build Scripts

This directory contains helper scripts used during the Docker image build process.

## Files

### patch_sageattention.py
**Purpose**: Patches the SageAttention setup.py to support building without GPU present.

**What it does**:
- Adds support for the `TORCH_CUDA_ARCH_LIST` environment variable to SageAttention
- Allows specifying target GPU architectures via environment variable
- Enables building Docker images on machines without NVIDIA GPUs

**Usage** (automatically called during Docker build):
```bash
cd SageAttention
python3 ../docker/patch_sageattention.py
```

**Why it's needed**:
SageAttention's original setup.py tries to detect GPU hardware during build time using `torch.cuda.device_count()`. This fails in Docker builds because:
1. Docker builds don't have GPU access by default (even with `--gpus all`)
2. GPU access during build is not guaranteed across all Docker configurations
3. Build machines may not have the same GPU as the target runtime machine

The patch adds a check for `TORCH_CUDA_ARCH_LIST` environment variable before attempting hardware detection, allowing explicit specification of target architectures.

### sageattention_setup.patch (not used)
Legacy patch file - kept for reference. The Python script approach is preferred.

## How the Build Process Works

1. **Environment Setup**: `TORCH_CUDA_ARCH_LIST` is set in Dockerfile via ARG/ENV
2. **Patch Application**: `patch_sageattention.py` modifies SageAttention's setup.py
3. **Extension Build**: Modified setup.py reads `TORCH_CUDA_ARCH_LIST` and compiles for specified architectures
4. **SpargeAttn Build**: Already supports `TORCH_CUDA_ARCH_LIST` natively, no patch needed

## Maintenance

If SageAttention is updated, one may need to:
1. Check if the patch still applies correctly
2. Update the target line in `patch_sageattention.py` if the setup.py structure changes
3. Test the build process with the new version

The patch is designed to be non-intrusive and should work across most SageAttention versions that follow the same setup.py structure.
