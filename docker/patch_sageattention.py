#!/usr/bin/env python3
"""
Patch for SageAttention setup.py to support TORCH_CUDA_ARCH_LIST environment variable.
This allows building without GPUs present during build time.
"""

import sys

setup_py_path = "setup.py"

# Read the original setup.py
with open(setup_py_path, 'r') as f:
    content = f.read()

# Find the line where compute_capabilities is initialized
target_line = "compute_capabilities = set()"

if target_line not in content:
    print("ERROR: Could not find target line in setup.py")
    sys.exit(1)

# Add our patch right after compute_capabilities initialization
patch_code = '''
# Check for TORCH_CUDA_ARCH_LIST environment variable first (Docker build support)
env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if env_arch_list:
    print(f"Using TORCH_CUDA_ARCH_LIST from environment: {env_arch_list}")
    arch_list = env_arch_list.replace(" ", ";").split(";")
    for arch in arch_list:
        arch = arch.strip()
        if not arch:
            continue
        if arch.endswith("+PTX"):
            arch = arch[:-4].strip()
        if arch:
            compute_capabilities.add(arch)
'''

# Insert the patch
content = content.replace(
    target_line,
    target_line + patch_code
)

# Write back
with open(setup_py_path, 'w') as f:
    f.write(content)

print("✓ Successfully patched setup.py to support TORCH_CUDA_ARCH_LIST")
