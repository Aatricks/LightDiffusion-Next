"""Simplified device and memory management for LightDiffusion-Next."""
import logging
import platform
import sys
from enum import Enum
from typing import Optional, Union, Tuple
import psutil
import torch

# Enable TF32 on supported hardware
try:
    torch.backends.cuda.matmul.allow_tf32 = True
except:
    pass


class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Global state
vram_state = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
directml_enabled = False
xpu_available = False
DISABLE_SMART_MEMORY = False
FORCE_FP32 = False
FORCE_FP16 = False
WINDOWS = any(platform.win32_ver())
EXTRA_RESERVED_VRAM = 600 * 1024 * 1024 if WINDOWS else 400 * 1024 * 1024

# Detect hardware
try:
    xpu_available = torch.xpu.is_available()
except:
    pass
try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
except:
    pass

# Library availability
XFORMERS_IS_AVAILABLE = False
XFORMERS_ENABLED_VAE = True
SAGEATTENTION_IS_AVAILABLE = False
SAGEATTENTION_ENABLED_VAE = True
SPARGEATTN_IS_AVAILABLE = False
SPARGEATTN_ENABLED_VAE = True
ENABLE_PYTORCH_ATTENTION = False
VAE_DTYPE = torch.float32

try:
    import xformers.ops
    XFORMERS_IS_AVAILABLE = getattr(xformers, '_has_cpp_library', True)
    v = getattr(xformers.version, '__version__', '')
    if v.startswith("0.0.18"):
        XFORMERS_ENABLED_VAE = False
        logging.warning("xformers 0.0.18 has black image bugs")
except:
    pass

try:
    import sageattention
    SAGEATTENTION_IS_AVAILABLE = True
except:
    pass

try:
    import spas_sage_attn
    SPARGEATTN_IS_AVAILABLE = True
except:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception


def is_intel_xpu() -> bool:
    return cpu_state == CPUState.GPU and xpu_available


def is_nvidia() -> bool:
    return cpu_state == CPUState.GPU and bool(torch.version.cuda)


def is_rocm() -> bool:
    return cpu_state == CPUState.GPU and bool(torch.version.hip)


def get_torch_device() -> torch.device:
    if directml_enabled:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    if torch.cuda.is_available():
        return torch.device(torch.cuda.current_device())
    return torch.device("cpu")


def get_total_memory(dev: torch.device = None, torch_total_too: bool = False) -> Union[int, Tuple[int, int]]:
    dev = dev or get_torch_device()
    if hasattr(dev, "type") and dev.type in ("cpu", "mps"):
        mem = psutil.virtual_memory().total
        return (mem, mem) if torch_total_too else mem
    if directml_enabled:
        mem = 1024 ** 3
        return (mem, mem) if torch_total_too else mem
    if is_intel_xpu():
        stats = torch.xpu.memory_stats(dev)
        mem_torch = stats["reserved_bytes.all.current"]
        mem_total = torch.xpu.get_device_properties(dev).total_memory
    else:
        stats = torch.cuda.memory_stats(dev)
        mem_torch = stats["reserved_bytes.all.current"]
        _, mem_total = torch.cuda.mem_get_info(dev)
    return (mem_total, mem_torch) if torch_total_too else mem_total


def get_free_memory(dev: torch.device = None, torch_free_too: bool = False) -> Union[int, Tuple[int, int]]:
    dev = dev or get_torch_device()
    if hasattr(dev, "type") and dev.type in ("cpu", "mps"):
        mem = psutil.virtual_memory().available
        return (mem, mem) if torch_free_too else mem
    if directml_enabled:
        mem = 1024 ** 3
        return (mem, mem) if torch_free_too else mem
    if is_intel_xpu():
        stats = torch.xpu.memory_stats(dev)
        active = stats["active_bytes.all.current"]
        reserved = stats["reserved_bytes.all.current"]
        free_torch = reserved - active
        free_total = torch.xpu.get_device_properties(dev).total_memory - reserved + free_torch
    else:
        stats = torch.cuda.memory_stats(dev)
        active = stats["active_bytes.all.current"]
        reserved = stats["reserved_bytes.all.current"]
        free_cuda, _ = torch.cuda.mem_get_info(dev)
        free_torch = reserved - active
        free_total = free_cuda + free_torch
    return (free_total, free_torch) if torch_free_too else free_total


def soft_empty_cache(force: bool = False) -> None:
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available() and (force or is_nvidia()):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Initialize PyTorch attention and VAE dtype
try:
    if is_nvidia() or is_rocm():
        if int(torch.version.__version__[0]) >= 2:
            ENABLE_PYTORCH_ATTENTION = True
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            if is_nvidia() and torch.cuda.get_device_properties(0).major >= 8:
                VAE_DTYPE = torch.bfloat16
            elif is_rocm():
                VAE_DTYPE = torch.bfloat16
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if ENABLE_PYTORCH_ATTENTION and torch.cuda.is_available():
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

# Apply vram_state based on cpu_state
if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED
elif cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

total_vram = get_total_memory() / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info(f"VRAM: {total_vram:.0f} MB, RAM: {total_ram:.0f} MB, Device: {get_torch_device()}, VAE dtype: {VAE_DTYPE}")


# Model management
current_loaded_models = []


def module_size(module: torch.nn.Module) -> int:
    return sum(t.nelement() * t.element_size() for t in module.state_dict().values())


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def __eq__(self, other):
        return isinstance(other, LoadedModel) and self.model == other.model

    def model_memory(self):
        return self.model.model_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if hasattr(self.model, 'current_loaded_device') and device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        return self.model_memory()

    def model_load(self, lowvram_model_memory: int = 0, force_patch_weights: bool = False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())
        load_weights = not self.weights_loaded
        
        try:
            if hasattr(self.model, "patch_model_lowvram") and lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(
                    device_to=self.device, lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights)
            else:
                self.real_model = self.model.patch_model(device_to=self.device, patch_weights=load_weights)
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e
        
        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights: bool = False) -> bool:
        return force_patch_weights and self.model.lowvram_patch_counter > 0

    def model_unload(self, unpatch_weights: bool = True):
        self.model.unpatch_model(self.model.offload_device, unpatch_weights=unpatch_weights)
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def model_use_more_vram(self, extra_memory: int) -> int:
        return self.model.partially_load(self.device, extra_memory)


def minimum_inference_memory() -> int:
    return 1024 * 1024 * 1024


def extra_reserved_memory() -> int:
    return EXTRA_RESERVED_VRAM


def unload_model_clones(model, unload_weights_only: bool = True, force_unload: bool = True):
    to_unload = [i for i in range(len(current_loaded_models) - 1, -1, -1) 
                 if model.is_clone(current_loaded_models[i].model)]
    if not to_unload:
        return True
    if not force_unload and unload_weights_only:
        return None
    for i in to_unload:
        current_loaded_models.pop(i).model_unload(unpatch_weights=True)
    return True


def free_memory(memory_required: int, device: torch.device, keep_loaded: list = []):
    can_unload = [(sys.getrefcount(m.model), m.model_memory(), i) 
                  for i, m in enumerate(current_loaded_models) 
                  if m.device == device and m not in keep_loaded]
    unloaded = []
    for x in sorted(can_unload):
        if not DISABLE_SMART_MEMORY and get_free_memory(device) > memory_required:
            break
        current_loaded_models[x[-1]].model_unload()
        unloaded.append(x[-1])
    for i in sorted(unloaded, reverse=True):
        current_loaded_models.pop(i)
    if unloaded:
        soft_empty_cache()


def load_models_gpu(models: list, memory_required: int = 0, force_patch_weights: bool = False,
                    minimum_memory_required: int = None, force_full_load: bool = False):
    global vram_state
    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)
    min_mem = minimum_memory_required or extra_mem
    
    models_to_load, models_already_loaded = [], []
    for x in set(models):
        loaded_model = LoadedModel(x)
        try:
            idx = current_loaded_models.index(loaded_model)
            loaded = current_loaded_models[idx]
            if loaded.should_reload_model(force_patch_weights=force_patch_weights):
                current_loaded_models.pop(idx).model_unload(unpatch_weights=True)
                models_to_load.append(loaded_model)
            else:
                models_already_loaded.append(loaded)
        except ValueError:
            if hasattr(x, "model"):
                logging.info(f"Loading {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)
    
    if not models_to_load:
        for d in set(m.device for m in models_already_loaded):
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return
    
    # Calculate and free memory
    mem_required = {}
    for m in models_to_load:
        if unload_model_clones(m.model, unload_weights_only=True, force_unload=False):
            mem_required[m.device] = mem_required.get(m.device, 0) + m.model_memory_required(m.device)
    
    for device, mem in mem_required.items():
        if device != torch.device("cpu"):
            free_memory(mem * 1.3 + extra_mem, device, models_already_loaded)
    
    for m in models_to_load:
        weights_unloaded = unload_model_clones(m.model, unload_weights_only=False, force_unload=False)
        if weights_unloaded is not None:
            m.weights_loaded = not weights_unloaded
    
    # Load models
    for loaded_model in models_to_load:
        torch_dev = loaded_model.model.load_device
        vram_set = VRAMState.DISABLED if is_device_cpu(torch_dev) else vram_state
        lowvram_mem = 0
        
        if vram_set in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not force_full_load:
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free = get_free_memory(torch_dev)
            lowvram_mem = int(max(64 * 1024 * 1024, (current_free - 1024 * 1024 * 1024) / 1.3))
            if model_size <= current_free - inference_memory:
                lowvram_mem = 0
        
        if vram_set == VRAMState.NO_VRAM:
            lowvram_mem = 64 * 1024 * 1024
        
        loaded_model.model_load(lowvram_mem, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)


def load_model_gpu(model):
    load_models_gpu([model])


def cleanup_models(keep_clone_weights_loaded: bool = False):
    to_delete = [i for i in range(len(current_loaded_models) - 1, -1, -1)
                 if sys.getrefcount(current_loaded_models[i].model) <= 2 and
                 (not keep_clone_weights_loaded or sys.getrefcount(current_loaded_models[i].real_model) <= 3)]
    for i in to_delete:
        current_loaded_models.pop(i).model_unload()


def unload_all_models():
    free_memory(int(1e30), get_torch_device())


# Device utilities
def is_device_type(device, dtype: str) -> bool:
    return hasattr(device, "type") and device.type == dtype


def is_device_cpu(device) -> bool:
    return is_device_type(device, "cpu")


def is_device_mps(device) -> bool:
    return is_device_type(device, "mps")


def is_device_cuda(device) -> bool:
    return is_device_type(device, "cuda")


def cpu_mode() -> bool:
    return cpu_state == CPUState.CPU


def mps_mode() -> bool:
    return cpu_state == CPUState.MPS


# Dtype utilities
def dtype_size(dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return getattr(dtype, 'itemsize', 4)


def supports_dtype(device, dtype) -> bool:
    if dtype == torch.float32:
        return True
    return not is_device_cpu(device)


def supports_cast(device, dtype) -> bool:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return True
    if directml_enabled or is_device_mps(device):
        return False
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def cast_to_device(tensor, device, dtype, copy: bool = False):
    non_blocking = not is_device_mps(device)
    can_cast = tensor.dtype in (torch.float32, torch.float16) or \
               (tensor.dtype == torch.bfloat16 and (is_device_cuda(device) or is_intel_xpu()))
    if can_cast:
        if copy and tensor.device == device:
            return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
        return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def pick_weight_dtype(dtype, fallback_dtype, device):
    dtype = dtype or fallback_dtype
    if dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype
    if not supports_cast(device, dtype):
        dtype = fallback_dtype
    return dtype


# UNet/VAE/text encoder device helpers
def unet_offload_device() -> torch.device:
    return get_torch_device() if vram_state == VRAMState.HIGH_VRAM else torch.device("cpu")


def unet_inital_load_device(parameters, dtype) -> torch.device:
    if vram_state == VRAMState.HIGH_VRAM or DISABLE_SMART_MEMORY:
        return get_torch_device() if vram_state == VRAMState.HIGH_VRAM else torch.device("cpu")
    model_size = dtype_size(dtype) * parameters
    if get_free_memory(get_torch_device()) > get_free_memory(torch.device("cpu")) and model_size < get_free_memory(get_torch_device()):
        return get_torch_device()
    return torch.device("cpu")


def unet_dtype(device=None, model_params: int = 0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True) and torch.float16 in supported_dtypes:
        return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True) and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    return torch.float32


def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32:
        return None
    if should_use_fp16(inference_device, prioritize_performance=False) and weight_dtype == torch.float16:
        return None
    if should_use_bf16(inference_device) and weight_dtype == torch.bfloat16:
        return None
    if should_use_fp16(inference_device, prioritize_performance=False) and torch.float16 in supported_dtypes:
        return torch.float16
    if should_use_bf16(inference_device) and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    return torch.float32


def text_encoder_offload_device() -> torch.device:
    return torch.device("cpu")


def text_encoder_device() -> torch.device:
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM) and should_use_fp16(prioritize_performance=False):
        return get_torch_device()
    return torch.device("cpu")


def text_encoder_initial_device(load_device, offload_device, model_size: int = 0):
    if load_device == offload_device or model_size <= 1024 ** 3 or is_device_mps(load_device):
        return offload_device
    if get_free_memory(load_device) > get_free_memory(offload_device) * 0.5 and model_size * 1.2 < get_free_memory(load_device):
        return load_device
    return offload_device


def text_encoder_dtype(device=None):
    if is_device_cpu(device):
        return torch.float16
    return torch.bfloat16 if should_use_bf16(device) else torch.float16


def intermediate_device() -> torch.device:
    return torch.device("cpu")


def vae_device() -> torch.device:
    return get_torch_device()


def vae_offload_device() -> torch.device:
    return torch.device("cpu")


def vae_dtype():
    return VAE_DTYPE


def get_autocast_device(dev) -> str:
    return getattr(dev, "type", "cuda")


# Feature detection
def sageattention_enabled() -> bool:
    if cpu_state != CPUState.GPU or is_intel_xpu() or directml_enabled or is_rocm():
        return False
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 12:
                return False
        except:
            pass
    return SAGEATTENTION_IS_AVAILABLE


def sageattention_enabled_vae() -> bool:
    return sageattention_enabled() and SAGEATTENTION_ENABLED_VAE


def spargeattn_enabled() -> bool:
    if cpu_state != CPUState.GPU or is_intel_xpu() or directml_enabled or is_rocm():
        return False
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 12:
                return False
        except:
            pass
    return SPARGEATTN_IS_AVAILABLE


def spargeattn_enabled_vae() -> bool:
    return spargeattn_enabled() and SPARGEATTN_ENABLED_VAE


def xformers_enabled() -> bool:
    if cpu_state != CPUState.GPU or is_intel_xpu() or directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae() -> bool:
    return xformers_enabled() and XFORMERS_ENABLED_VAE


def pytorch_attention_enabled() -> bool:
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention() -> bool:
    return ENABLE_PYTORCH_ATTENTION and (is_nvidia() or is_rocm())


def device_supports_non_blocking(device) -> bool:
    return not is_device_mps(device)


# FP16/BF16 support detection
def should_use_fp16(device=None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if FORCE_FP16:
        return True
    if FORCE_FP32 or directml_enabled or cpu_mode():
        return False
    if device and is_device_cpu(device):
        return False
    if mps_mode() or (device and is_device_mps(device)):
        return True
    if is_intel_xpu() or is_rocm():
        return True
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True
    if props.major < 6:
        return False
    # Check 10-series cards
    fp16_works = any(x in props.name.lower() for x in ["1080", "1070", "titan x", "p3000", "p4000", "p5000", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"])
    if fp16_works or manual_cast:
        if not prioritize_performance or model_params * 4 > get_free_memory() * 0.9 - minimum_inference_memory():
            return True
    if props.major < 7:
        return False
    # Exclude 16-series
    return not any(x in props.name for x in ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"])


def should_use_bf16(device=None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if FORCE_FP32 or directml_enabled or cpu_mode() or mps_mode():
        return False
    if device and (is_device_cpu(device) or is_device_mps(device)):
        return False
    if is_intel_xpu():
        return True
    if is_rocm():
        try:
            return torch.cuda.is_bf16_supported()
        except:
            return False
    device = device or torch.device("cuda")
    if torch.cuda.get_device_properties(device).major >= 8:
        return True
    try:
        bf16_works = torch.cuda.is_bf16_supported()
        if bf16_works or manual_cast:
            if not prioritize_performance or model_params * 4 > get_free_memory() * 0.9 - minimum_inference_memory():
                return True
    except:
        pass
    return False


def resolve_lowvram_weight(weight, model, key):
    return weight
