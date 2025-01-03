import logging
import sys
from enum import Enum
from typing import Tuple, Union

import psutil
import torch

# Only this extra line of code is required to use oneDNN Graph
torch.jit.enable_onednn_fusion(True)
# torch.optimizer.zero_grad(set_to_none=True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.profiler.profile(enabled=False)


# FIXME: This is a workaround for the torch.backends.cuda.matmul.allow_tf32 attribute error

# if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.12.0"):
#     torch.backends.cuda.matmul.allow_tf32 = True


class VRAMState(Enum):
    """#### Enum for VRAM states.
    """
    DISABLED = 0  # No vram present: no need to move _internal to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but _internal still need to be moved between both.


class CPUState(Enum):
    """#### Enum for CPU states.
    """
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

directml_enabled = False
try:
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass


def is_intel_xpu() -> bool:
    """#### Check if Intel XPU is available.

    #### Returns:
        - `bool`: Whether Intel XPU is available.
    """
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device() -> torch.device:
    """#### Get the torch device.
    
    #### Returns:
        - `torch.device`: The torch device.
    """
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev: torch.device = None, torch_total_too: bool = False) -> int:
    """#### Get the total memory.

    #### Args:
        - `dev` (torch.device, optional): The device. Defaults to None.
        - `torch_total_too` (bool, optional): Whether to get the total memory in PyTorch. Defaults to False.

    #### Returns:
        - `int`: The total memory.
    """
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
logging.info(
    "Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram)
)
try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except:
        pass
    try:
        XFORMERS_VERSION = xformers.version.__version__
        logging.info("xformers version: {}".format(XFORMERS_VERSION))
        if XFORMERS_VERSION.startswith("0.0.18"):
            logging.warning(
                "\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images."
            )
            logging.warning(
                "Please downgrade or upgrade xformers to a different version.\n"
            )
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False


def is_nvidia() -> bool:
    """#### Checks if user has an Nvidia GPU

    #### Returns
        - `bool`: Whether the GPU is Nvidia
    """
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION is False:
                ENABLE_PYTORCH_ATTENTION = True
            if (
                torch.cuda.is_bf16_supported()
                and torch.cuda.get_device_properties(torch.cuda.current_device()).major
                >= 8
            ):
                VAE_DTYPE = torch.bfloat16
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


FORCE_FP32 = False
FORCE_FP16 = False

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = False

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")


def get_torch_device_name(device: torch.device) -> str:
    """#### Get the name of the torch compatible device

    #### Args:
        - `device` (torch.device): the device

    #### Returns:
        - `str`: the name of the device
    """
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(
                device, torch.cuda.get_device_name(device), allocator_backend
            )
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")

logging.info("VAE dtype: {}".format(VAE_DTYPE))

current_loaded_models = []


def module_size(module: torch.nn.Module) -> int:
    """#### Get the size of a module
    
    #### Args:
        - `module` (torch.nn.Module): The module
    
    #### Returns:
        - `int`: The size of the module
    """
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    """#### Class to load a model
    """
    def __init__(self, model: torch.nn.Module):
        """#### Initialize the class
        
        #### Args:
            - `model`: The model
        """
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None

    def model_memory(self):
        """#### Get the model memory
        
        #### Returns:
            - `int`: The model memory
        """
        return self.model.model_size()

    def model_memory_required(self, device: torch.device) -> int:
        """#### Get the required model memory
        
        #### Args:
            - `device`: The device
        
        #### Returns:
            - `int`: The required model memory
        """
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory: int = 0, force_patch_weights: bool = False) -> torch.nn.Module:
        """#### Load the model
        
        #### Args:
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch the weights. Defaults to False.
        
        #### Returns:
            - `torch.nn.Module`: The real model
        """
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        try:
            if lowvram_model_memory > 0 and load_weights:
                self.real_model = self.model.patch_model_lowvram(
                    device_to=patch_model_to,
                    lowvram_model_memory=lowvram_model_memory,
                    force_patch_weights=force_patch_weights,
                )
            else:
                self.real_model = self.model.patch_model(
                    device_to=patch_model_to, patch_weights=load_weights
                )
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e
        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights: bool = False) -> bool:
        """#### Checks if the model should be reloaded

        #### Args:
            - `force_patch_weights` (bool, optional): If model reloading should be enforced. Defaults to False.

        #### Returns:
            - `bool`: Whether the model should be reloaded
        """
        if force_patch_weights and self.model.lowvram_patch_counter > 0:
            return True
        return False

    def model_unload(self, unpatch_weights: bool = True) -> None:
        """#### Unloads the patched model

        #### Args:
            - `unpatch_weights` (bool, optional): Whether the weights should be unpatched. Defaults to True.
        """
        self.model.unpatch_model(
            self.model.offload_device, unpatch_weights=unpatch_weights
        )
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None

    def __eq__(self, other: torch.nn.Module) -> bool:
        """#### Verify if the model is equal to another

        #### Args:
            - `other` (torch.nn.Module): the other model

        #### Returns:
            - `bool`: Whether the two models are equal
        """
        return self.model is other.model


def minimum_inference_memory() -> int:
    """#### The minimum memory requirement for inference, equals to 1024^3

    #### Returns:
        - `int`: the memory requirement
    """
    return 1024 * 1024 * 1024


def unload_model_clones(model: torch.nn.Module, unload_weights_only:bool = True, force_unload: bool = True) -> bool:
    """#### Unloads the model clones

    #### Args:
        - `model` (torch.nn.Module): The model
        - `unload_weights_only` (bool, optional): Whether to unload only the weights. Defaults to True.
        - `force_unload` (bool, optional): Whether to force the unload. Defaults to True.

    #### Returns:
        - `bool`: Whether the model was unloaded
    """
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight is False:
            return None

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight


def free_memory(memory_required: int, device: torch.device, keep_loaded: list = []) -> None:
    """#### Free memory
    
    #### Args:
        - `memory_required` (int): The required memory
        - `device` (torch.device): The device
        - `keep_loaded` (list, optional): The list of loaded models to keep. Defaults to [].
    """
    unloaded_model = []
    can_unload = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append(
                    (sys.getrefcount(shift_model.model), shift_model.model_memory(), i)
                )

    for x in sorted(can_unload):
        i = x[-1]
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        current_loaded_models[i].model_unload()
        unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        current_loaded_models.pop(i)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(
                device, torch_free_too=True
            )
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def load_models_gpu(models: list, memory_required: int = 0, force_patch_weights: bool = False) -> None:
    """#### Load models on the GPU
    
    #### Args:
        - `models`(list): The models
        - `memory_required` (int, optional): The required memory. Defaults to 0.
        - `force_patch_weights` (bool, optional): Whether to force patch the weights. Defaults to False.
    """
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models = set(models)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        loaded = None

        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            if loaded.should_reload_model(force_patch_weights=force_patch_weights):
                current_loaded_models.pop(loaded_model_index).model_unload(
                    unpatch_weights=True
                )
                loaded = None
            else:
                models_already_loaded.append(loaded)

        if loaded is None:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return

    logging.info(
        f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}"
    )

    total_memory_required = {}
    for loaded_model in models_to_load:
        if (
            unload_model_clones(
                loaded_model.model, unload_weights_only=True, force_unload=False
            )
            is True
        ):  # unload clones where the weights are different
            total_memory_required[loaded_model.device] = total_memory_required.get(
                loaded_model.device, 0
            ) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(
                total_memory_required[device] * 1.3 + extra_mem,
                device,
                models_already_loaded,
            )

    for loaded_model in models_to_load:
        weights_unloaded = unload_model_clones(
            loaded_model.model, unload_weights_only=False, force_unload=False
        )  # unload the rest of the clones where the weights can stay loaded
        if weights_unloaded is not None:
            loaded_model.weights_loaded = not weights_unloaded

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (
            vram_set_state == VRAMState.LOW_VRAM
            or vram_set_state == VRAMState.NORMAL_VRAM
        ):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(
                max(64 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3)
            )
            if model_size > (
                current_free_mem - inference_memory
            ):  # only switch to lowvram if really necessary
                vram_set_state = VRAMState.LOW_VRAM
            else:
                lowvram_model_memory = 0

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 64 * 1024 * 1024

        loaded_model.model_load(
            lowvram_model_memory, force_patch_weights=force_patch_weights
        )
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model: torch.nn.Module) -> None:
    """#### Load a model on the GPU
    
    #### Args:
        - `model` (torch.nn.Module): The model
    """
    return load_models_gpu([model])


def cleanup_models(keep_clone_weights_loaded:bool = False):
    """#### Cleanup the models
    
    #### Args:
        - `keep_clone_weights_loaded` (bool, optional): Whether to keep the clone weights loaded. Defaults to False.
    """
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            elif (
                sys.getrefcount(current_loaded_models[i].real_model) <= 3
            ):  # references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype: torch.dtype) -> int:
    """#### Get the size of a dtype

    #### Args:
        - `dtype` (torch.dtype): The dtype

    #### Returns:
        - `int`: The size of the dtype
    """
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device() -> torch.device:
    """#### Get the offload device for UNet
    
    #### Returns:
        - `torch.device`: The offload device
    """
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype) -> torch.device:
    """#### Get the initial load device for UNet
    
    #### Args:
        - `parameters` (int): The parameters
        - `dtype` (torch.dtype): The dtype
    
    #### Returns:
        - `torch.device`: The initial load device
    """
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_dtype(
    device: torch.dtype = None,
    model_params: int = 0,
    supported_dtypes: list = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """#### Get the dtype for UNet

    #### Args:
        - `device` (torch.dtype, optional): The device. Defaults to None.
        - `model_params` (int, optional): The model parameters. Defaults to 0.
        - `supported_dtypes` (list, optional): The supported dtypes. Defaults to [torch.float16, torch.bfloat16, torch.float32].

    #### Returns:
        - `torch.dtype`: The dtype
    """
    if should_use_fp16(device=device, model_params=model_params, manual_cast=True):
        if torch.float16 in supported_dtypes:
            return torch.float16
    if should_use_bf16(device, model_params=model_params, manual_cast=True):
        if torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
    return torch.float32


# None means no manual cast
def unet_manual_cast(
    weight_dtype: torch.dtype,
    inference_device: torch.device,
    supported_dtypes: list = [torch.float16, torch.bfloat16, torch.float32],
) -> torch.dtype:
    """#### Manual cast for UNet

    #### Args:
        - `weight_dtype` (torch.dtype): The dtype of the weights
        - `inference_device` (torch.device): The device used for inference
        - `supported_dtypes` (list, optional): The supported dtypes. Defaults to [torch.float16, torch.bfloat16, torch.float32].

    #### Returns:
        - `torch.dtype`: The dtype
    """
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    if fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    elif bf16_supported and torch.bfloat16 in supported_dtypes:
        return torch.bfloat16
    else:
        return torch.float32


def text_encoder_offload_device() -> torch.device:
    """#### Get the offload device for the text encoder
    
    #### Returns:
        - `torch.device`: The offload device
    """
    return torch.device("cpu")


def text_encoder_device() -> torch.device:
    """#### Get the device for the text encoder
    
    #### Returns:
        - `torch.device`: The device
    """
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device: torch.device = None) -> torch.dtype:
    """#### Get the dtype for the text encoder

    #### Args:
        - `device` (torch.device, optional): The device used by the text encoder. Defaults to None.

    Returns:
        torch.dtype: The dtype
    """
    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device() -> torch.device:
    """#### Get the intermediate device
    
    #### Returns:
        - `torch.device`: The intermediate device
    """
    return torch.device("cpu")


def vae_device() -> torch.device:
    """#### Get the VAE device
    
    #### Returns:
        - `torch.device`: The VAE device
    """
    return get_torch_device()


def vae_offload_device() -> torch.device:
    """#### Get the offload device for VAE
    
    #### Returns:
        - `torch.device`: The offload device
    """
    return torch.device("cpu")


def vae_dtype():
    """#### Get the dtype for VAE
    
    #### Returns:
        - `torch.dtype`: The dtype
    """
    global VAE_DTYPE
    return VAE_DTYPE


def get_autocast_device(dev: torch.device) -> str:
    """#### Get the autocast device
    
    #### Args:
        - `dev` (torch.device): The device
    
    #### Returns:
        - `str`: The autocast device type
    """
    if hasattr(dev, "type"):
        return dev.type
    return "cuda"


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """#### Check if the device supports the dtype
    
    #### Args:
        - `device` (torch.device): The device to check
        - `dtype`  (torch.dtype): The dtype to check support
        
    #### Returns:
        - `bool`: Whether the dtype is supported by the device
    """
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def device_supports_non_blocking(device: torch.device) -> bool:
    """#### Check if the device supports non-blocking

    #### Args:
        - `device` (torch.device): The device to check

    #### Returns:
        - `bool`: Whether the device supports non-blocking
    """
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    return True


def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, copy: bool = False) -> torch.Tensor:
    """#### Cast a tensor to a device

    #### Args:
        - `tensor` (torch.Tensor): The tensor to cast
        - `device` (torch.device): The device to cast the tensor to
        - `dtype` (torch.dtype): The dtype precision to cast to
        - `copy` (bool, optional): Whether to copy the tensor. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The tensor cast to the device
    """
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, "type") and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_supports_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
        else:
            return tensor.to(device, non_blocking=non_blocking).to(
                dtype, non_blocking=non_blocking
            )
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def xformers_enabled() -> bool:
    """#### Check if xformers is enabled
    
    #### Returns:
        - `bool`: Whether xformers is enabled
    """
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae() -> bool:
    """#### Check if xformers is enabled for VAE
    
    #### Returns:
        - `bool`: Whether xformers is enabled for VAE
    """
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled() -> bool:
    """#### Check if PyTorch attention is enabled
    
    #### Returns:
        - `bool`: Whether PyTorch attention is enabled
    """
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention() -> bool:
    """#### Check if PyTorch flash attention is enabled and supported.

    #### Returns:
        - `bool`: True if PyTorch flash attention is enabled and supported, False otherwise.
    """
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
    return False


def get_free_memory(dev: torch.device = None, torch_free_too: bool = False) -> Union[int, Tuple[int, int]]:
    """#### Get the free memory available on the device.

    #### Args:
        - `dev` (torch.device, optional): The device to check memory for. Defaults to None.
        - `torch_free_too` (bool, optional): Whether to return both total and torch free memory. Defaults to False.

    #### Returns:
        - `int` or `Tuple[int, int]`: The free memory available. If `torch_free_too` is True, returns a tuple of total and torch free memory.
    """
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = (
                torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            )
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode() -> bool:
    """#### Check if the current mode is CPU.

    #### Returns:
        - `bool`: True if the current mode is CPU, False otherwise.
    """
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode() -> bool:
    """#### Check if the current mode is MPS.

    #### Returns:
        - `bool`: True if the current mode is MPS, False otherwise.
    """
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device: torch.device, type: str) -> bool:
    """#### Check if the device is of a specific type.

    #### Args:
        - `device` (torch.device): The device to check.
        - `type` (str): The type to check for.

    #### Returns:
        - `bool`: True if the device is of the specified type, False otherwise.
    """
    if hasattr(device, "type"):
        if device.type == type:
            return True
    return False


def is_device_cpu(device: torch.device) -> bool:
    """#### Check if the device is a CPU.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is a CPU, False otherwise.
    """
    return is_device_type(device, "cpu")


def is_device_mps(device: torch.device) -> bool:
    """#### Check if the device is an MPS.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is an MPS, False otherwise.
    """
    return is_device_type(device, "mps")


def is_device_cuda(device: torch.device) -> bool:
    """#### Check if the device is a CUDA device.

    #### Args:
        - `device` (torch.device): The device to check.

    #### Returns:
        - `bool`: True if the device is a CUDA device, False otherwise.
    """
    return is_device_type(device, "cuda")


def should_use_fp16(
    device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False
) -> bool:
    """#### Determine if FP16 should be used.

    #### Args:
        - `device` (torch.device, optional): The device to check. Defaults to None.
        - `model_params` (int, optional): The number of model parameters. Defaults to 0.
        - `prioritize_performance` (bool, optional): Whether to prioritize performance. Defaults to True.
        - `manual_cast` (bool, optional): Whether to manually cast. Defaults to False.

    #### Returns:
        - `bool`: True if FP16 should be used, False otherwise.
    """
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    fp16_works = False
    nvidia_10_series = [
        "1080",
        "1070",
        "titan x",
        "p3000",
        "p3200",
        "p4000",
        "p4200",
        "p5000",
        "p5200",
        "p6000",
        "1060",
        "1050",
        "p40",
        "p100",
        "p6",
        "p4",
    ]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    nvidia_16_series = [
        "1660",
        "1650",
        "1630",
        "T500",
        "T550",
        "T600",
        "MX550",
        "MX450",
        "CMP 30HX",
        "T2000",
        "T1000",
        "T1200",
    ]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(
    device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False
) -> bool:
    """#### Determine if BF16 should be used.

    #### Args:
        - `device` (torch.device, optional): The device to check. Defaults to None.
        - `model_params` (int, optional): The number of model parameters. Defaults to 0.
        - `prioritize_performance` (bool, optional): Whether to prioritize performance. Defaults to True.
        - `manual_cast` (bool, optional): Whether to manually cast. Defaults to False.

    #### Returns:
        - `bool`: True if BF16 should be used, False otherwise.
    """
    if device is not None:
        if is_device_cpu(device):
            return False

    if device is not None:
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works or manual_cast:
        free_model_memory = get_free_memory() * 0.9 - minimum_inference_memory()
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def soft_empty_cache(force: bool = False) -> None:
    """#### Softly empty the cache.

    #### Args:
        - `force` (bool, optional): Whether to force emptying the cache. Defaults to False.
    """
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if (
            force or is_nvidia()
        ):  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models() -> None:
    """#### Unload all models."""
    free_memory(1e30, get_torch_device())


def resolve_lowvram_weight(weight: torch.Tensor, model: torch.nn.Module, key: str) -> torch.Tensor:
    """#### Resolve low VRAM weight.

    #### Args:
        - `weight` (torch.Tensor): The weight tensor.
        - `model` (torch.nn.Module): The model.
        - `key` (str): The key.

    #### Returns:
        - `torch.Tensor`: The resolved weight tensor.
    """
    return weight