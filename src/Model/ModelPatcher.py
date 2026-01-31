"""ModelPatcher - Handles LoRA and weight patching for models."""
import copy
import logging
import uuid
import torch

from src.Device import Device
from src.NeuralNetwork import unet
from src.Utilities import util

try:
    import tomesd
    TOMESD_AVAILABLE = True
except ImportError:
    TOMESD_AVAILABLE = False
    tomesd = None


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights
    m.weight_function = None
    m.bias_function = None


class LowVramPatch:
    """Callback for lazy weight calculation in low VRAM mode."""
    def __init__(self, key: str, model_patcher: "ModelPatcher"):
        self.key = key
        self.model_patcher = model_patcher

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        return self.model_patcher.calculate_weight(
            self.model_patcher.patches[self.key], weight, self.key
        )


class ModelPatcher:
    def __init__(self, model: torch.nn.Module, load_device: torch.device,
                 offload_device: torch.device, size: int = 0,
                 current_device: torch.device = None, weight_inplace_update: bool = False):
        self.size = size
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.current_device = current_device if current_device else self.offload_device
        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()
        
        for attr, default in [("model_loaded_weight_memory", 0), 
                              ("model_lowvram", False), ("lowvram_patch_counter", 0)]:
            if not hasattr(self.model, attr):
                setattr(self.model, attr, default)
        
        self.tome_enabled = False
        self.tome_ratio = 0.5
        self.tome_info = {}

    def named_modules(self):
        """Compatibility method for tomesd - yields (name, module) from diffusion_model."""
        if hasattr(self.model, 'diffusion_model'):
            yield from self.model.diffusion_model.named_modules()
        else:
            return iter([])

    def loaded_size(self) -> int:
        return self.model.model_loaded_weight_memory

    def model_size(self) -> int:
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = Device.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self) -> "ModelPatcher":
        n = ModelPatcher(self.model, self.load_device, self.offload_device,
                         self.size, self.current_device, weight_inplace_update=self.weight_inplace_update)
        n.patches = {k: v[:] for k, v in self.patches.items()}
        n.patches_uuid = self.patches_uuid
        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other: object) -> bool:
        return hasattr(other, "model") and self.model is other.model

    def memory_required(self, input_shape: tuple) -> float:
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: callable):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function: callable):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name: str) -> object:
        return util.get_attr(self.model, name)

    def model_patches_to(self, device: torch.device):
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self) -> torch.dtype:
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0) -> list:
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches
        self.patches_uuid = uuid.uuid4()
        return list(p)

    def set_model_patch(self, patch: list, name: str):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch: list):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch: list):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_output_patch(self, patch: list):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch: list):
        self.set_model_patch(patch, "attn2_output_patch")

    def model_state_dict(self, filter_prefix: str = None) -> dict:
        return self.model.state_dict()

    def patch_weight_to_device(self, key: str, device_to: torch.device = None):
        if key not in self.patches:
            return
        weight = util.get_attr(self.model, key)
        inplace_update = self.weight_inplace_update
        if key not in self.backup:
            self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)
        temp_weight = (Device.cast_to_device(weight, device_to, torch.float32, copy=True) 
                       if device_to else weight.to(torch.float32, copy=True))
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
        if inplace_update:
            util.copy_to_param(self.model, key, out_weight)
        else:
            util.set_attr_param(self.model, key, out_weight)

    def load(self, device_to: torch.device = None, lowvram_model_memory: int = 0,
             force_patch_weights: bool = False, full_load: bool = False):
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        loading = [(Device.module_size(m), n, m) for n, m in self.model.named_modules()
                   if hasattr(m, "comfy_cast_weights") or hasattr(m, "weight")]
        loading.sort(reverse=True)
        load_completely = []

        for module_mem, n, m in loading:
            lowvram_weight = not full_load and hasattr(m, "comfy_cast_weights") and mem_counter + module_mem >= lowvram_model_memory
            weight_key, bias_key = f"{n}.weight", f"{n}.bias"

            if lowvram_weight:
                lowvram_counter += 1
                if hasattr(m, "prev_comfy_cast_weights"):
                    continue
                if force_patch_weights:
                    if weight_key in self.patches:
                        self.patch_weight_to_device(weight_key)
                    if bias_key in self.patches:
                        self.patch_weight_to_device(bias_key)
                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "comfy_cast_weights") and m.comfy_cast_weights:
                    wipe_lowvram_weight(m)
                if hasattr(m, "weight"):
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m))

        load_completely.sort(reverse=True)
        for _, n, m in load_completely:
            weight_key, bias_key = f"{n}.weight", f"{n}.bias"
            if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights:
                continue
            self.patch_weight_to_device(weight_key, device_to=device_to)
            self.patch_weight_to_device(bias_key, device_to=device_to)
            logging.debug(f"lowvram: loaded module regularly {n} {m}")
            m.comfy_patched_weights = True

        for _, _, m in load_completely:
            m.to(device_to)

        if lowvram_counter > 0:
            logging.info(f"loaded partially {lowvram_model_memory / (1024 * 1024):.1f} {mem_counter / (1024 * 1024):.1f} {patch_counter}")
            self.model.model_lowvram = True
        else:
            logging.info(f"loaded completely {lowvram_model_memory / (1024 * 1024):.1f} {mem_counter / (1024 * 1024):.1f} {full_load}")
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter

    def _apply_object_patches(self):
        """Apply object patches and backup originals."""
        for k in self.object_patches:
            old = util.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

    def patch_model_flux(self, device_to: torch.device = None, lowvram_model_memory: int = 0,
                         load_weights: bool = True, force_patch_weights: bool = False):
        self._apply_object_patches()
        full_load = lowvram_model_memory == 0
        if load_weights:
            self.load(device_to, lowvram_model_memory=lowvram_model_memory,
                      force_patch_weights=force_patch_weights, full_load=full_load)
        return self.model

    def patch_model_lowvram_flux(self, device_to: torch.device = None,
                                  lowvram_model_memory: int = 0, force_patch_weights: bool = False) -> torch.nn.Module:
        return self._patch_model_lowvram_impl(device_to, lowvram_model_memory, force_patch_weights)

    def patch_model(self, device_to: torch.device = None, patch_weights: bool = True) -> torch.nn.Module:
        self._apply_object_patches()
        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning(f"could not patch. key doesn't exist in model: {key}")
                    continue
                self.patch_weight_to_device(key, device_to)
            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to
        return self.model

    def patch_model_lowvram(self, device_to: torch.device = None,
                            lowvram_model_memory: int = 0, force_patch_weights: bool = False) -> torch.nn.Module:
        return self._patch_model_lowvram_impl(device_to, lowvram_model_memory, force_patch_weights)

    def _patch_model_lowvram_impl(self, device_to, lowvram_model_memory, force_patch_weights):
        """Shared implementation for lowvram patching."""
        self.patch_model(device_to, patch_weights=False)
        logging.info(f"loading in lowvram mode {lowvram_model_memory / (1024 * 1024):.1f}")

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = hasattr(m, "comfy_cast_weights") and mem_counter + Device.module_size(m) >= lowvram_model_memory
            weight_key, bias_key = f"{n}.weight", f"{n}.bias"

            if lowvram_weight:
                for pkey in [weight_key, bias_key]:
                    if pkey in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(pkey)
                        else:
                            setattr(m, 'weight_function' if 'weight' in pkey else 'bias_function',
                                    LowVramPatch(pkey, self))
                            patch_counter += 1
                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            elif hasattr(m, "weight"):
                self.patch_weight_to_device(weight_key, device_to)
                self.patch_weight_to_device(bias_key, device_to)
                m.to(device_to)
                mem_counter += Device.module_size(m)
                logging.debug(f"lowvram: loaded module regularly {m}")

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches: list, weight: torch.Tensor, key: str) -> torch.Tensor:
        for p in patches:
            alpha, v = p[0], p[1]
            v = v[1]
            mat1 = Device.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = Device.cast_to_device(v[1], weight.device, torch.float32)
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
        return weight

    def unpatch_model(self, device_to: torch.device = None, unpatch_weights: bool = True):
        if unpatch_weights:
            for k in list(self.backup.keys()):
                util.set_attr_param(self.model, k, self.backup[k])
            self.backup.clear()
            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to
        self.object_patches_backup.clear()

    def partially_load(self, device_to: torch.device, extra_memory: int = 0) -> int:
        self.unpatch_model(unpatch_weights=False)
        self.patch_model(patch_weights=False)
        if not self.model.model_lowvram:
            return 0
        full_load = self.model.model_loaded_weight_memory + extra_memory > self.model_size()
        current_used = self.model.model_loaded_weight_memory
        self.load(device_to, lowvram_model_memory=current_used + extra_memory, full_load=full_load)
        return self.model.model_loaded_weight_memory - current_used

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def apply_tome(self, ratio: float = 0.5, max_downsample: int = 1) -> bool:
        """Apply Token Merging (ToMe) to the model."""
        if not TOMESD_AVAILABLE:
            logging.warning("Token Merging (tomesd) not available. Install with: pip install tomesd")
            return False
        try:
            try:
                tomesd.remove_patch(self)
            except:
                pass
        except:
            pass
        self.tome_enabled = False
        self.tome_ratio = 0.5

        try:
            if hasattr(self.model, 'diffusion_model'):
                tomesd.apply_patch(self, ratio=ratio, max_downsample=max_downsample)
                self.tome_enabled = True
                self.tome_ratio = ratio
                logging.info(f"Applied Token Merging with ratio={ratio}, max_downsample={max_downsample}")
                print(f"✓ Token Merging ACTIVE: {ratio*100:.0f}% merge ratio, max_downsample={max_downsample}")
                return True
            logging.warning("Model does not have 'diffusion_model' attribute, cannot apply ToMe")
            return False
        except Exception as e:
            logging.error(f"Failed to apply Token Merging: {e}")
            return False

    def remove_tome(self) -> bool:
        """Remove Token Merging (ToMe) from the model."""
        if not TOMESD_AVAILABLE or not self.tome_enabled:
            return False
        try:
            tomesd.remove_patch(self)
            self.tome_enabled = False
            self.tome_ratio = 0.5
            self.tome_info = {}
            logging.info("Removed Token Merging patch")
            return True
        except Exception as e:
            logging.error(f"Failed to remove Token Merging: {e}")
            return False


def unet_prefix_from_state_dict(state_dict: dict) -> str:
    """Get the UNet prefix from the state dictionary."""
    candidates = ["model.diffusion_model.", "model.model."]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    return top if counts[top] > 5 else "model."


def load_diffusion_model_state_dict(sd, model_options={}) -> ModelPatcher:
    """Load the diffusion model state dictionary."""
    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = unet_prefix_from_state_dict(sd)
    temp_sd = util.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = util.calculate_parameters(sd)
    load_device = Device.get_torch_device()
    model_config = unet.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd

    offload_device = Device.unet_offload_device()
    unet_dtype2 = dtype if dtype else Device.unet_dtype(
        model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = Device.unet_manual_cast(unet_dtype2, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype2, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info(f"left over keys in unet: {left_over}")
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device)
