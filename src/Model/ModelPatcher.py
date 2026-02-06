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
    m.weight_function = m.bias_function = None


class LowVramPatch:
    def __init__(self, key: str, model_patcher: "ModelPatcher"):
        self.key, self.model_patcher = key, model_patcher

    def __call__(self, weight: torch.Tensor) -> torch.Tensor:
        return self.model_patcher.calculate_weight(self.model_patcher.patches[self.key], weight, self.key)


class ModelPatcher:
    def __init__(self, model: torch.nn.Module, load_device: torch.device, offload_device: torch.device,
                 size: int = 0, current_device: torch.device = None, weight_inplace_update: bool = False):
        self.size, self.model, self.patches, self.backup = size, model, {}, {}
        self.object_patches, self.object_patches_backup = {}, {}
        self.model_options = {"transformer_options": {}}
        self.model_size()
        self.load_device, self.offload_device = load_device, offload_device
        self.current_device = current_device or self.offload_device
        self.weight_inplace_update, self.model_lowvram, self.lowvram_patch_counter = weight_inplace_update, False, 0
        self.patches_uuid = uuid.uuid4()
        self.tome_enabled, self.tome_ratio, self.tome_info = False, 0.5, {}
        for attr, default in [("model_loaded_weight_memory", 0), ("model_lowvram", False), ("lowvram_patch_counter", 0)]:
            if not hasattr(self.model, attr):
                setattr(self.model, attr, default)

    def named_modules(self):
        yield from self.model.diffusion_model.named_modules() if hasattr(self.model, 'diffusion_model') else iter([])

    def loaded_size(self):
        return self.model.model_loaded_weight_memory

    def model_size(self) -> int:
        if self.size > 0:
            return self.size
        self.size = Device.module_size(self.model)
        self.model_keys = set(self.model.state_dict().keys())
        return self.size

    def clone(self) -> "ModelPatcher":
        n = ModelPatcher(self.model, self.load_device, self.offload_device, self.size, self.current_device, self.weight_inplace_update)
        n.patches = {k: v[:] for k, v in self.patches.items()}
        n.patches_uuid, n.object_patches = self.patches_uuid, self.object_patches.copy()
        n.model_options, n.model_keys, n.backup = copy.deepcopy(self.model_options), self.model_keys, self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        return hasattr(other, "model") and self.model is other.model

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, f):
        self.model_options["model_function_wrapper"] = f

    def set_model_denoise_mask_function(self, f):
        self.model_options["denoise_mask_function"] = f

    def get_model_object(self, name):
        return util.get_attr(self.model, name)

    def model_patches_to(self, device):
        wrap_func = self.model_options.get("model_function_wrapper")
        if wrap_func and hasattr(wrap_func, "to"):
            self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        return self.model.get_dtype() if hasattr(self.model, "get_dtype") else None

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                self.patches[k] = self.patches.get(k, []) + [(strength_patch, patches[k], strength_model)]
        self.patches_uuid = uuid.uuid4()
        return list(p)

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        to.setdefault("patches", {})[name] = to.get("patches", {}).get(name, []) + [patch]

    def set_model_attn1_patch(self, patch):
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch):
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_output_patch(self, patch):
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch):
        self.set_model_patch(patch, "attn2_output_patch")

    def model_state_dict(self, filter_prefix=None):
        return self.model.state_dict()

    def patch_weight_to_device(self, key, device_to=None):
        if key not in self.patches:
            return
        weight = util.get_attr(self.model, key)
        if key not in self.backup:
            self.backup[key] = weight.to(device=self.offload_device, copy=self.weight_inplace_update)
        temp_weight = Device.cast_to_device(weight, device_to, torch.float32, copy=True) if device_to else weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
        (util.copy_to_param if self.weight_inplace_update else util.set_attr_param)(self.model, key, out_weight)

    def weight_only_quantize(self, dtype: torch.dtype = torch.float8_e4m3fn):
        """Quantize all model weights to the target dtype (weight-only)."""
        logging.info(f"Quantizing model weights to {dtype}")
        with torch.no_grad():
            for n, m in self.model.named_modules():
                if hasattr(m, "weight") and m.weight is not None:
                    # Don't quantize small tensors or non-float weights
                    if m.weight.numel() > 4096 and m.weight.is_floating_point():
                        q_weight = m.weight.to(dtype)
                        # We keep it as a Parameter so it can be used in forward
                        m.weight = torch.nn.Parameter(q_weight, requires_grad=False)
                        # Enable weight casting so it dequantizes to input dtype on the fly
                        if hasattr(m, "comfy_cast_weights"):
                            m.comfy_cast_weights = True
                if hasattr(m, "bias") and m.bias is not None:
                    # Biases are usually kept in higher precision
                    pass

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        mem_counter, patch_counter, lowvram_counter = 0, 0, 0
        loading = sorted([(Device.module_size(m), n, m) for n, m in self.model.named_modules()
                          if hasattr(m, "comfy_cast_weights") or hasattr(m, "weight")], reverse=True)
        load_completely = []
        for module_mem, n, m in loading:
            lowvram_weight = not full_load and hasattr(m, "comfy_cast_weights") and mem_counter + module_mem >= lowvram_model_memory
            if lowvram_weight:
                lowvram_counter += 1
                if hasattr(m, "prev_comfy_cast_weights"):
                    continue
                if force_patch_weights:
                    for pkey in [f"{n}.weight", f"{n}.bias"]:
                        if pkey in self.patches:
                            self.patch_weight_to_device(pkey)
                m.prev_comfy_cast_weights, m.comfy_cast_weights = m.comfy_cast_weights, True
            else:
                if hasattr(m, "comfy_cast_weights") and m.comfy_cast_weights:
                    wipe_lowvram_weight(m)
                if hasattr(m, "weight"):
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m))
        for _, n, m in sorted(load_completely, reverse=True):
            if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights:
                continue
            self.patch_weight_to_device(f"{n}.weight", device_to)
            self.patch_weight_to_device(f"{n}.bias", device_to)
            m.comfy_patched_weights = True
        for _, _, m in load_completely:
            m.to(device_to)
        if lowvram_counter > 0:
            logging.info(f"loaded partially {lowvram_model_memory / 1e6:.1f} {mem_counter / 1e6:.1f} {patch_counter}")
            self.model.model_lowvram = True
        else:
            logging.info(f"loaded completely {lowvram_model_memory / 1e6:.1f} {mem_counter / 1e6:.1f} {full_load}")
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()
        self.model.lowvram_patch_counter += patch_counter
        self.model.device, self.model.model_loaded_weight_memory = device_to, mem_counter

    def _apply_object_patches(self):
        for k in self.object_patches:
            old = util.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

    def patch_model_flux(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        self._apply_object_patches()
        if load_weights:
            self.load(device_to, lowvram_model_memory, force_patch_weights, full_load=lowvram_model_memory == 0)
        return self.model

    def patch_model_lowvram_flux(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        return self._patch_model_lowvram_impl(device_to, lowvram_model_memory, force_patch_weights)

    def patch_model(self, device_to=None, patch_weights=True):
        self._apply_object_patches()
        if patch_weights:
            for key in self.patches:
                if key not in self.model.state_dict():
                    logging.warning(f"could not patch. key doesn't exist in model: {key}")
                    continue
                self.patch_weight_to_device(key, device_to)
            if device_to:
                self.model.to(device_to)
                self.current_device = device_to
        return self.model

    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        return self._patch_model_lowvram_impl(device_to, lowvram_model_memory, force_patch_weights)

    def _patch_model_lowvram_impl(self, device_to, lowvram_model_memory, force_patch_weights):
        self.patch_model(device_to, patch_weights=False)
        logging.info(f"loading in lowvram mode {lowvram_model_memory / 1e6:.1f}")
        mem_counter, patch_counter = 0, 0
        for n, m in self.model.named_modules():
            lowvram_weight = hasattr(m, "comfy_cast_weights") and mem_counter + Device.module_size(m) >= lowvram_model_memory
            if lowvram_weight:
                for pkey in [f"{n}.weight", f"{n}.bias"]:
                    if pkey in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(pkey)
                        else:
                            setattr(m, 'weight_function' if 'weight' in pkey else 'bias_function', LowVramPatch(pkey, self))
                            patch_counter += 1
                m.prev_comfy_cast_weights, m.comfy_cast_weights = m.comfy_cast_weights, True
            elif hasattr(m, "weight"):
                self.patch_weight_to_device(f"{n}.weight", device_to)
                self.patch_weight_to_device(f"{n}.bias", device_to)
                m.to(device_to)
                mem_counter += Device.module_size(m)
        self.model_lowvram, self.lowvram_patch_counter = True, patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha, v = p[0], p[1][1]
            mat1 = Device.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = Device.cast_to_device(v[1], weight.device, torch.float32)
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            
            patch_shape = (mat1.shape[0], mat2.shape[1])
            if patch_shape != weight.shape:
                # Handle cases where weight might be flattened but patch is not, or vice versa
                if mat1.flatten(start_dim=1).shape[0] * mat2.flatten(start_dim=1).shape[1] != weight.numel():
                    logging.warning(f"Skipping patch for {key}: shape mismatch. Weight: {weight.shape}, Patch: {patch_shape}")
                    continue
                
            try:
                weight += (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))).reshape(weight.shape).type(weight.dtype)
            except Exception as e:
                logging.error(f"Failed to apply patch for {key}: {e}")
                continue
        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for k in list(self.backup.keys()):
                util.set_attr_param(self.model, k, self.backup[k])
            self.backup.clear()
            if device_to:
                self.model.to(device_to)
                self.current_device = device_to
        self.object_patches_backup.clear()

    def partially_load(self, device_to, extra_memory=0):
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

    def apply_tome(self, ratio=0.5, max_downsample=1):
        if not TOMESD_AVAILABLE:
            logging.warning("Token Merging (tomesd) not available")
            return False
        try:
            tomesd.remove_patch(self)
        except:
            pass
        self.tome_enabled, self.tome_ratio = False, 0.5
        try:
            if hasattr(self.model, 'diffusion_model'):
                tomesd.apply_patch(self, ratio=ratio, max_downsample=max_downsample)
                self.tome_enabled, self.tome_ratio = True, ratio
                logging.info(f"Applied Token Merging with ratio={ratio}, max_downsample={max_downsample}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to apply Token Merging: {e}")
            return False

    def remove_tome(self):
        if not TOMESD_AVAILABLE or not self.tome_enabled:
            return False
        try:
            tomesd.remove_patch(self)
            self.tome_enabled, self.tome_ratio, self.tome_info = False, 0.5, {}
            return True
        except Exception as e:
            logging.error(f"Failed to remove Token Merging: {e}")
            return False


def unet_prefix_from_state_dict(state_dict):
    counts = {c: sum(1 for k in state_dict if k.startswith(c)) for c in ["model.diffusion_model.", "model.model."]}
    top = max(counts, key=counts.get)
    return top if counts[top] > 5 else "model."


def load_diffusion_model_state_dict(sd, model_options={}):
    dtype = model_options.get("dtype")
    prefix = unet_prefix_from_state_dict(sd)
    temp_sd = util.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
    parameters, load_device = util.calculate_parameters(sd), Device.get_torch_device()
    model_config = unet.model_config_from_unet(sd, "")
    offload_device = Device.unet_offload_device()
    unet_dtype2 = dtype or Device.unet_dtype(model_params=parameters, supported_dtypes=model_config.supported_inference_dtypes)
    manual_cast_dtype = Device.unet_manual_cast(unet_dtype2, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype2, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    model = model_config.get_model(sd, "").to(offload_device)
    model.load_model_weights(sd, "")
    return ModelPatcher(model, load_device=load_device, offload_device=offload_device)
