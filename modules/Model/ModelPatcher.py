import copy
import logging
import uuid

import torch

from modules.Utilities import util
from modules.Device import Device



class ModelPatcher:
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        current_device=None,
        weight_inplace_update=False,
    ):
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
        if current_device is None:
            self.current_device = self.offload_device
        else:
            self.current_device = current_device

        self.weight_inplace_update = weight_inplace_update
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.patches_uuid = uuid.uuid4()

    def model_size(self):
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = Device.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self):
        n = ModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            self.current_device,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.model_keys = self.model_keys
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        return n

    def is_clone(self, other):
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape):
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, unet_wrapper_function):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name):
        return util.get_attr(self.model, name)

    def model_patches_to(self, device):
        to = self.model_options["transformer_options"]
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self):
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        self.patches_uuid = uuid.uuid4()
        return list(p)

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        return sd

    def patch_weight_to_device(self, key, device_to=None):
        if key not in self.patches:
            return

        weight = util.get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

        if device_to is not None:
            temp_weight = Device.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
        if inplace_update:
            util.copy_to_param(self.model, key, out_weight)
        else:
            util.set_attr_param(self.model, key, out_weight)
    
    def patch_model(self, device_to=None, patch_weights=True):
        for k in self.object_patches:
            old = util.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning("could not patch. key doesn't exist in model: {}".format(key))
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False):
        self.patch_model(device_to, patch_weights=False)

        logging.info("loading in lowvram mode {}".format(lowvram_model_memory/(1024 * 1024)))
        class LowVramPatch:
            def __init__(self, key, model_patcher):
                self.key = key
                self.model_patcher = model_patcher
            def __call__(self, weight):
                return self.model_patcher.calculate_weight(self.model_patcher.patches[self.key], weight, self.key)

        mem_counter = 0
        patch_counter = 0
        for n, m in self.model.named_modules():
            lowvram_weight = False
            if hasattr(m, "comfy_cast_weights"):
                module_mem = Device.module_size(m)
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = LowVramPatch(weight_key, self)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = LowVramPatch(bias_key, self)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "weight"):
                    self.patch_weight_to_device(weight_key, device_to)
                    self.patch_weight_to_device(bias_key, device_to)
                    m.to(device_to)
                    mem_counter += Device.module_size(m)
                    logging.debug("lowvram: loaded module regularly {}".format(m))

        self.model_lowvram = True
        self.lowvram_patch_counter = patch_counter
        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]
            patch_type = v[0]
            v = v[1]
            mat1 = Device.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = Device.cast_to_device(v[1], weight.device, torch.float32)
            dora_scale = v[4]
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            weight += (
                (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
                .reshape(weight.shape)
                .type(weight.dtype)
            )
        return weight

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            keys = list(self.backup.keys())
            for k in keys:
                util.set_attr_param(self.model, k, self.backup[k])
            self.backup.clear()
            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        keys = list(self.object_patches_backup.keys())
        self.object_patches_backup.clear()