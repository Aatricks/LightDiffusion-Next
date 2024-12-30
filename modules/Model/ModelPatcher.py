import copy
import logging
import uuid

import torch

from modules.Utilities import util
from modules.Device import Device


class ModelPatcher:
    def __init__(
        self,
        model: torch.nn.Module,
        load_device: torch.device,
        offload_device: torch.device,
        size: int = 0,
        current_device: torch.device = None,
        weight_inplace_update: bool = False,
    ):
        """#### Initialize the ModelPatcher class.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `load_device` (torch.device): The device to load the model on.
            - `offload_device` (torch.device): The device to offload the model to.
            - `size` (int, optional): The size of the model. Defaults to 0.
            - `current_device` (torch.device, optional): The current device. Defaults to None.
            - `weight_inplace_update` (bool, optional): Whether to update weights in place. Defaults to False.
        """
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

    def model_size(self) -> int:
        """#### Get the size of the model.

        #### Returns:
            - `int`: The size of the model.
        """
        if self.size > 0:
            return self.size
        model_sd = self.model.state_dict()
        self.size = Device.module_size(self.model)
        self.model_keys = set(model_sd.keys())
        return self.size

    def clone(self) -> "ModelPatcher":
        """#### Clone the ModelPatcher object.

        #### Returns:
            - `ModelPatcher`: The cloned ModelPatcher object.
        """
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

    def is_clone(self, other: object) -> bool:
        """#### Check if the object is a clone.

        #### Args:
            - `other` (object): The other object.

        #### Returns:
            - `bool`: Whether the object is a clone.
        """
        if hasattr(other, "model") and self.model is other.model:
            return True
        return False

    def memory_required(self, input_shape: tuple) -> float:
        """#### Calculate the memory required for the model.

        #### Args:
            - `input_shape` (tuple): The input shape.

        #### Returns:
            - `float`: The memory required.
        """
        return self.model.memory_required(input_shape=input_shape)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: callable) -> None:
        """#### Set the UNet function wrapper for the model.

        #### Args:
            - `unet_wrapper_function` (callable): The UNet function wrapper.
        """
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function: callable) -> None:
        """#### Set the denoise mask function for the model.

        #### Args:
            - `denoise_mask_function` (callable): The denoise mask function.
        """
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def get_model_object(self, name: str) -> object:
        """#### Get an object from the model.

        #### Args:
            - `name` (str): The name of the object.

        #### Returns:
            - `object`: The object.
        """
        return util.get_attr(self.model, name)

    def model_patches_to(self, device: torch.device) -> None:
        """#### Move model patches to a device.

        #### Args:
            - `device` (torch.device): The device.
        """
        self.model_options["transformer_options"]
        if "model_function_wrapper" in self.model_options:
            wrap_func = self.model_options["model_function_wrapper"]
            if hasattr(wrap_func, "to"):
                self.model_options["model_function_wrapper"] = wrap_func.to(device)

    def model_dtype(self) -> torch.dtype:
        """#### Get the data type of the model.

        #### Returns:
            - `torch.dtype`: The data type.
        """
        if hasattr(self.model, "get_dtype"):
            return self.model.get_dtype()

    def add_patches(
        self, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> list:
        """#### Add patches to the model.

        #### Args:
            - `patches` (dict): The patches to add.
            - `strength_patch` (float, optional): The strength of the patches. Defaults to 1.0.
            - `strength_model` (float, optional): The strength of the model. Defaults to 1.0.

        #### Returns:
            - `list`: The list of patched keys.
        """
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
        """#### Set a patch for the model.

        #### Args:
            - `patch` (list): The patch.
            - `name` (str): The name of the patch.
        """
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_attn1_patch(self, patch: list):
        """#### Set the attention 1 patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch: list):
        """#### Set the attention 2 patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn2_patch")
        
    def set_model_attn1_output_patch(self, patch: list):
        """#### Set the attention 1 output patch for the model.

        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch: list):
        """#### Set the attention 2 output patch for the model.
        
        #### Args:
            - `patch` (list): The patch.
        """
        self.set_model_patch(patch, "attn2_output_patch")
    
    def model_state_dict(self, filter_prefix: str = None) -> dict:
        """#### Get the state dictionary of the model.

        #### Args:
            - `filter_prefix` (str, optional): The prefix to filter. Defaults to None.

        #### Returns:
            - `dict`: The state dictionary.
        """
        sd = self.model.state_dict()
        list(sd.keys())
        return sd

    def patch_weight_to_device(self, key: str, device_to: torch.device = None) -> None:
        """#### Patch the weight of a key to a device.

        #### Args:
            - `key` (str): The key.
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
        """
        if key not in self.patches:
            return

        weight = util.get_attr(self.model, key)

        inplace_update = self.weight_inplace_update

        if key not in self.backup:
            self.backup[key] = weight.to(
                device=self.offload_device, copy=inplace_update
            )

        if device_to is not None:
            temp_weight = Device.cast_to_device(
                weight, device_to, torch.float32, copy=True
            )
        else:
            temp_weight = weight.to(torch.float32, copy=True)
        out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(
            weight.dtype
        )
        if inplace_update:
            util.copy_to_param(self.model, key, out_weight)
        else:
            util.set_attr_param(self.model, key, out_weight)

    def patch_model(
        self, device_to: torch.device = None, patch_weights: bool = True
    ) -> torch.nn.Module:
        """#### Patch the model.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `patch_weights` (bool, optional): Whether to patch weights. Defaults to True.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        for k in self.object_patches:
            old = util.set_attr(self.model, k, self.object_patches[k])
            if k not in self.object_patches_backup:
                self.object_patches_backup[k] = old

        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    logging.warning(
                        "could not patch. key doesn't exist in model: {}".format(key)
                    )
                    continue

                self.patch_weight_to_device(key, device_to)

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def patch_model_lowvram(
        self,
        device_to: torch.device = None,
        lowvram_model_memory: int = 0,
        force_patch_weights: bool = False,
    ) -> torch.nn.Module:
        """#### Patch the model for low VRAM.

        #### Args:
            - `device_to` (torch.device, optional): The device to patch to. Defaults to None.
            - `lowvram_model_memory` (int, optional): The low VRAM model memory. Defaults to 0.
            - `force_patch_weights` (bool, optional): Whether to force patch weights. Defaults to False.

        #### Returns:
            - `torch.nn.Module`: The patched model.
        """
        self.patch_model(device_to, patch_weights=False)

        logging.info(
            "loading in lowvram mode {}".format(lowvram_model_memory / (1024 * 1024))
        )

        class LowVramPatch:
            def __init__(self, key: str, model_patcher: "ModelPatcher"):
                self.key = key
                self.model_patcher = model_patcher

            def __call__(self, weight: torch.Tensor) -> torch.Tensor:
                return self.model_patcher.calculate_weight(
                    self.model_patcher.patches[self.key], weight, self.key
                )

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

    def calculate_weight(
        self, patches: list, weight: torch.Tensor, key: str
    ) -> torch.Tensor:
        """#### Calculate the weight of a key.

        #### Args:
            - `patches` (list): The list of patches.
            - `weight` (torch.Tensor): The weight tensor.
            - `key` (str): The key.

        #### Returns:
            - `torch.Tensor`: The calculated weight.
        """
        for p in patches:
            alpha = p[0]
            v = p[1]
            p[2]
            v[0]
            v = v[1]
            mat1 = Device.cast_to_device(v[0], weight.device, torch.float32)
            mat2 = Device.cast_to_device(v[1], weight.device, torch.float32)
            v[4]
            if v[2] is not None:
                alpha *= v[2] / mat2.shape[0]
            weight += (
                (alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
                .reshape(weight.shape)
                .type(weight.dtype)
            )
        return weight

    def unpatch_model(
        self, device_to: torch.device = None, unpatch_weights: bool = True
    ) -> None:
        """#### Unpatch the model.

        #### Args:
            - `device_to` (torch.device, optional): The device to unpatch to. Defaults to None.
            - `unpatch_weights` (bool, optional): Whether to unpatch weights. Defaults to True.
        """
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
