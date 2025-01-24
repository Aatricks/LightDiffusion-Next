import copy
import logging
import gguf
import torch

from modules.Device import Device
from modules.Model import ModelPatcher
from modules.Utilities import util
from modules.clip import Clip
from modules.cond import cast

# Constants for torch-compatible quantization types
TORCH_COMPATIBLE_QTYPES = {
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
}


def is_torch_compatible(tensor: torch.Tensor) -> bool:
    """#### Check if a tensor is compatible with PyTorch operations.

    #### Args:
        - `tensor` (torch.Tensor): The tensor to check.

    #### Returns:
        - `bool`: Whether the tensor is torch-compatible.
    """
    return (
        tensor is None
        or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES
    )


def is_quantized(tensor: torch.Tensor) -> bool:
    """#### Check if a tensor is quantized.

    #### Args:
        - `tensor` (torch.Tensor): The tensor to check.

    #### Returns:
        - `bool`: Whether the tensor is quantized.
    """
    return not is_torch_compatible(tensor)


def dequantize(
    data: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    oshape: tuple,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """#### Dequantize tensor back to usable shape/dtype.

    #### Args:
        - `data` (torch.Tensor): The quantized data.
        - `qtype` (gguf.GGMLQuantizationType): The quantization type.
        - `oshape` (tuple): The output shape.
        - `dtype` (torch.dtype, optional): The output dtype. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The dequantized tensor.
    """
    # Get block size and type size for quantization format
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    # Reshape data into blocks
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))

    # Dequantize blocks and reshape to target shape
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def split_block_dims(blocks: torch.Tensor, *args) -> list:
    """#### Split blocks into dimensions.

    #### Args:
        - `blocks` (torch.Tensor): The blocks to split.
        - `*args`: The dimensions to split into.

    #### Returns:
        - `list`: The split blocks.
    """
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


# Legacy Quantization Functions
def dequantize_blocks_Q8_0(
    blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype = None
) -> torch.Tensor:
    """#### Dequantize Q8_0 quantized blocks.

    #### Args:
        - `blocks` (torch.Tensor): The quantized blocks.
        - `block_size` (int): The block size.
        - `type_size` (int): The type size.
        - `dtype` (torch.dtype, optional): The output dtype. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The dequantized blocks.
    """
    # Split blocks into scale and quantized values
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

# Mapping of quantization types to dequantization functions
dequantize_functions = {
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
}


def dequantize_tensor(
    tensor: torch.Tensor, dtype: torch.dtype = None, dequant_dtype: torch.dtype = None
) -> torch.Tensor:
    """#### Dequantize a potentially quantized tensor.

    #### Args:
        - `tensor` (torch.Tensor): The tensor to dequantize.
        - `dtype` (torch.dtype, optional): Target dtype. Defaults to None.
        - `dequant_dtype` (torch.dtype, optional): Intermediate dequantization dtype. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The dequantized tensor.
    """
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)


class GGMLLayer(torch.nn.Module):
    """#### Base class for GGML quantized layers.

    Handles dynamic dequantization of weights during forward pass.
    """

    comfy_cast_weights: bool = True
    dequant_dtype: torch.dtype = None
    patch_dtype: torch.dtype = None
    torch_compatible_tensor_types: set = {
        None,
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
    }

    def is_ggml_quantized(
        self, *, weight: torch.Tensor = None, bias: torch.Tensor = None
    ) -> bool:
        """#### Check if layer weights are GGML quantized.

        #### Args:
            - `weight` (torch.Tensor, optional): Weight tensor to check. Defaults to self.weight.
            - `bias` (torch.Tensor, optional): Bias tensor to check. Defaults to self.bias.

        #### Returns:
            - `bool`: Whether weights are quantized.
        """
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *args, **kwargs
    ) -> None:
        """#### Load quantized weights from state dict.

        #### Args:
            - `state_dict` (dict): State dictionary.
            - `prefix` (str): Key prefix.
            - `*args`: Additional arguments.
            - `**kwargs`: Additional keyword arguments.
        """
        weight = state_dict.get(f"{prefix}weight")
        bias = state_dict.get(f"{prefix}bias")
        # Use modified loader for quantized or linear layers
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(
            self, torch.nn.Linear
        ):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """#### Load GGML quantized weights from state dict.

        #### Args:
            - `state_dict` (dict): State dictionary.
            - `prefix` (str): Key prefix.
            - `local_metadata` (dict): Local metadata.
            - `strict` (bool): Strict loading mode.
            - `missing_keys` (list): Keys missing from state dict.
            - `unexpected_keys` (list): Unexpected keys found.
            - `error_msgs` (list): Error messages.
        """
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, *args, **kwargs) -> None:
        """#### Save layer state to state dict.

        #### Args:
            - `*args`: Additional arguments.
            - `**kwargs`: Additional keyword arguments.
        """
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(
        self, destination: dict, prefix: str, keep_vars: bool
    ) -> None:
        """#### Save GGML layer state to state dict.

        #### Args:
            - `destination` (dict): Destination dictionary.
            - `prefix` (str): Key prefix.
            - `keep_vars` (bool): Whether to keep variables.
        """
        # Create fake tensors for VRAM estimation
        weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias
        return

    def get_weight(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """#### Get dequantized weight tensor.

        #### Args:
            - `tensor` (torch.Tensor): Input tensor.
            - `dtype` (torch.dtype): Target dtype.

        #### Returns:
            - `torch.Tensor`: Dequantized tensor.
        """
        if tensor is None:
            return

        # Consolidate and load patches to GPU asynchronously
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # Dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # Apply patches
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                # For testing, may degrade image quality
                patch_dtype = (
                    dtype if self.patch_dtype == "target" else self.patch_dtype
                )
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    def cast_bias_weight(
        self,
        input: torch.Tensor = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        bias_dtype: torch.dtype = None,
    ) -> tuple:
        """#### Cast layer weights and bias to target dtype/device.

        #### Args:
            - `input` (torch.Tensor, optional): Input tensor for type/device inference.
            - `dtype` (torch.dtype, optional): Target dtype.
            - `device` (torch.device, optional): Target device.
            - `bias_dtype` (torch.dtype, optional): Target bias dtype.

        #### Returns:
            - `tuple`: (cast_weight, cast_bias)
        """
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = Device.device_supports_non_blocking(device)
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = cast.cast_to(
                bias, bias_dtype, device, non_blocking=non_blocking, copy=False
            )

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = cast.cast_to(
            weight, dtype, device, non_blocking=non_blocking, copy=False
        )
        return weight, bias

    def forward_comfy_cast_weights(
        self, input: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """#### Forward pass with weight casting.

        #### Args:
            - `input` (torch.Tensor): Input tensor.
            - `*args`: Additional arguments.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `torch.Tensor`: Output tensor.
        """
        if self.is_ggml_quantized():
            return self.forward_ggml_cast_weights(input, *args, **kwargs)
        return super().forward_comfy_cast_weights(input, *args, **kwargs)


class GGMLOps(cast.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """

    class Linear(GGMLLayer, cast.manual_cast.Linear):
        def __init__(
            self, in_features, out_features, bias=True, device=None, dtype=None
        ):
            """
            Initialize the Linear layer.

            Args:
                in_features (int): Number of input features.
                out_features (int): Number of output features.
                bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
                device (torch.device, optional): The device to store the layer's parameters. Defaults to None.
                dtype (torch.dtype, optional): The data type of the layer's parameters. Defaults to None.
            """
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input: torch.Tensor) -> torch.Tensor:
            """
            Forward pass with GGML cast weights.

            Args:
                input (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
            """
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Embedding(GGMLLayer, cast.manual_cast.Embedding):
        def forward_ggml_cast_weights(
            self, input: torch.Tensor, out_dtype: torch.dtype = None
        ) -> torch.Tensor:
            """
            Forward pass with GGML cast weights for embedding.

            Args:
                input (torch.Tensor): The input tensor.
                out_dtype (torch.dtype, optional): The output data type. Defaults to None.

            Returns:
                torch.Tensor: The output tensor.
            """
            output_dtype = out_dtype
            if (
                self.weight.dtype == torch.float16
                or self.weight.dtype == torch.bfloat16
            ):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(
                self, device=input.device, dtype=out_dtype
            )
            return torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)


def gguf_sd_loader_get_orig_shape(
    reader: gguf.GGUFReader, tensor_name: str
) -> torch.Size:
    """#### Get the original shape of a tensor from a GGUF reader.

    #### Args:
        - `reader` (gguf.GGUFReader): The GGUF reader.
        - `tensor_name` (str): The name of the tensor.

    #### Returns:
        - `torch.Size`: The original shape of the tensor.
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}"
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        """
        Initialize the GGMLTensor.

        Args:
            *args: Variable length argument list.
            tensor_type: The type of the tensor.
            tensor_shape: The shape of the tensor.
            patches (list, optional): List of patches. Defaults to [].
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        """
        Create a new instance of GGMLTensor.

        Args:
            *args: Variable length argument list.
            tensor_type: The type of the tensor.
            tensor_shape: The shape of the tensor.
            patches (list, optional): List of patches. Defaults to [].
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: A new instance of GGMLTensor.
        """
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        """
        Convert the tensor to a specified device and/or dtype.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: The converted tensor.
        """
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        """
        Clone the tensor.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: The cloned tensor.
        """
        return self

    def detach(self, *args, **kwargs):
        """
        Detach the tensor from the computation graph.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: The detached tensor.
        """
        return self

    def copy_(self, *args, **kwargs):
        """
        Copy the values from another tensor into this tensor.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: The tensor with copied values.
        """
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")

    def __deepcopy__(self, *args, **kwargs):
        """
        Create a deep copy of the tensor.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGMLTensor: The deep copied tensor.
        """
        new = super().__deepcopy__(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    @property
    def shape(self):
        """
        Get the shape of the tensor.

        Returns:
            torch.Size: The shape of the tensor.
        """
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape


def gguf_sd_loader(path: str, handle_prefix: str = "model.diffusion_model."):
    """#### Load a GGUF file into a state dict.

    #### Args:
        - `path` (str): The path to the GGUF file.
        - `handle_prefix` (str, optional): The prefix to handle. Defaults to "model.diffusion_model.".

    #### Returns:
        - `dict`: The loaded state dict.
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = None
    arch_field = reader.get_field("general.architecture")
    if arch_field is not None:
        if (
            len(arch_field.types) != 1
            or arch_field.types[0] != gguf.GGUFValueType.STRING
        ):
            raise TypeError(
                f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}"
            )
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
        if arch_str not in {"flux", "sd1", "sdxl", "t5", "t5encoder"}:
            raise ValueError(
                f"Unexpected architecture type in GGUF file, expected one of flux, sd1, sdxl, t5encoder but got {arch_str!r}"
            )

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        tensor_type_str = str(tensor.tensor_type)
        torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = gguf_sd_loader_get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any(
                    [
                        tensor_name.endswith(x)
                        for x in (".proj_in.weight", ".proj_out.weight")
                    ]
                ):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape
        )
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k, v in qtype_dict.items():
        print(f" {k:30}{v:3}")

    return state_dict


class GGUFModelPatcher(ModelPatcher.ModelPatcher):
    patch_on_device = False

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """
        Unpatch the model.

        Args:
            device_to (torch.device, optional): The device to move the model to. Defaults to None.
            unpatch_weights (bool, optional): Whether to unpatch the weights. Defaults to True.

        Returns:
            GGUFModelPatcher: The unpatched model.
        """
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        self.object_patches = {}
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(
            device_to=device_to, unpatch_weights=unpatch_weights
        )

    mmap_released = False

    def load(self, *args, force_patch_weights=False, **kwargs):
        """
        Load the model.

        Args:
            *args: Variable length argument list.
            force_patch_weights (bool, optional): Whether to force patch weights. Defaults to False.
            **kwargs: Arbitrary keyword arguments.
        """
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def add_object_patch(self, name, obj):
            self.object_patches[name] = obj

    def clone(self, *args, **kwargs):
        """
        Clone the model patcher.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            GGUFModelPatcher: The cloned model patcher.
        """
        n = GGUFModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n


class UnetLoaderGGUF:
    def load_unet(
        self,
        unet_name: str,
        dequant_dtype: str = None,
        patch_dtype: str = None,
        patch_on_device: bool = None,
    ) -> tuple:
        """
        Load the UNet model.

        Args:
            unet_name (str): The name of the UNet model.
            dequant_dtype (str, optional): The dequantization data type. Defaults to None.
            patch_dtype (str, optional): The patch data type. Defaults to None.
            patch_on_device (bool, optional): Whether to patch on device. Defaults to None.

        Returns:
            tuple: The loaded model.
        """
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        unet_path = "./_internal/unet/" + unet_name
        sd = gguf_sd_loader(unet_path)
        model = ModelPatcher.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError(
                "ERROR: Could not detect model type of: {}".format(unet_path)
            )
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)


clip_sd_map = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

clip_name_dict = {
    "stable_diffusion": Clip.CLIPType.STABLE_DIFFUSION,
    "sdxl": Clip.CLIPType.STABLE_DIFFUSION,
    "sd3": Clip.CLIPType.SD3,
    "flux": Clip.CLIPType.FLUX,
}


def gguf_clip_loader(path: str) -> dict:
    """#### Load a CLIP model from a GGUF file.

    #### Args:
        - `path` (str): The path to the GGUF file.

    #### Returns:
        - `dict`: The loaded CLIP model.
    """
    raw_sd = gguf_sd_loader(path)
    assert "enc.blk.23.ffn_up.weight" in raw_sd, "Invalid Text Encoder!"
    sd = {}
    for k, v in raw_sd.items():
        for s, d in clip_sd_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


class CLIPLoaderGGUF:
    def load_data(self, ckpt_paths: list) -> list:
        """
        Load data from checkpoint paths.

        Args:
            ckpt_paths (list): List of checkpoint paths.

        Returns:
            list: List of loaded data.
        """
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                clip_data.append(gguf_clip_loader(p))
            else:
                sd = util.load_torch_file(p, safe_load=True)
                clip_data.append(
                    {
                        k: GGMLTensor(
                            v,
                            tensor_type=gguf.GGMLQuantizationType.F16,
                            tensor_shape=v.shape,
                        )
                        for k, v in sd.items()
                    }
                )
        return clip_data

    def load_patcher(self, clip_paths: list, clip_type: str, clip_data: list) -> Clip:
        """
        Load the model patcher.

        Args:
            clip_paths (list): List of clip paths.
            clip_type (str): The type of the clip.
            clip_data (list): List of clip data.

        Returns:
            Clip: The loaded clip.
        """
        clip = Clip.load_text_encoder_state_dicts(
            clip_type=clip_type,
            state_dicts=clip_data,
            model_options={
                "custom_operations": GGMLOps,
                "initial_device": Device.text_encoder_offload_device(),
            },
            embedding_directory="models/embeddings",
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        # for some reason this is just missing in some SAI checkpoints
        if getattr(clip.cond_stage_model, "clip_l", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_l.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_l.transformer.text_projection = (
                    cast.manual_cast.Linear(768, 768)
                )
        if getattr(clip.cond_stage_model, "clip_g", None) is not None:
            if (
                getattr(
                    clip.cond_stage_model.clip_g.transformer.text_projection.weight,
                    "tensor_shape",
                    None,
                )
                is None
            ):
                clip.cond_stage_model.clip_g.transformer.text_projection = (
                    cast.manual_cast.Linear(1280, 1280)
                )

        return clip


class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    def load_clip(self, clip_name1: str, clip_name2: str, type: str) -> tuple:
        """
        Load dual clips.

        Args:
            clip_name1 (str): The name of the first clip.
            clip_name2 (str): The name of the second clip.
            type (str): The type of the clip.

        Returns:
            tuple: The loaded clips.
        """
        clip_path1 = "./_internal/clip/" + clip_name1
        clip_path2 = "./_internal/clip/" + clip_name2
        clip_paths = (clip_path1, clip_path2)
        clip_type = clip_name_dict.get(type, Clip.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)


class CLIPTextEncodeFlux:
    def encode(
        self,
        clip: Clip,
        clip_l: str,
        t5xxl: str,
        guidance: str,
        flux_enabled: bool = False,
    ) -> tuple:
        """
        Encode text using CLIP and T5XXL.

        Args:
            clip (Clip): The clip object.
            clip_l (str): The clip text.
            t5xxl (str): The T5XXL text.
            guidance (str): The guidance text.
            flux_enabled (bool, optional): Whether flux is enabled. Defaults to False.

        Returns:
            tuple: The encoded text.
        """
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        output = clip.encode_from_tokens(
            tokens, return_pooled=True, return_dict=True, flux_enabled=flux_enabled
        )
        cond = output.pop("cond")
        output["guidance"] = guidance
        return ([[cond, output]],)


class ConditioningZeroOut:
    def zero_out(self, conditioning: list) -> list:
        """
        Zero out the conditioning.

        Args:
            conditioning (list): The conditioning list.

        Returns:
            list: The zeroed out conditioning.
        """
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c,)
