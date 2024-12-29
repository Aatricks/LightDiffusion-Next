from modules.Device import Device
import torch
from typing import List, Dict, Tuple, Any

def get_models_from_cond(cond: dict, model_type: str) -> List[object]:
    """#### Get models from a condition.

    #### Args:
        - `cond` (dict): The condition.
        - `model_type` (str): The model type.

    #### Returns:
        - `List[object]`: The list of models.
    """
    models = []
    return models

def get_additional_models(conds: dict, dtype: torch.dtype) -> Tuple[List[object], int]:
    """#### Load additional models in conditioning.

    #### Args:
        - `conds` (dict): The conditions.
        - `dtype` (torch.dtype): The data type.

    #### Returns:
        - `Tuple[List[object], int]`: The list of models and the inference memory.
    """
    cnets = []
    gligen = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen
    return models, inference_memory

def prepare_sampling(model: object, noise_shape: Tuple[int], conds: dict) -> Tuple[object, dict, List[object]]:
    """#### Prepare the model for sampling.

    #### Args:
        - `model` (object): The model.
        - `noise_shape` (Tuple[int]): The shape of the noise.
        - `conds` (dict): The conditions.

    #### Returns:
        - `Tuple[object, dict, List[object]]`: The prepared model, conditions, and additional models.
    """
    device = model.load_device
    real_model = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    Device.load_models_gpu(
        [model] + models,
        model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]))
        + inference_memory,
    )
    real_model = model.model

    return real_model, conds, models

def cleanup_models(conds: dict, models: List[object]) -> None:
    """#### Clean up the models after sampling.

    #### Args:
        - `conds` (dict): The conditions.
        - `models` (List[object]): The list of models.
    """
    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

def cond_equal_size(c1: Any, c2: Any) -> bool:
    """#### Check if two conditions have equal size.

    #### Args:
        - `c1` (Any): The first condition.
        - `c2` (Any): The second condition.

    #### Returns:
        - `bool`: Whether the conditions have equal size.
    """
    if c1 is c2:
        return True
    return True

def can_concat_cond(c1: Any, c2: Any) -> bool:
    """#### Check if two conditions can be concatenated.

    #### Args:
        - `c1` (Any): The first condition.
        - `c2` (Any): The second condition.

    #### Returns:
        - `bool`: Whether the conditions can be concatenated.
    """
    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list: List[dict]) -> dict:
    """#### Concatenate a list of conditions.

    #### Args:
        - `c_list` (List[dict]): The list of conditions.

    #### Returns:
        - `dict`: The concatenated conditions.
    """
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def resolve_areas_and_cond_masks(conditions: List[dict], h: int, w: int, device: torch.device) -> None:
    """#### Resolve areas and condition masks.

    #### Args:
        - `conditions` (List[dict]): The list of conditions.
        - `h` (int): The height.
        - `w` (int): The width.
        - `device` (torch.device): The device.
    """
    for i in range(len(conditions)):
        c = conditions[i]

def create_cond_with_same_area_if_none(conds: List[dict], c: dict) -> None:
    """#### Create a condition with the same area if none exists.

    #### Args:
        - `conds` (List[dict]): The list of conditions.
        - `c` (dict): The condition.
    """
    if "area" not in c:
        return