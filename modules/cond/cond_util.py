from modules.Device import Device
import torch
from typing import List, Tuple, Any


def get_models_from_cond(cond: dict, model_type: str) -> List[object]:
    """#### Get models from a condition.

    #### Args:
        - `cond` (dict): The condition.
        - `model_type` (str): The model type.

    #### Returns:
        - `List[object]`: The list of models.
    """
    models = []
    for c in cond:
        if model_type in c:
            models += [c[model_type]]
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


def prepare_sampling(
    model: object, noise_shape: Tuple[int], conds: dict, flux_enabled: bool = False
) -> Tuple[object, dict, List[object]]:
    """#### Prepare the model for sampling.

    #### Args:
        - `model` (object): The model.
        - `noise_shape` (Tuple[int]): The shape of the noise.
        - `conds` (dict): The conditions.
        - `flux_enabled` (bool, optional): Whether flux is enabled. Defaults to False.

    #### Returns:
        - `Tuple[object, dict, List[object]]`: The prepared model, conditions, and additional models.
    """
    real_model = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    memory_required = (
        model.memory_required([noise_shape[0] * 2] + list(noise_shape[1:]))
        + inference_memory
    )
    minimum_memory_required = (
        model.memory_required([noise_shape[0]] + list(noise_shape[1:]))
        + inference_memory
    )
    Device.load_models_gpu(
        [model] + models,
        memory_required=memory_required,
        minimum_memory_required=minimum_memory_required,
        flux_enabled=flux_enabled,
    )
    real_model = model.model

    return real_model, conds, models

def cleanup_additional_models(models: List[object]) -> None:
    """#### Clean up additional models.
    
    #### Args:
        - `models` (List[object]): The list of models.
    """
    for m in models:
        if hasattr(m, "cleanup"):
            m.cleanup()

def cleanup_models(conds: dict, models: List[object]) -> None:
    """#### Clean up the models after sampling.

    #### Args:
        - `conds` (dict): The conditions.
        - `models` (List[object]): The list of models.
    """
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))


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
    if c1.keys() != c2.keys():
        return False
    return True


def can_concat_cond(c1: Any, c2: Any) -> bool:
    """#### Check if two conditions can be concatenated.

    #### Args:
        - `c1` (Any): The first condition.
        - `c2` (Any): The second condition.

    #### Returns:
        - `bool`: Whether the conditions can be concatenated.
    """
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        """#### Check if two objects can be concatenated."""
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list: List[dict]) -> dict:
    """#### Concatenate a list of conditions.

    #### Args:
        - `c_list` (List[dict]): The list of conditions.

    #### Returns:
        - `dict`: The concatenated conditions.
    """
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


def create_cond_with_same_area_if_none(conds: List[dict], c: dict) -> None:
    """#### Create a condition with the same area if none exists.

    #### Args:
        - `conds` (List[dict]): The list of conditions.
        - `c` (dict): The condition.
    """
    if "area" not in c:
        return

    c_area = c["area"]
    smallest = None
    for x in conds:
        if "area" in x:
            a = x["area"]
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif "area" not in smallest:
                            smallest = x
                        else:
                            if smallest["area"][0] * smallest["area"][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if "area" in smallest:
        if smallest["area"] == c_area:
            return

    out = c.copy()
    out["model_conds"] = smallest[
        "model_conds"
    ].copy()
    conds += [out]
