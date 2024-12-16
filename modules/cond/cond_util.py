from modules import Device


def get_models_from_cond(cond, model_type):
    models = []
    return models

def get_additional_models(conds, dtype):
    """loads additional _internal in conditioning"""
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


def prepare_sampling(model, noise_shape, conds):
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


def cleanup_models(conds, models):
    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    return True


def can_concat_cond(c1, c2):
    return cond_equal_size(c1.conditioning, c2.conditioning)


def cond_cat(c_list):
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

def resolve_areas_and_cond_masks(conditions, h, w, device):
    for i in range(len(conditions)):
        c = conditions[i]


def create_cond_with_same_area_if_none(conds, c):
    if "area" not in c:
        return