import torch

from src.Device import Device
from src.Model.ModelPatcher import ModelPatcher
from src.cond import cond_util
from src.sample.CFG import CFGGuider


class DummyDiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def memory_required(self, input_shape=None):
        return 1


def test_model_function_wrappers_compose_in_application_order():
    patcher = ModelPatcher(
        DummyDiffusionModel(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    call_order = []

    def wrapper_one(model_function, params):
        call_order.append("wrapper_one_before")
        out = model_function(params["input"], params["timestep"], **params["c"])
        call_order.append("wrapper_one_after")
        return out + 1

    def wrapper_two(model_function, params):
        call_order.append("wrapper_two_before")
        out = model_function(params["input"], params["timestep"], **params["c"])
        call_order.append("wrapper_two_after")
        return out * 2

    patcher.set_model_unet_function_wrapper(wrapper_one)
    patcher.set_model_unet_function_wrapper(wrapper_two)

    wrapped = patcher.model_options["model_function_wrapper"]

    def base_model_function(input_x, timestep, **c_kwargs):
        call_order.append("base")
        return input_x + c_kwargs["bias"]

    result = wrapped(
        base_model_function,
        {
            "input": torch.tensor([1.0]),
            "timestep": torch.tensor([0.0]),
            "c": {"bias": torch.tensor([3.0])},
        },
    )

    assert torch.equal(result, torch.tensor([10.0]))
    assert call_order == [
        "wrapper_two_before",
        "wrapper_one_before",
        "base",
        "wrapper_one_after",
        "wrapper_two_after",
    ]


def test_sageattention_enabled_allows_compute_12_when_available(monkeypatch):
    monkeypatch.setattr(Device, "cpu_state", Device.CPUState.GPU)
    monkeypatch.setattr(Device, "directml_enabled", False)
    monkeypatch.setattr(Device, "SAGEATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(Device, "SPARGEATTN_IS_AVAILABLE", True)
    monkeypatch.setattr(Device, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(Device, "is_rocm", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (12, 0))

    assert Device.sageattention_enabled() is True
    assert Device.spargeattn_enabled() is False


def test_cfg_guider_reads_model_options_from_wrapped_model():
    patcher = ModelPatcher(
        DummyDiffusionModel(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    patcher.model_options["sentinel"] = "wrapped"

    class Wrapper:
        def __init__(self, model):
            self.model = model
            self.load_device = torch.device("cpu")

    guider = CFGGuider(Wrapper(patcher))

    assert guider.model_options["sentinel"] == "wrapped"


def test_prepare_sampling_accepts_wrapper_objects(monkeypatch):
    patcher = ModelPatcher(
        DummyDiffusionModel(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    conds = {"positive": [], "negative": []}
    load_calls = []

    monkeypatch.setattr(
        Device,
        "load_models_gpu",
        lambda models, memory_required, minimum_memory_required, force_full_load=False: load_calls.append(
            {
                "models": models,
                "memory_required": memory_required,
                "minimum_memory_required": minimum_memory_required,
                "force_full_load": force_full_load,
            }
        ),
    )

    class Wrapper:
        def __init__(self, model):
            self.model = model
            self.load_device = torch.device("cpu")

    real_model, returned_conds, loaded_models = cond_util.prepare_sampling(
        Wrapper(patcher),
        noise_shape=(1, 4, 8, 8),
        conds=conds,
    )

    assert real_model is patcher.model
    assert returned_conds is conds
    assert loaded_models == []
    assert load_calls[0]["models"][0] is patcher


def test_prepare_sampling_keeps_direct_patcher_instead_of_unwrapping_to_raw_module(monkeypatch):
    patcher = ModelPatcher(
        DummyDiffusionModel(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    conds = {"positive": [], "negative": []}
    load_calls = []

    monkeypatch.setattr(
        Device,
        "load_models_gpu",
        lambda models, memory_required, minimum_memory_required, force_full_load=False: load_calls.append(models),
    )

    real_model, returned_conds, loaded_models = cond_util.prepare_sampling(
        patcher,
        noise_shape=(1, 4, 8, 8),
        conds=conds,
    )

    assert real_model is patcher.model
    assert returned_conds is conds
    assert loaded_models == []
    assert load_calls[0][0] is patcher


def test_cfg_guider_sample_uses_wrapped_model_load_device(monkeypatch):
    patcher = ModelPatcher(
        DummyDiffusionModel(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )

    class Wrapper:
        def __init__(self, model):
            self.model = model

    guider = CFGGuider(Wrapper(patcher))
    guider.original_conds = {"positive": [], "negative": []}

    monkeypatch.setattr(
        cond_util,
        "prepare_sampling",
        lambda model, noise_shape, conds: (patcher, conds, []),
    )

    captured = {}

    def fake_inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, pipeline=False):
        captured["device"] = device
        return noise

    guider.inner_sample = fake_inner_sample

    class DummyCache:
        def prevent_model_cleanup(self, conds, loaded_models):
            return None

    monkeypatch.setattr("src.Device.ModelCache.get_model_cache", lambda: DummyCache())

    output = guider.sample(
        noise=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        latent_image=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        sampler=None,
        sigmas=torch.zeros((1,), dtype=torch.float32),
    )

    assert torch.equal(output, torch.zeros((1, 4, 8, 8), dtype=torch.float32))
    assert captured["device"] == torch.device("cpu")
