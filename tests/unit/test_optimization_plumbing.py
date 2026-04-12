import torch

from src.Device import Device
from src.Model.ModelPatcher import ModelPatcher


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
