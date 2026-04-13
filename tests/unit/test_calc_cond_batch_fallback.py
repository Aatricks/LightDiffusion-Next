import torch
import pytest

from src.cond import cond as cond_module
from src.cond.cond import CONDRegular, calc_cond_batch


class DummyModel:
    def memory_required(self, input_shape):
        # return small value so batching heuristics don't interfere
        return 1

    def apply_model(self, *args, **kwargs):
        # Should not be called if fallback is triggered
        inp = args[0] if args else kwargs.get("input")
        return inp


class RecordingDummyModel(DummyModel):
    def __init__(self):
        self.batch_sizes = []

    def apply_model(self, *args, **kwargs):
        inp = args[0] if args else kwargs.get("input")
        self.batch_sizes.append(int(inp.shape[0]))
        return inp


def test_calc_cond_batch_fallback_on_transformer_options_mismatch(monkeypatch):
    called = {"flag": False}

    def spy_run_model_per_chunk(model, x_in, timestep, input_x_list, c_list, batch_sizes, batch_indices_list, cond_or_uncond, model_options):
        called["flag"] = True
        # return zeroed outputs with the same shapes
        return [torch.zeros_like(x) for x in input_x_list]

    monkeypatch.setattr(cond_module, "_run_model_per_chunk", spy_run_model_per_chunk)


    # Create input_x shaped as 4x4 token grid
    x_in = torch.zeros((1, 128, 4, 4))

    # Build a minimal condition (actual transformer_options come from our fake_get_area_and_mult)
    cond_dict = {"model_conds": {"c_crossattn": CONDRegular(torch.zeros((1, 1, 1, 1)))}}

    # Pass conds as lists (positive and None for uncond)
    conds = [[cond_dict], None]

    # Run calc_cond_batch and ensure fallback was taken (spy was called)
    model_opts = {"transformer_options": {"img_h": 200, "img_w": 200}}
    out = calc_cond_batch(DummyModel(), conds, x_in, timestep=0, model_options=model_opts)

    assert called["flag"], "Expected _run_model_per_chunk to be called due to transformer_options mismatch"
    # out should be a list of two tensors matching input shape
    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape == x_in.shape
    assert out[1].shape == x_in.shape


def test_calc_cond_batch_honors_batched_cfg_toggle():
    x_in = torch.zeros((1, 4, 8, 8))
    cond_dict = {"model_conds": {"c_crossattn": CONDRegular(torch.zeros((1, 1, 1, 1)))}}
    conds = [[cond_dict], [cond_dict]]

    batched_model = RecordingDummyModel()
    calc_cond_batch(
        batched_model,
        conds,
        x_in,
        timestep=0,
        model_options={"batched_cfg": True},
    )

    unbatched_model = RecordingDummyModel()
    calc_cond_batch(
        unbatched_model,
        conds,
        x_in,
        timestep=0,
        model_options={"batched_cfg": False},
    )

    assert batched_model.batch_sizes == [2]
    assert unbatched_model.batch_sizes == [1, 1]
