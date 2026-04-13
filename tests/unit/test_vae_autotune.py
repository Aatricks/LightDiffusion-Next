import types

import torch

from src.AutoEncoders import VariationalAE


def test_vae_autotune_gates_torch_compile(monkeypatch):
    calls = []
    compiled_decoder = object()

    def fake_compile(module, **kwargs):
        calls.append((module, kwargs))
        return compiled_decoder

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    vae = object.__new__(VariationalAE.VAE)
    vae.first_stage_model = types.SimpleNamespace(decoder=object())
    vae._compiled_decoder = False
    vae._autotune_enabled = False

    vae._ensure_compiled()
    assert calls == []
    assert vae._compiled_decoder is False

    vae.set_autotune_enabled(True)
    vae._ensure_compiled()
    vae._ensure_compiled()

    assert len(calls) == 1
    assert vae.first_stage_model.decoder is compiled_decoder
    assert vae._compiled_decoder is True
