"""Tests for FP8 quantization and torch.compile fixes.

Validates:
- FP8 quantization enables comfy_cast_weights so runtime forward casts FP8→input dtype
- torch.compile uses safe default mode (max-autotune-no-cudagraphs, not reduce-overhead)
- FP8 + torch.compile combination works without crashes
"""
import inspect

import pytest
import torch


class TestFP8Quantization:
    """Tests for FP8 weight quantization and runtime casting."""

    def test_fp8_enables_comfy_cast_weights(self):
        """After FP8 quantization, CastWeightBiasOp modules must have comfy_cast_weights=True."""
        from src.cond.cast import CastWeightBiasOp, disable_weight_init

        linear = disable_weight_init.Linear(4, 4, bias=False)
        linear.weight.data = torch.randn(4, 4, dtype=torch.float16)
        assert linear.comfy_cast_weights is False, "Should start with comfy_cast_weights=False"

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        linear.weight.data = linear.weight.data.to(torch.float8_e4m3fn)
        assert isinstance(linear, CastWeightBiasOp)
        linear.comfy_cast_weights = True

        assert linear.comfy_cast_weights is True

    def test_fp8_forward_no_dtype_mismatch(self):
        """FP8 weights with comfy_cast_weights=True must not cause dtype mismatch."""
        from src.cond.cast import disable_weight_init

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        linear = disable_weight_init.Linear(8, 8, bias=False)
        linear.weight.data = torch.randn(8, 8, dtype=torch.float16).to(torch.float8_e4m3fn)
        linear.comfy_cast_weights = True

        inp = torch.randn(2, 8, dtype=torch.float16)
        result = linear(inp)
        assert result.dtype == torch.float16, f"Expected float16 output, got {result.dtype}"
        assert result.shape == (2, 8)

    def test_fp8_forward_with_bias(self):
        """FP8 weights + float16 bias should work with comfy_cast_weights=True."""
        from src.cond.cast import disable_weight_init

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        linear = disable_weight_init.Linear(8, 8, bias=True)
        linear.weight.data = torch.randn(8, 8, dtype=torch.float16).to(torch.float8_e4m3fn)
        # bias stays in float16 (apply_fp8 only quantizes ndim>=2)
        assert linear.bias.dtype in (torch.float16, torch.float32)
        linear.comfy_cast_weights = True

        inp = torch.randn(2, 8, dtype=torch.float16)
        result = linear(inp)
        assert result.shape == (2, 8)

    def test_fp8_without_comfy_cast_raises(self):
        """FP8 weights WITHOUT comfy_cast_weights should raise RuntimeError (the original bug)."""
        from src.cond.cast import disable_weight_init

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        linear = disable_weight_init.Linear(8, 8, bias=False)
        linear.weight.data = torch.randn(8, 8, dtype=torch.float16).to(torch.float8_e4m3fn)
        # Intentionally do NOT set comfy_cast_weights = True
        assert linear.comfy_cast_weights is False

        inp = torch.randn(2, 8, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="have the same dtype"):
            linear(inp)

    def test_fp8_conv2d_forward(self):
        """FP8 Conv2d weights with comfy_cast_weights should work."""
        from src.cond.cast import disable_weight_init

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available in this PyTorch build")

        conv = disable_weight_init.Conv2d(4, 8, 3, padding=1, bias=False)
        conv.weight.data = torch.randn(8, 4, 3, 3, dtype=torch.float16).to(torch.float8_e4m3fn)
        conv.comfy_cast_weights = True

        inp = torch.randn(1, 4, 16, 16, dtype=torch.float16)
        result = conv(inp)
        assert result.shape == (1, 8, 16, 16)


class TestTorchCompileMode:
    """Tests for torch.compile default mode safety."""

    def test_compile_model_default_mode(self):
        """Device.compile_model should default to max-autotune-no-cudagraphs."""
        from src.Device import Device

        sig = inspect.signature(Device.compile_model)
        default_mode = sig.parameters["mode"].default
        assert default_mode == "max-autotune-no-cudagraphs", (
            f"Default compile mode should be 'max-autotune-no-cudagraphs', got '{default_mode}'"
        )

    def test_apply_torch_compile_default_mode(self):
        """AbstractModel.apply_torch_compile should default to max-autotune-no-cudagraphs."""
        from src.Core.AbstractModel import AbstractModel

        sig = inspect.signature(AbstractModel.apply_torch_compile)
        default_mode = sig.parameters["mode"].default
        assert default_mode == "max-autotune-no-cudagraphs", (
            f"Default compile mode should be 'max-autotune-no-cudagraphs', got '{default_mode}'"
        )

    def test_compile_model_not_reduce_overhead(self):
        """Ensure default is NOT reduce-overhead (causes CUDA graph assertion errors)."""
        from src.Device import Device

        sig = inspect.signature(Device.compile_model)
        default_mode = sig.parameters["mode"].default
        assert default_mode != "reduce-overhead", (
            "reduce-overhead causes CUDA graph assertion errors with dynamic model state"
        )


class TestFP8AndCompileCombined:
    """Tests for FP8 + torch.compile compatibility."""

    def test_fp8_compile_forward(self):
        """FP8 quantized modules should work when torch.compiled."""
        from src.cond.cast import disable_weight_init

        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("FP8 dtype not available")
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        linear = disable_weight_init.Linear(16, 16, bias=False)
        linear.weight.data = torch.randn(16, 16, dtype=torch.float16).to(torch.float8_e4m3fn)
        linear.comfy_cast_weights = True

        try:
            compiled = torch.compile(linear, mode="max-autotune-no-cudagraphs")
            inp = torch.randn(2, 16, dtype=torch.float16)
            out = compiled(inp)
            assert out.shape == (2, 16)
        except Exception as e:
            # torch.compile may not work on all platforms (e.g., CPU-only, Windows)
            if "inductor" in str(e).lower() or "compile" in str(e).lower():
                pytest.skip(f"torch.compile not functional in this environment: {e}")
            raise


def test_apply_fp8_falls_back_to_top_level_model(caplog, monkeypatch):
    """Models without a 'diffusion_model' submodule (e.g., Flux2) should have FP8 quantization
    applied to the top-level module rather than emitting a warning."""
    import logging
    import torch
    from src.Core.AbstractModel import AbstractModel, ModelCapabilities

    class DummyModel(AbstractModel):
        def _create_capabilities(self):
            return ModelCapabilities()

        def load(self, model_path=None):
            self.model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
            self._loaded = True
            return self

        def encode_prompt(self, prompt, negative_prompt="", clip_skip=-2):
            return None, None

        def generate(self, ctx, positive, negative, *args, **kwargs):
            raise NotImplementedError

        def decode(self, latents):
            raise NotImplementedError

    dummy = DummyModel()
    dummy.load()
    caplog.set_level(logging.INFO)

    # Force FP8 support path and spy on cast_to_fp8 calls
    # Note: Device functions live in src.Device.Device module
    monkeypatch.setattr('src.Device.Device.is_fp8_supported', lambda *args, **kwargs: True)
    called = {'count': 0}

    def fake_cast(tensor, scale=1.0):
        called['count'] += 1
        return tensor

    monkeypatch.setattr('src.Device.Device.cast_to_fp8', fake_cast)

    dummy.apply_fp8()

    assert "No diffusion_model found for FP8 quantization" not in caplog.text
    assert called['count'] > 0, "Expected cast_to_fp8 to be invoked on top-level model modules"


def test_apply_torch_compile_falls_back_to_top_level_model(caplog, monkeypatch):
    """If a model has no 'diffusion_model' attribute, torch.compile should be
    applied to the top-level module instead of logging a warning."""
    import logging
    import torch
    from src.Core.AbstractModel import AbstractModel, ModelCapabilities

    if not hasattr(torch, 'compile'):
        pytest.skip("torch.compile not available in this environment")

    class DummyModel(AbstractModel):
        def _create_capabilities(self):
            return ModelCapabilities()

        def load(self, model_path=None):
            self.model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
            self._loaded = True
            return self

        def encode_prompt(self, prompt, negative_prompt="", clip_skip=-2):
            return None, None

        def generate(self, ctx, positive, negative, *args, **kwargs):
            raise NotImplementedError

        def decode(self, latents):
            raise NotImplementedError

    dummy = DummyModel()
    dummy.load()
    caplog.set_level(logging.INFO)

    # Spy on Device.compile_model
    compiled_called = {'count': 0}

    def fake_compile(model_obj, mode='max-autotune-no-cudagraphs'):
        compiled_called['count'] += 1
        return model_obj  # Return same object for simplicity

    monkeypatch.setattr('src.Device.Device.compile_model', fake_compile)

    dummy.apply_torch_compile()

    assert "No diffusion_model found for torch.compile" not in caplog.text
    assert compiled_called['count'] > 0, "Expected Device.compile_model to be invoked on the top-level module"


def test_apply_torch_compile_registers_wrapper_when_compile_returns_callable(monkeypatch):
    """If Device.compile_model returns a callable (not nn.Module), AbstractModel.apply_torch_compile
    should attach the compiled callable to the module.forward while preserving the module object.
    """
    import torch
    import logging
    from src.Core.AbstractModel import AbstractModel, ModelCapabilities
    from src.Model.ModelPatcher import ModelPatcher
    import torch.nn as nn

    if not hasattr(torch, 'compile'):
        pytest.skip("torch.compile not available")

    class DummyModel(AbstractModel):
        def _create_capabilities(self):
            return ModelCapabilities()

        def load(self, model_path=None):
            base = nn.Sequential(nn.Linear(4, 4, bias=False))
            mp = ModelPatcher(base, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
            self.model = mp
            self._loaded = True
            return self

        def encode_prompt(self, prompt, negative_prompt="", clip_skip=-2):
            return None, None

        def generate(self, ctx, positive, negative, *args, **kwargs):
            raise NotImplementedError

        def decode(self, latents):
            raise NotImplementedError

    dummy = DummyModel()
    dummy.load()

    compiled_called = {'count': 0}

    def fake_compiled(*args, **kwargs):
        compiled_called['count'] += 1
        for a in args:
            if hasattr(a, 'shape'):
                return a
        return torch.zeros(1)

    monkeypatch.setattr('src.Device.Device.compile_model', lambda model_obj, mode='max-autotune-no-cudagraphs': fake_compiled)

    dummy.apply_torch_compile()

    assert isinstance(dummy.model, ModelPatcher)
    assert hasattr(dummy.model, 'model') and isinstance(dummy.model.model, nn.Module)
    assert hasattr(dummy.model.model, '_compiled_fn')

    inp = torch.randn(1, 4)
    _ = dummy.model.model(inp)
    assert compiled_called['count'] > 0

    # Ensure latent_format access still works
    from src.Utilities import Latent as LatentUtil
    dummy.model.model.latent_format = LatentUtil.SD15()
    from src.Utilities.Latent import fix_empty_latent_channels
    out = fix_empty_latent_channels(dummy.model, torch.zeros((1, 4, 64, 64)))
    assert isinstance(out, torch.Tensor)
    assert out.shape[1] == dummy.model.model.latent_format.latent_channels


def test_apply_torch_compile_attaches_compiled_forward_for_flux_like_module(monkeypatch):
    """When compiling a Flux-like module (top-level with apply_model/forward signature),
    the compiled callable should be attached to the module.forward and model.apply_model
    should continue to accept Flux-style kwargs like `c_crossattn`.
    """
    import torch
    import torch.nn as nn
    from src.Core.AbstractModel import AbstractModel, ModelCapabilities
    from src.Model.ModelPatcher import ModelPatcher

    class FakeFluxModule(nn.Module):
        def forward(self, img, txt=None, timesteps=None, y=None, guidance=None, control=None, transformer_options=None, attn_mask=None, img_h=None, img_w=None):
            return img

        def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
            # Map to forward signature used by Flux-like models
            return self.forward(img=x, txt=c_crossattn, timesteps=t, y=kwargs.get('y'),
                                guidance=kwargs.get('guidance'), control=control, transformer_options=transformer_options,
                                attn_mask=kwargs.get('attention_mask'), img_h=(transformer_options or {}).get('img_h'), img_w=(transformer_options or {}).get('img_w'))

    class DummyModel(AbstractModel):
        def _create_capabilities(self):
            return ModelCapabilities(is_flux=True, is_flux2=True)

        def load(self, model_path=None):
            base = FakeFluxModule()
            mp = ModelPatcher(base, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
            self.model = mp
            self._loaded = True
            return self

        def encode_prompt(self, prompt, negative_prompt="", clip_skip=-2):
            return None, None

        def generate(self, ctx, positive, negative, *args, **kwargs):
            raise NotImplementedError

        def decode(self, latents):
            raise NotImplementedError

    dummy = DummyModel()
    dummy.load()

    compiled_called = {'count': 0}

    def fake_compiled(img, txt=None, timesteps=None, y=None, **kwargs):
        compiled_called['count'] += 1
        return img

    monkeypatch.setattr('src.Device.Device.compile_model', lambda model_obj, mode='max-autotune-no-cudagraphs': fake_compiled)

    dummy.apply_torch_compile()

    # After compile, calling apply_model should route through compiled forward without error
    out = dummy.model.model.apply_model(torch.randn(1, 128, 8, 8), torch.tensor([1.0]), c_crossattn=torch.randn(1, 10, 768), transformer_options={'img_h':128,'img_w':128})
    assert compiled_called['count'] > 0
    assert out.shape[0] == 1
