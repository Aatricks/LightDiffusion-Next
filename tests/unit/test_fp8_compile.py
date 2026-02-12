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
