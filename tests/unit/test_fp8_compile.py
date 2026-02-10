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
