import contextlib
import functools
import importlib.util
import logging
import os
import traceback
from dataclasses import dataclass

import torch

try:
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
    from sfast.compilers.diffusion_pipeline_compiler import (
        _enable_xformers,
        _modify_model,
    )
    from sfast.cuda.graphs import make_dynamic_graphed_callable
    from sfast.jit import utils as jit_utils
    from sfast.jit.trace_helper import trace_with_kwargs
except Exception:  # pragma: no cover - sfast optional dependency
    CompilationConfig = None
    _enable_xformers = None
    _modify_model = None
    make_dynamic_graphed_callable = None
    jit_utils = None
    trace_with_kwargs = None


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)


class ModuleFactory:
    def get_converted_kwargs(self):
        return self.converted_kwargs


class BaseModelApplyModelModule(torch.nn.Module):
    def __init__(self, func, module):
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x,
        timestep,
        c_concat=None,
        c_crossattn=None,
        y=None,
        control=None,
        transformer_options={},
    ):
        kwargs = {"y": y}

        new_transformer_options = {}

        return self.func(
            input_x,
            timestep,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            control=control,
            transformer_options=new_transformer_options,
            **kwargs,
        )


class BaseModelApplyModelModuleFactory(ModuleFactory):
    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable, kwargs) -> None:
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg
        return converted_kwargs

    def gen_cache_key(self):
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            key_kwargs[k] = v

        patch_module_cache_key = {}
        return (
            self.callable.__class__.__qualname__,
            hash_arg(self.unet_config),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)


logger = logging.getLogger()


@dataclass
class TracedModuleCacheItem:
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    traced_modules = {}
    cuda_graph_modules = {}

    def __init__(self, config=None, patch_id=None, **kwargs_) -> None:
        self.config = config
        self.patch_id = patch_id
        self.kwargs_ = kwargs_
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )

    def ts_compiler(
        self,
        m,
    ):
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                # raw freeze causes Tensor reference leak
                # because the constant Tensors in the GraphFunction of
                # the compilation unit are never freed.
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function, /, **kwargs):
        module_factory = BaseModelApplyModelModuleFactory(model_function, kwargs)
        kwargs = module_factory.get_converted_kwargs()
        key = module_factory.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                logger.info(
                    f'Tracing {getattr(m_model, "__name__", m_model.__class__.__name__)}'
                )

                tensor_debug = os.getenv("STABLE_FAST_DEBUG_TENSOR", "0") == "1"
                if tensor_debug:
                    original_tensor = torch.tensor

                    def _debug_tensor(*args, **kwargs):
                        stack = "".join(traceback.format_stack(limit=8))
                        logger.warning(
                            "Stable Fast trace hit torch.tensor with args=%s kwargs=%s\n%s",
                            args,
                            kwargs,
                            stack,
                        )
                        return original_tensor(*args, **kwargs)

                    torch.tensor = _debug_tensor

                try:
                    traced_m, call_helper = trace_with_kwargs(
                        m_model, None, m_kwargs, **self.kwargs_
                    )
                finally:
                    if tensor_debug:
                        torch.tensor = original_tensor

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            self.cuda_graph_modules[key] = traced_module

        return traced_module(**kwargs)


def build_lazy_trace_module(config, device, patch_id):
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )


def gen_stable_fast_config(enable_cuda_graph=False):
    """
    Generate a StableFast compilation config, enabling or disabling CUDA graph
    compilation based on the `enable_cuda_graph` parameter.
    If the optional sfast package is not available, a fallback config is returned.
    """
    if CompilationConfig is None:
        logger.warning("StableFast: optional 'sfast' dependency not available; using fallback no-op config")

        class _FallbackConfig:
            def __init__(self):
                self.enable_xformers = False
                self.enable_cuda_graph = False
                self.enable_jit_freeze = False
                self.enable_cnn_optimization = False
                self.prefer_lowp_gemm = False
                self.enable_triton = False
                self.memory_format = None

        return _FallbackConfig()

    config = CompilationConfig.Default()
    if importlib.util.find_spec("xformers") is not None:
        config.enable_xformers = True
    else:
        print("xformers not installed, skip")

    # CUDA Graph is suggested for small batch sizes and can improve performance.
    # When enabled, the model is specialized for a fixed image size after capture.
    config.enable_cuda_graph = enable_cuda_graph
    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device):
        if isinstance(device, torch.device):
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disable enable_cuda_graph.\33[0m"
                    )
        return self


class ApplyStableFastUnet:
    def apply_stable_fast(self, model, enable_cuda_graph):
        # If the compilation components are not present, skip applying
        # the StableFast transformation and return the original model
        # in a single-element tuple so callers can index [0] as usual.
        if CompilationConfig is None or trace_with_kwargs is None or _modify_model is None:
            logger.warning(
                "StableFast.apply_stable_fast: sfast optional components missing; skipping stable-fast patch and returning original model"
            )
            return (model,)

        config = gen_stable_fast_config(enable_cuda_graph)

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
