from dataclasses import dataclass
import logging
import torch
import functools

from modules.StableFast.ModuleFactory import ModuleFactory


logger = logging.getLogger()

try:
    from sfast.compilers.diffusion_pipeline_compiler import (
        _enable_xformers,
        _modify_model,
    )
    from sfast.cuda.graphs import make_dynamic_graphed_callable
    from sfast.jit import utils as jit_utils
    from sfast.jit.trace_helper import trace_with_kwargs
except ImportError:
    pass


@dataclass
class TracedModuleCacheItem:
    """#### Data class for storing traced module cache items.

    #### Attributes:
        - `module` (object): The traced module.
        - `patch_id` (int): The patch ID.
        - `device` (str): The device.
    """
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    """#### Class for lazy tracing of modules."""

    traced_modules: dict = {}

    def __init__(self, config: object = None, patch_id: int = None, **kwargs) -> None:
        """#### Initialize the LazyTraceModule.

        #### Args:
            - `config` (object, optional): The configuration object. Defaults to None.
            - `patch_id` (int, optional): The patch ID. Defaults to None.
            - `**kwargs`: Additional keyword arguments.
        """
        self.config = config
        self.patch_id = patch_id
        self.kwargs = kwargs
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )
        self.cuda_graph_modules = {}

    def ts_compiler(self, m: torch.nn.Module) -> torch.nn.Module:
        """#### TorchScript compiler for the module.

        #### Args:
            - `m` (torch.nn.Module): The module to compile.

        #### Returns:
            - `torch.nn.Module`: The compiled module.
        """
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function: callable, **kwargs) -> callable:
        """#### Call the LazyTraceModule.

        #### Args:
            - `model_function` (callable): The model function.
            - `**kwargs`: Additional keyword arguments.

        #### Returns:
            - `callable`: The traced module.
        """
        module_factory = ModuleFactory.BaseModelApplyModelModuleFactory(
            model_function, kwargs
        )
        kwargs = module_factory.get_converted_kwargs()
        key = module_factory.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None:
            with module_factory.converted_module_context() as (m_model, m_kwargs):
                logger.info(
                    f'Tracing {getattr(m_model, "__name__", m_model.__class__.__name__)}'
                )
                traced_m, call_helper = trace_with_kwargs(
                    m_model, None, m_kwargs, **self.kwargs
                )

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            self.cuda_graph_modules[key] = traced_module

        return traced_module(**kwargs)


def build_lazy_trace_module(config: object, device: torch.device, patch_id: int) -> LazyTraceModule:
    """#### Build a LazyTraceModule.

    #### Args:
        - `config` (object): The configuration object.
        - `device` (torch.device): The device.
        - `patch_id` (int): The patch ID.

    #### Returns:
        - `LazyTraceModule`: The LazyTraceModule instance.
    """
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )