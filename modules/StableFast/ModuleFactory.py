import contextlib
import torch
from modules.StableFast import SF_util

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
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self):
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self):
        key_kwargs = {}
        for k, v in self.converted_kwargs.items():
            key_kwargs[k] = v

        patch_module_cache_key = {}
        return (
            self.callable.__class__.__qualname__,
            SF_util.hash_arg(self.unet_config),
            SF_util.hash_arg(key_kwargs),
            SF_util.hash_arg(patch_module_cache_key),
        )

    @contextlib.contextmanager
    def converted_module_context(self):
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)
