import contextlib
import torch
from modules.StableFast import SF_util


class ModuleFactory:
    """#### Base class for module factories."""

    def get_converted_kwargs(self) -> dict:
        """#### Get the converted keyword arguments.

        #### Returns:
            - `dict`: The converted keyword arguments.
        """
        return self.converted_kwargs


class BaseModelApplyModelModule(torch.nn.Module):
    """#### Module for applying a model function."""

    def __init__(self, func: callable, module: torch.nn.Module):
        """#### Initialize the BaseModelApplyModelModule.

        #### Args:
            - `func` (callable): The function to apply.
            - `module` (torch.nn.Module): The module to apply the function to.
        """
        super().__init__()
        self.func = func
        self.module = module

    def forward(
        self,
        input_x: torch.Tensor,
        timestep: torch.Tensor,
        c_concat: torch.Tensor = None,
        c_crossattn: torch.Tensor = None,
        y: torch.Tensor = None,
        control: torch.Tensor = None,
        transformer_options: dict = {},
    ) -> torch.Tensor:
        """#### Forward pass of the module.

        #### Args:
            - `input_x` (torch.Tensor): The input tensor.
            - `timestep` (torch.Tensor): The timestep tensor.
            - `c_concat` (torch.Tensor, optional): The concatenated conditioning tensor. Defaults to None.
            - `c_crossattn` (torch.Tensor, optional): The cross-attention conditioning tensor. Defaults to None.
            - `y` (torch.Tensor, optional): The target tensor. Defaults to None.
            - `control` (torch.Tensor, optional): The control tensor. Defaults to None.
            - `transformer_options` (dict, optional): The transformer options. Defaults to {}.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
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
    """#### Factory for creating BaseModelApplyModelModule instances."""

    kwargs_name = (
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    )

    def __init__(self, callable: callable, kwargs: dict) -> None:
        """#### Initialize the BaseModelApplyModelModuleFactory.

        #### Args:
            - `callable` (callable): The callable to use.
            - `kwargs` (dict): The keyword arguments.
        """
        self.callable = callable
        self.unet_config = callable.__self__.model_config.unet_config
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}
        self.converted_kwargs = self.gen_converted_kwargs()

    def gen_converted_kwargs(self) -> dict:
        """#### Generate the converted keyword arguments.

        #### Returns:
            - `dict`: The converted keyword arguments.
        """
        converted_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            if arg_name in self.kwargs_name:
                converted_kwargs[arg_name] = arg

        transformer_options = self.kwargs.get("transformer_options", {})
        transformer_options.get("patches", {})

        patch_module = {}
        patch_module_parameter = {}

        new_transformer_options = {}
        new_transformer_options["patches"] = patch_module_parameter

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return converted_kwargs

    def gen_cache_key(self) -> tuple:
        """#### Generate a cache key.

        #### Returns:
            - `tuple`: The cache key.
        """
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
        """#### Context manager for the converted module.

        #### Yields:
            - `tuple`: The module and the converted keyword arguments.
        """
        module = BaseModelApplyModelModule(self.callable, self.callable.__self__)
        yield (module, self.converted_kwargs)