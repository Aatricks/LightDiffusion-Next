import torch
from modules.StableFast.ModuleFactory import ModuleFactory

try:
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
except ImportError:
    pass

# Taken from https://github.com/gameltb/ComfyUI_stable_fast


def gen_stable_fast_config() -> CompilationConfig:
    """#### Generate the StableFast configuration.

    #### Returns:
        - `CompilationConfig`: The StableFast configuration.
    """
    config = CompilationConfig.Default()
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = False
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    """#### Class representing a StableFast patch."""

    def __init__(self, model: torch.nn.Module, config: CompilationConfig):
        """#### Initialize the StableFastPatch.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `config` (CompilationConfig): The configuration.
        """
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function: callable, params: dict) -> torch.Tensor:
        """#### Call the StableFastPatch.

        #### Args:
            - `model_function` (callable): The model function.
            - `params` (dict): The parameters.

        #### Returns:
            - `torch.Tensor`: The output tensor.
        """
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = ModuleFactory.build_lazy_trace_module(
                self.config,
                input_x.device,
                id(self),
            )

        return self.stable_fast_model(
            model_function, input_x=input_x, timestep=timestep_, **c
        )

    def to(self, device: torch.device) -> StableFastPatch:
        """#### Move the model to a specific device.

        #### Args:
            - `device` (torch.device): The device.

        #### Returns:
            - `StableFastPatch`: The StableFastPatch instance.
        """
        if isinstance(device, torch.device):
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. If you experience a noticeable delay every time you start sampling, please consider disabling enable_cuda_graph.\33[0m"
                    )
        return self


class ApplyStableFastUnet:
    """#### Class for applying StableFast to a UNet model."""

    def apply_stable_fast(self, model: torch.nn.Module, enable_cuda_graph: bool) -> tuple:
        """#### Apply StableFast to the model.

        #### Args:
            - `model` (torch.nn.Module): The model.
            - `enable_cuda_graph` (bool): Whether to enable CUDA graph.

        #### Returns:
            - `tuple`: The StableFast model.
        """
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)