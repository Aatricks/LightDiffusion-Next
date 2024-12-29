import torch

from modules.StableFast.ModuleFactory import ModuleFactory

try:
    from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig
except:
    pass


def gen_stable_fast_config():
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
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function, params):
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

    def to(self, device):
        if type(device) is torch.device:
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
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
