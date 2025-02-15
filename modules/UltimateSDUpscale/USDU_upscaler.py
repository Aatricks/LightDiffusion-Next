import logging as logger
import torch
from PIL import Image

from modules.Device import Device
from modules.UltimateSDUpscale import RDRB
from modules.UltimateSDUpscale import image_util
from modules.Utilities import util


def load_state_dict(state_dict: dict) -> RDRB.PyTorchModel:
    """#### Load a state dictionary into a PyTorch model.

    #### Args:
        - `state_dict` (dict): The state dictionary.

    #### Returns:
        - `RDRB.PyTorchModel`: The loaded PyTorch model.
    """
    logger.debug("Loading state dict into pytorch model arch")
    state_dict_keys = list(state_dict.keys())
    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    model = RDRB.RRDBNet(state_dict)
    return model


class UpscaleModelLoader:
    """#### Class for loading upscale models."""

    def load_model(self, model_name: str) -> tuple:
        """#### Load an upscale model.

        #### Args:
            - `model_name` (str): The name of the model.

        #### Returns:
            - `tuple`: The loaded model.
        """
        model_path = f"./_internal/ESRGAN/{model_name}"
        sd = util.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = util.state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)


class ImageUpscaleWithModel:
    """#### Class for upscaling images with a model."""

    def upscale(self, upscale_model: torch.nn.Module, image: torch.Tensor) -> tuple:
        """#### Upscale an image using a model.

        #### Args:
            - `upscale_model` (torch.nn.Module): The upscale model.
            - `image` (torch.Tensor): The input image tensor.

        #### Returns:
            - `tuple`: The upscaled image tensor.
        """
        if torch.cuda.is_available():
            device = torch.device(torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        Device.get_free_memory(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            steps = in_img.shape[0] * image_util.get_tiled_scale_steps(
                in_img.shape[3],
                in_img.shape[2],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = util.ProgressBar(steps)
            s = image_util.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False

        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return (s,)


def torch_gc() -> None:
    """#### Perform garbage collection for PyTorch."""
    pass


class Script:
    """#### Class representing a script."""
    pass


class Options:
    """#### Class representing options."""

    img2img_background_color: str = "#ffffff"  # Set to white for now


class State:
    """#### Class representing the state."""

    interrupted: bool = False

    def begin(self) -> None:
        """#### Begin the state."""
        pass

    def end(self) -> None:
        """#### End the state."""
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
actual_upscaler = None

# Batch of images to upscale
batch = None


if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:
    """#### Class for upscaling images."""

    def _upscale(self, img: Image.Image, scale: float) -> Image.Image:
        """#### Upscale an image.

        #### Args:
            - `img` (Image.Image): The input image.
            - `scale` (float): The scale factor.

        #### Returns:
            - `Image.Image`: The upscaled image.
        """
        global actual_upscaler
        tensor = image_util.pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return image_util.tensor_to_pil(upscaled)

    def upscale(self, img: Image.Image, scale: float, selected_model: str = None) -> Image.Image:
        """#### Upscale an image with a selected model.

        #### Args:
            - `img` (Image.Image): The input image.
            - `scale` (float): The scale factor.
            - `selected_model` (str, optional): The selected model. Defaults to None.

        #### Returns:
            - `Image.Image`: The upscaled image.
        """
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    """#### Class for storing upscaler data."""

    name: str = ""
    data_path: str = ""

    def __init__(self):
        self.scaler = Upscaler()