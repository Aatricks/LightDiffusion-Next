import logging as logger
import torch
from PIL import Image

from src.Device import Device
from src.UltimateSDUpscale import RDRB, image_util
from src.Utilities import util


def load_state_dict(state_dict: dict) -> RDRB.PyTorchModel:
    """Load state dict into PyTorch upscale model."""
    logger.debug("Loading state dict into pytorch model arch")
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    return RDRB.RRDBNet(state_dict)


class UpscaleModelLoader:
    """Load upscale models from disk."""
    def load_model(self, model_name: str) -> tuple:
        model_path = f"./include/ESRGAN/{model_name}"
        sd = util.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = util.state_dict_prefix_replace(sd, {"module.": ""})
        return (load_state_dict(sd).eval(),)


class ImageUpscaleWithModel:
    """Upscale images using ESRGAN model."""
    def upscale(self, upscale_model: torch.nn.Module, image: torch.Tensor) -> tuple:
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        Device.get_free_memory(device)

        tile, overlap = 512, 32
        steps = in_img.shape[0] * image_util.get_tiled_scale_steps(
            in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
        s = image_util.tiled_scale(
            in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile,
            overlap=overlap, upscale_amount=upscale_model.scale, pbar=util.ProgressBar(steps))

        upscale_model.cpu()
        return (torch.clamp(s.movedim(-3, -1), min=0, max=1.0),)


def torch_gc() -> None:
    pass


class Script:
    pass


class Options:
    img2img_background_color: str = "#ffffff"


class State:
    interrupted: bool = False
    def begin(self) -> None: pass
    def end(self) -> None: pass


opts = Options()
state = State()
sd_upscalers = [None]
actual_upscaler = None
batch = None

if not hasattr(Image, "Resampling"):
    Image.Resampling = Image


class Upscaler:
    """Upscale images using loaded model."""
    def _upscale(self, img: Image.Image, scale: float) -> Image.Image:
        global actual_upscaler
        tensor = image_util.pil_to_tensor(img)
        (upscaled,) = ImageUpscaleWithModel().upscale(actual_upscaler, tensor)
        return image_util.tensor_to_pil(upscaled)

    def upscale(self, img: Image.Image, scale: float, selected_model: str = None) -> Image.Image:
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    name: str = ""
    data_path: str = ""
    def __init__(self):
        self.scaler = Upscaler()
