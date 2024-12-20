import math
import logging as logger
import torch

from modules.Device import Device
from modules.UltimateSDUpscale import RDRB
from modules.UltimateSDUpscale import image_util
from modules.Utilities import util


def load_state_dict(state_dict) -> RDRB.PyTorchModel:
    logger.debug(f"Loading state dict into pytorch model arch")
    state_dict_keys = list(state_dict.keys())
    if "params_ema" in state_dict_keys:
        state_dict = state_dict["params_ema"]
    model = RDRB.RRDBNet(state_dict)
    return model


class UpscaleModelLoader:
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = f"_internal/ESRGAN/{model_name}"
        sd = util.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = util.state_dict_prefix_replace(sd, {"module.": ""})
        out = load_state_dict(sd).eval()
        return (out,)




class ImageUpscaleWithModel:

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = torch.device(torch.cuda.current_device())
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        free_memory = Device.get_free_memory(device)

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


def torch_gc():
    pass


class Script:
    pass


class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
actual_upscaler = None

# Batch of images to upscale
batch = None

import numpy as np
import torch.nn.functional as F


from PIL import Image

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def _upscale(self, img: Image, scale):
        global actual_upscaler
        tensor = image_util.pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return image_util.tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
        
