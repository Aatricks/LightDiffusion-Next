from typing import Dict, Tuple
import torch
from modules.Device import Device
from modules.Utilities import util

class LatentFormat:
    """#### Base class for latent formats.

    #### Attributes:
        - `scale_factor` (float): The scale factor for the latent format.

    #### Returns:
        - `LatentFormat`: A latent format object.
    """

    scale_factor: float = 1.0

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent input, by multiplying it by the scale factor.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """#### Process the latent output, by dividing it by the scale factor.

        #### Args:
            - `latent` (torch.Tensor): The latent tensor.

        #### Returns:
            - `torch.Tensor`: The processed latent tensor.
        """
        return latent / self.scale_factor

class SD15(LatentFormat):
    """#### SD15 latent format.

    #### Args:
        - `LatentFormat` (LatentFormat): The base latent format class.
    """

    def __init__(self, scale_factor: float = 0.18215):
        """#### Initialize the SD15 latent format.

        #### Args:
            - `scale_factor` (float, optional): The scale factor. Defaults to 0.18215.
        """
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [0.3512, 0.2297, 0.3227],
            [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721],
            [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"
        
class SD3(LatentFormat):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645, 0.0177, 0.1052],
            [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360],
            [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364],
            [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283],
            [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700],
            [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220],
            [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259],
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class Flux1(SD3):
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609],
            [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503],
            [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530],
            [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262],
            [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001],
            [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013],
            [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976],
            [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020],
            [-0.1273, -0.0932, -0.0680],
        ]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

class EmptyLatentImage:
    """#### A class to generate an empty latent image.

    #### Args:
        - `Device` (Device): The device to use for the latent image.
    """

    def __init__(self):
        """#### Initialize the EmptyLatentImage class."""
        self.device = Device.intermediate_device()

    def generate(
        self, width: int, height: int, batch_size: int = 1
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """#### Generate an empty latent image

        #### Args:
            - `width` (int): The width of the latent image.
            - `height` (int): The height of the latent image.
            - `batch_size` (int, optional): The batch size. Defaults to 1.

        #### Returns:
            - `Tuple[Dict[str, torch.Tensor]]`: The generated latent image.
        """
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return ({"samples": latent},)

def fix_empty_latent_channels(model, latent_image):
    latent_channels = model.get_model_object(
        "latent_format"
    ).latent_channels  # Resize the empty latent image so it has the right number of channels
    if (
        latent_channels != latent_image.shape[1]
        and torch.count_nonzero(latent_image) == 0
    ):
        latent_image = util.repeat_to_batch_size(latent_image, latent_channels, dim=1)
    return latent_image