from typing import Dict, Tuple
import torch
from src.Device import Device
from src.Utilities import util


class LatentFormat:
    """Base class for latent formats."""
    scale_factor: float = 1.0
    latent_channels: int = 4
    downscale_factor: int = 8

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale latent for input."""
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale latent for output."""
        return latent / self.scale_factor


class SD15(LatentFormat):
    """SD1.5 latent format."""
    latent_channels: int = 4

    def __init__(self, scale_factor: float = 0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            [0.3512, 0.2297, 0.3227], [0.3250, 0.4974, 0.2350],
            [-0.2829, 0.1762, 0.2721], [-0.2120, -0.2616, -0.7177],
        ]
        self.taesd_decoder_name = "taesd_decoder"


class SDXL(LatentFormat):
    """SDXL latent format."""
    latent_channels: int = 4
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            [0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
            [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188],
        ]
        self.latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]
        self.taesd_decoder_name = "taesdxl_decoder"


class SDXL_Playground_2_5(LatentFormat):
    """SDXL Playground 2.5 with mean/std normalization."""
    latent_channels: int = 4

    def __init__(self):
        self.scale_factor = 0.5
        self.latents_mean = torch.tensor([-1.6574, 1.886, -1.383, 2.5155]).view(1, 4, 1, 1)
        self.latents_std = torch.tensor([8.4927, 5.9022, 6.5498, 5.2299]).view(1, 4, 1, 1)
        self.latent_rgb_factors = [
            [0.3920, 0.4054, 0.4549], [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755], [-0.3112, -0.2359, -0.2076],
        ]
        self.taesd_decoder_name = "taesdxl_decoder"

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - mean) * self.scale_factor / std

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latents_mean.to(latent.device, latent.dtype)
        std = self.latents_std.to(latent.device, latent.dtype)
        return latent * std / self.scale_factor + mean

        
class SD3(LatentFormat):
    """SD3 latent format with shift factor."""
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609
        self.latent_rgb_factors = [
            [-0.0645, 0.0177, 0.1052], [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360], [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364], [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283], [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700], [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220], [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207], [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259],
        ]
        self.taesd_decoder_name = "taesd3_decoder"

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent / self.scale_factor) + self.shift_factor


class Flux1(SD3):
    """Flux1 latent format."""
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0404, 0.0159, 0.0609], [0.0043, 0.0298, 0.0850],
            [0.0328, -0.0749, -0.0503], [-0.0245, 0.0085, 0.0549],
            [0.0966, 0.0894, 0.0530], [0.0035, 0.0399, 0.0123],
            [0.0583, 0.1184, 0.1262], [-0.0191, -0.0206, -0.0306],
            [-0.0324, 0.0055, 0.1001], [0.0955, 0.0659, -0.0545],
            [-0.0504, 0.0231, -0.0013], [0.0500, -0.0008, -0.0088],
            [0.0982, 0.0941, 0.0976], [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020], [-0.1273, -0.0932, -0.0680],
        ]
        self.taesd_decoder_name = "taef1_decoder"


class EmptyLatentImage:
    """Generate empty latent images."""
    def __init__(self):
        self.device = Device.intermediate_device()

    def generate(self, width: int, height: int, batch_size: int = 1) -> Tuple[Dict[str, torch.Tensor]]:
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples": latent},)


def fix_empty_latent_channels(model, latent_image):
    """Fix empty latent channels to match model requirements."""
    latent_channels = model.get_model_object("latent_format").latent_channels
    if latent_channels != latent_image.shape[1] and torch.count_nonzero(latent_image) == 0:
        latent_image = util.repeat_to_batch_size(latent_image, latent_channels, dim=1)
    return latent_image
