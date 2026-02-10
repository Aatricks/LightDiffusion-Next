import torch


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Standard bilinear interpolation for latent upscaling (fallback for bislerp)."""
    return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear", align_corners=False)


def common_upscale(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Upscale samples using bilinear interpolation."""
    return bislerp(samples, width, height)


class LatentUpscale:
    """Upscale latent codes."""
    def upscale(self, samples: dict, width: int, height: int, upscale_method: str = "bislerp", 
                downscale_factor: int = 8) -> tuple:
        if width == 0 and height == 0:
            return (samples,)
        s = samples.copy()
        s["samples"] = common_upscale(samples["samples"], max(64, width) // downscale_factor, 
                                     max(64, height) // downscale_factor)
        return (s,)
