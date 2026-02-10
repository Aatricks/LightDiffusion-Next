import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor as tv_to_tensor

from src.Device import Device


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert tensor [B,H,W,C] or [H,W,C] to PIL image using torchvision."""
    if image.dim() == 4:
        image = image[0]  # Take first from batch
    # HWC -> CHW for torchvision
    if image.shape[-1] in [1, 3, 4]:
        image = image.permute(2, 0, 1)
    return to_pil_image(torch.clamp(image, 0, 1))


def general_tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Resize tensor using bilinear interpolation. Expects [B,H,W,C]."""
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    return image.permute(0, 2, 3, 1)


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor [1,H,W,C] using torchvision."""
    return tv_to_tensor(image).unsqueeze(0).permute(0, 2, 3, 1)


class TensorBatchBuilder:
    """Utility for building a batch of tensors by concatenation."""
    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def concat(self, new_tensor: torch.Tensor) -> None:
        self.tensor = new_tensor if self.tensor is None else torch.cat([self.tensor, new_tensor], dim=0)


LANCZOS = Image.Resampling.LANCZOS


def tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Resize tensor [B,H,W,C] using LANCZOS (3+ channels) or bilinear."""
    if image.shape[3] >= 3:
        scaled = TensorBatchBuilder()
        for single in image:
            pil = tensor2pil(single.unsqueeze(0))
            scaled.concat(pil2tensor(pil.resize((w, h), resample=LANCZOS)))
        return scaled.tensor
    return general_tensor_resize(image, w, h)


def tensor_paste(
    image1: torch.Tensor,
    image2: torch.Tensor,
    left_top: tuple[int, int],
    mask: torch.Tensor,
) -> None:
    """Paste image2 onto image1 at left_top position using mask."""
    x, y = [int(round(c)) for c in left_top]
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape
    w, h = min(w1, x + w2) - x, min(h1, y + h2) - y
    
    # Ensure all tensors are on the same device as image1
    device = image1.device
    mask = mask[:, :h, :w, :].to(device)
    image2 = image2[:, :h, :w, :].to(device)
    
    image1[:, y:y+h, x:x+w, :] = (1 - mask) * image1[:, y:y+h, x:x+w, :] + mask * image2


def tensor_convert_rgba(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """Add alpha channel (ones) to tensor."""
    return torch.cat((image, torch.ones((*image.shape[:-1], 1))), axis=-1)


def tensor_convert_rgb(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """Return image unchanged (already RGB)."""
    return image


def tensor_get_size(image: torch.Tensor) -> tuple[int, int]:
    """Return (width, height) of tensor [B,H,W,C]."""
    _, h, w, _ = image.shape
    return (w, h)


def tensor_putalpha(image: torch.Tensor, mask: torch.Tensor) -> None:
    """Set alpha channel from mask."""
    image[..., -1] = mask[..., 0]


def tensor_gaussian_blur_mask(
    mask: torch.Tensor | np.ndarray, kernel_size: int, sigma: float = 10.0
) -> torch.Tensor:
    """Apply Gaussian blur to mask using torchvision."""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.ndim == 2:
        mask = mask[None, ..., None]
    
    device = Device.get_torch_device()
    mask = mask[:, None, ..., 0].to(device)
    blurred = torchvision.transforms.GaussianBlur(kernel_size=kernel_size*2+1, sigma=sigma)(mask)
    return blurred[:, 0, ..., None]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy array to tensor."""
    return torch.from_numpy(image)
