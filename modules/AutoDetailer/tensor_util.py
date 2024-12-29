import numpy as np
import torch
from PIL import Image
import torchvision

from modules.Device import Device


def _tensor_check_image(image: torch.Tensor) -> None:
    """#### Check if the input is a valid tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
    """
    return


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """#### Convert a tensor to a PIL image.

    #### Args:
        - `image` (torch.Tensor): The input tensor.

    #### Returns:
        - `Image.Image`: The converted PIL image.
    """
    _tensor_check_image(image)
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def general_tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """#### Resize a tensor image using bilinear interpolation.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `w` (int): The target width.
        - `h` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The resized tensor image.
    """
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """#### Convert a PIL image to a tensor.

    #### Args:
        - `image` (Image.Image): The input PIL image.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class TensorBatchBuilder:
    """#### Class for building a batch of tensors."""

    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def concat(self, new_tensor: torch.Tensor) -> None:
        """#### Concatenate a new tensor to the batch.

        #### Args:
            - `new_tensor` (torch.Tensor): The new tensor to concatenate.
        """
        self.tensor = new_tensor


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def tensor_resize(image: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """#### Resize a tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `w` (int): The target width.
        - `h` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The resized tensor image.
    """
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)


def tensor_paste(
    image1: torch.Tensor,
    image2: torch.Tensor,
    left_top: tuple[int, int],
    mask: torch.Tensor,
) -> None:
    """#### Paste one tensor image onto another using a mask.

    #### Args:
        - `image1` (torch.Tensor): The base tensor image.
        - `image2` (torch.Tensor): The tensor image to paste.
        - `left_top` (tuple[int, int]): The top-left corner where the image2 will be pasted.
        - `mask` (torch.Tensor): The mask tensor.
    """
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    mask = mask[:, :h, :w, :]
    image1[:, y : y + h, x : x + w, :] = (1 - mask) * image1[
        :, y : y + h, x : x + w, :
    ] + mask * image2[:, :h, :w, :]
    return


def tensor_convert_rgba(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """#### Convert a tensor image to RGBA format.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `prefer_copy` (bool, optional): Whether to prefer copying the tensor. Defaults to True.

    #### Returns:
        - `torch.Tensor`: The converted RGBA tensor image.
    """
    _tensor_check_image(image)
    alpha = torch.ones((*image.shape[:-1], 1))
    return torch.cat((image, alpha), axis=-1)


def tensor_convert_rgb(image: torch.Tensor, prefer_copy: bool = True) -> torch.Tensor:
    """#### Convert a tensor image to RGB format.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `prefer_copy` (bool, optional): Whether to prefer copying the tensor. Defaults to True.

    #### Returns:
        - `torch.Tensor`: The converted RGB tensor image.
    """
    _tensor_check_image(image)
    return image


def tensor_get_size(image: torch.Tensor) -> tuple[int, int]:
    """#### Get the size of a tensor image.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.

    #### Returns:
        - `tuple[int, int]`: The width and height of the tensor image.
    """
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor_putalpha(image: torch.Tensor, mask: torch.Tensor) -> None:
    """#### Add an alpha channel to a tensor image using a mask.

    #### Args:
        - `image` (torch.Tensor): The input tensor image.
        - `mask` (torch.Tensor): The mask tensor.
    """
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_mask(mask: torch.Tensor) -> None:
    """#### Check if the input is a valid tensor mask.

    #### Args:
        - `mask` (torch.Tensor): The input tensor mask.
    """
    return


def tensor_gaussian_blur_mask(
    mask: torch.Tensor | np.ndarray, kernel_size: int, sigma: float = 10.0
) -> torch.Tensor:
    """#### Apply Gaussian blur to a tensor mask.

    #### Args:
        - `mask` (torch.Tensor | np.ndarray): The input tensor mask.
        - `kernel_size` (int): The size of the Gaussian kernel.
        - `sigma` (float, optional): The standard deviation of the Gaussian kernel. Defaults to 10.0.

    #### Returns:
        - `torch.Tensor`: The blurred tensor mask.
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]

    _tensor_check_mask(mask)

    kernel_size = kernel_size * 2 + 1

    prev_device = mask.device
    device = Device.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(
        kernel_size=kernel_size, sigma=sigma
    )(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """#### Convert a numpy array to a tensor.

    #### Args:
        - `image` (np.ndarray): The input numpy array.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    return torch.from_numpy(image)
