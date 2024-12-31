import math
import numpy as np
import torch
from PIL import Image


def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int) -> int:
    """#### Calculate the number of steps required for tiled scaling.

    #### Args:
        - `width` (int): The width of the image.
        - `height` (int): The height of the image.
        - `tile_x` (int): The width of each tile.
        - `tile_y` (int): The height of each tile.
        - `overlap` (int): The overlap between tiles.

    #### Returns:
        - `int`: The number of steps required for tiled scaling.
    """
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples: torch.Tensor,
    function: callable,
    tile_x: int = 64,
    tile_y: int = 64,
    overlap: int = 8,
    upscale_amount: float = 4,
    out_channels: int = 3,
    pbar: any = None,
) -> torch.Tensor:
    """#### Perform tiled scaling on a batch of samples.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `function` (callable): The function to apply to each tile.
        - `tile_x` (int, optional): The width of each tile. Defaults to 64.
        - `tile_y` (int, optional): The height of each tile. Defaults to 64.
        - `overlap` (int, optional): The overlap between tiles. Defaults to 8.
        - `upscale_amount` (float, optional): The upscale amount. Defaults to 4.
        - `out_channels` (int, optional): The number of output channels. Defaults to 3.
        - `pbar` (any, optional): The progress bar. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The scaled output tensor.
    """
    output = torch.empty(
        (
            samples.shape[0],
            out_channels,
            round(samples.shape[2] * upscale_amount),
            round(samples.shape[3] * upscale_amount),
        ),
        device="cpu",
    )
    for b in range(samples.shape[0]):
        s = samples[b : b + 1]
        out = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        out_div = torch.zeros(
            (
                s.shape[0],
                out_channels,
                round(s.shape[2] * upscale_amount),
                round(s.shape[3] * upscale_amount),
            ),
            device="cpu",
        )
        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                s_in = s[:, :, y : y + tile_y, x : x + tile_x]

                ps = function(s_in).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    mask[:, :, t : 1 + t, :] *= (1.0 / feather) * (t + 1)
                    mask[:, :, mask.shape[2] - 1 - t : mask.shape[2] - t, :] *= (
                        1.0 / feather
                    ) * (t + 1)
                    mask[:, :, :, t : 1 + t] *= (1.0 / feather) * (t + 1)
                    mask[:, :, :, mask.shape[3] - 1 - t : mask.shape[3] - t] *= (
                        1.0 / feather
                    ) * (t + 1)
                out[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += ps * mask
                out_div[
                    :,
                    :,
                    round(y * upscale_amount) : round((y + tile_y) * upscale_amount),
                    round(x * upscale_amount) : round((x + tile_x) * upscale_amount),
                ] += mask

        output[b : b + 1] = out / out_div
    return output


def flatten(img: Image.Image, bgcolor: str) -> Image.Image:
    """#### Replace transparency with a background color.

    #### Args:
        - `img` (Image.Image): The input image.
        - `bgcolor` (str): The background color.

    #### Returns:
        - `Image.Image`: The image with transparency replaced by the background color.
    """
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert(
        "RGB"
    )


BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
    """#### Convert a tensor to a PIL image.

    #### Args:
        - `img_tensor` (torch.Tensor): The input tensor.
        - `batch_index` (int, optional): The batch index. Defaults to 0.

    #### Returns:
        - `Image.Image`: The converted PIL image.
    """
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """#### Convert a PIL image to a tensor.

    #### Args:
        - `image` (Image.Image): The input PIL image.

    #### Returns:
        - `torch.Tensor`: The converted tensor.
    """
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_crop_region(mask: Image.Image, pad: int = 0) -> tuple:
    """#### Get the coordinates of the white rectangular mask region.

    #### Args:
        - `mask` (Image.Image): The input mask image in 'L' mode.
        - `pad` (int, optional): The padding to apply. Defaults to 0.

    #### Returns:
        - `tuple`: The coordinates of the crop region.
    """
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region: tuple, image_size: tuple) -> tuple:
    """#### Remove the extra pixel added by the get_crop_region function.

    #### Args:
        - `region` (tuple): The crop region coordinates.
        - `image_size` (tuple): The size of the image.

    #### Returns:
        - `tuple`: The fixed crop region coordinates.
    """
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region: tuple, width: int, height: int, target_width: int, target_height: int) -> tuple:
    """#### Expand a crop region to a specified target size.

    #### Args:
        - `region` (tuple): The crop region coordinates.
        - `width` (int): The width of the image.
        - `height` (int): The height of the image.
        - `target_width` (int): The desired width of the crop region.
        - `target_height` (int): The desired height of the crop region.

    #### Returns:
        - `tuple`: The expanded crop region coordinates and the target size.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def crop_cond(cond: list, region: tuple, init_size: tuple, canvas_size: tuple, tile_size: tuple, w_pad: int = 0, h_pad: int = 0) -> list:
    """#### Crop conditioning data to match a specific region.

    #### Args:
        - `cond` (list): The conditioning data.
        - `region` (tuple): The crop region coordinates.
        - `init_size` (tuple): The initial size of the image.
        - `canvas_size` (tuple): The size of the canvas.
        - `tile_size` (tuple): The size of the tile.
        - `w_pad` (int, optional): The width padding. Defaults to 0.
        - `h_pad` (int, optional): The height padding. Defaults to 0.

    #### Returns:
        - `list`: The cropped conditioning data.
    """
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        cropped.append(n)
    return cropped