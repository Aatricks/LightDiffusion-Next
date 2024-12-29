import math
import numpy as np
import torch
from PIL import Image


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil(
        (width / (tile_x - overlap))
    )


@torch.inference_mode()
def tiled_scale(
    samples,
    function,
    tile_x=64,
    tile_y=64,
    overlap=8,
    upscale_amount=4,
    out_channels=3,
    pbar=None,
):
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


def flatten(img, bgcolor):
    # Replace transparency with bgcolor
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert(
        "RGB"
    )


BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor, batch_index=0):
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255.0 * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    return image


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
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


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    """
    Expands a crop region to a specified target size.
    :param region: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param width: The width of the image the crop region is from.
    :param height: The height of the image the crop region is from.
    :param target_width: The desired width of the crop region.
    :param target_height: The desired height of the crop region.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1
    # target_width = math.ceil(actual_width / 8) * 8
    # target_height = math.ceil(actual_height / 8) * 8

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


def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        cropped.append(n)
    return cropped
