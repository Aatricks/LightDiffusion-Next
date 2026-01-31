import math
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor as tv_to_tensor


def get_tiled_scale_steps(width: int, height: int, tile_x: int, tile_y: int, overlap: int) -> int:
    """Calculate steps required for tiled scaling."""
    return math.ceil(height / (tile_y - overlap)) * math.ceil(width / (tile_x - overlap))


@torch.inference_mode()
def tiled_scale(samples: torch.Tensor, function: callable, tile_x: int = 64, tile_y: int = 64,
                overlap: int = 8, upscale_amount: float = 4, out_channels: int = 3, pbar=None) -> torch.Tensor:
    """Perform tiled upscaling on samples."""
    h_up, w_up = round(samples.shape[2] * upscale_amount), round(samples.shape[3] * upscale_amount)
    output = torch.empty((samples.shape[0], out_channels, h_up, w_up), device="cpu")

    for b in range(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros((s.shape[0], out_channels, h_up, w_up), device="cpu")
        out_div = torch.zeros_like(out)

        for y in range(0, s.shape[2], tile_y - overlap):
            for x in range(0, s.shape[3], tile_x - overlap):
                ps = function(s[:, :, y:y + tile_y, x:x + tile_x]).cpu()
                mask = torch.ones_like(ps)
                feather = round(overlap * upscale_amount)
                for t in range(feather):
                    f = (1.0 / feather) * (t + 1)
                    mask[:, :, t:1 + t, :] *= f
                    mask[:, :, -1 - t:-t or None, :] *= f
                    mask[:, :, :, t:1 + t] *= f
                    mask[:, :, :, -1 - t:-t or None] *= f

                y_start, y_end = round(y * upscale_amount), round((y + tile_y) * upscale_amount)
                x_start, x_end = round(x * upscale_amount), round((x + tile_x) * upscale_amount)
                out[:, :, y_start:y_end, x_start:x_end] += ps * mask
                out_div[:, :, y_start:y_end, x_start:x_end] += mask

        output[b:b + 1] = out / out_div
    return output


def flatten(img: Image.Image, bgcolor: str) -> Image.Image:
    """Replace transparency with background color."""
    if img.mode == "RGB":
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")


BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
    """Convert tensor to PIL image using torchvision."""
    tensor = img_tensor[batch_index]
    if tensor.dim() == 3 and tensor.shape[-1] in [1, 3, 4]:
        tensor = tensor.permute(2, 0, 1)
    return to_pil_image(torch.clamp(tensor, 0, 1))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor using torchvision."""
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    return tv_to_tensor(image).unsqueeze(0).permute(0, 2, 3, 1)


def get_crop_region(mask: Image.Image, pad: int = 0) -> tuple:
    """Get crop region from mask bounding box."""
    bbox = mask.getbbox()
    x1, y1, x2, y2 = bbox if bbox else (mask.width, mask.height, 0, 0)
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    x2, y2 = min(x2 + pad, mask.width), min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region: tuple, image_size: tuple) -> tuple:
    """Fix crop region by removing extra pixel."""
    w, h = image_size
    x1, y1, x2, y2 = region
    return x1, y1, x2 - 1 if x2 < w else x2, y2 - 1 if y2 < h else y2


def expand_crop(region: tuple, width: int, height: int, target_width: int, target_height: int) -> tuple:
    """Expand crop region to target size."""
    x1, y1, x2, y2 = region
    # Expand horizontally
    diff = target_width - (x2 - x1)
    x2 = min(x2 + diff // 2, width)
    diff = target_width - (x2 - x1)
    x1 = max(x1 - diff, 0)
    x2 = min(x2 + target_width - (x2 - x1), width)
    # Expand vertically
    diff = target_height - (y2 - y1)
    y2 = min(y2 + diff // 2, height)
    diff = target_height - (y2 - y1)
    y1 = max(y1 - diff, 0)
    y2 = min(y2 + target_height - (y2 - y1), height)
    return (x1, y1, x2, y2), (target_width, target_height)


def crop_cond(cond: list, region: tuple, init_size: tuple, canvas_size: tuple,
              tile_size: tuple, w_pad: int = 0, h_pad: int = 0) -> list:
    """Crop conditioning data to match region."""
    return [[emb, x.copy()] for emb, x in cond]
