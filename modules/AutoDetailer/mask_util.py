import numpy as np
import torch


def center_of_bbox(bbox: list) -> tuple[float, float]:
    """#### Calculate the center of a bounding box.

    #### Args:
        - `bbox` (list): The bounding box coordinates [x1, y1, x2, y2].

    #### Returns:
        - `tuple[float, float]`: The center coordinates (x, y).
    """
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def make_2d_mask(mask: torch.Tensor) -> torch.Tensor:
    """#### Convert a mask to 2D.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.

    #### Returns:
        - `torch.Tensor`: The 2D mask tensor.
    """
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask


def combine_masks2(masks: list) -> torch.Tensor | None:
    """#### Combine multiple masks into one.

    #### Args:
        - `masks` (list): A list of mask tensors.

    #### Returns:
        - `torch.Tensor | None`: The combined mask tensor or None if no masks are provided.
    """
    try:
        mask = torch.from_numpy(np.array(masks[0]).astype(np.uint8))
    except:
        print("No Human Detected")
        return None
    return mask


def dilate_mask(mask: torch.Tensor, dilation_factor: int, iter: int = 1) -> torch.Tensor:
    """#### Dilate a mask.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.
        - `dilation_factor` (int): The dilation factor.
        - `iter` (int, optional): The number of iterations. Defaults to 1.

    #### Returns:
        - `torch.Tensor`: The dilated mask tensor.
    """
    return make_2d_mask(mask)


def make_3d_mask(mask: torch.Tensor) -> torch.Tensor:
    """#### Convert a mask to 3D.

    #### Args:
        - `mask` (torch.Tensor): The input mask tensor.

    #### Returns:
        - `torch.Tensor`: The 3D mask tensor.
    """
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask