from collections import namedtuple
import numpy as np
import torch
from modules.AutoDetailer import mask_util

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)


def segs_bitwise_and_mask(segs: tuple, mask: torch.Tensor) -> tuple:
    """#### Apply bitwise AND operation between segmentation masks and a given mask.

    #### Args:
        - `segs` (tuple): A tuple containing segmentation information.
        - `mask` (torch.Tensor): The mask tensor.

    #### Returns:
        - `tuple`: A tuple containing the original segmentation and the updated items.
    """
    mask = mask_util.make_2d_mask(mask)
    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[
            crop_region[1] : crop_region[3], crop_region[0] : crop_region[2]
        ]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(
            seg.cropped_image,
            new_mask,
            seg.confidence,
            seg.crop_region,
            seg.bbox,
            seg.label,
            None,
        )
        items.append(item)

    return segs[0], items


class SegsBitwiseAndMask:
    """#### Class to apply bitwise AND operation between segmentation masks and a given mask."""

    def doit(self, segs: tuple, mask: torch.Tensor) -> tuple:
        """#### Apply bitwise AND operation between segmentation masks and a given mask.

        #### Args:
            - `segs` (tuple): A tuple containing segmentation information.
            - `mask` (torch.Tensor): The mask tensor.

        #### Returns:
            - `tuple`: A tuple containing the original segmentation and the updated items.
        """
        return (segs_bitwise_and_mask(segs, mask),)


class SEGSLabelFilter:
    """#### Class to filter segmentation labels."""

    @staticmethod
    def filter(segs: tuple, labels: list) -> tuple:
        """#### Filter segmentation labels.

        #### Args:
            - `segs` (tuple): A tuple containing segmentation information.
            - `labels` (list): A list of labels to filter.

        #### Returns:
            - `tuple`: A tuple containing the original segmentation and an empty list.
        """
        labels = set([label.strip() for label in labels])
        return (
            segs,
            (segs[0], []),
        )
