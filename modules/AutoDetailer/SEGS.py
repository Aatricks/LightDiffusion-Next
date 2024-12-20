from collections import namedtuple
import numpy as np

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

def segs_bitwise_and_mask(segs, mask):
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
    def doit(self, segs, mask):
        return (segs_bitwise_and_mask(segs, mask),)
    
class SEGSLabelFilter:
    @staticmethod
    def filter(segs, labels):
        labels = set([label.strip() for label in labels])
        return (
            segs,
            (segs[0], []),
        )