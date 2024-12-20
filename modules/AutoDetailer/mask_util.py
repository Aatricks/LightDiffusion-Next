import numpy as np
import torch


def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w / 2, bbox[1] + h / 2


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask


def combine_masks2(masks):
    mask = torch.from_numpy(np.array(masks[0]).astype(np.uint8))
    return mask


def dilate_mask(mask, dilation_factor, iter=1):
    return make_2d_mask(mask)


def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask