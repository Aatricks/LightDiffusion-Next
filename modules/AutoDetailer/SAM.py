import os
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch

from modules.AutoDetailer import mask_util
from modules.Device import Device


def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels, box=box
    )

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks


def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index


class SafeToGPU:
    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device):
        if is_same_device(device, "cpu"):
            obj.to(device)
        else:
            if is_same_device(obj.device, "cpu"):  # cpu to gpu
                Device.free_memory(self.size * 1.3, device)
                if Device.get_free_memory(device) > self.size * 1.3:
                    try:
                        obj.to(device)
                    except:
                        print(
                            f"WARN: The model is not moved to the '{device}' due to insufficient memory. [1]"
                        )
                else:
                    print(
                        f"WARN: The model is not moved to the '{device}' due to insufficient memory. [2]"
                    )


class SAMWrapper:
    def __init__(self, model, is_auto_mode, safe_to_gpu=None):
        self.model = model
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        if self.is_auto_mode:
            device = Device.get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)


class SAMLoader:
    def load_model(self, model_name, device_mode="auto"):
        modelname = "./_internal/yolos/" + model_name

        if "vit_h" in model_name:
            model_kind = "vit_h"
        elif "vit_l" in model_name:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        size = os.path.getsize(modelname)
        safe_to = SafeToGPU(size)

        # Unless user explicitly wants to use CPU, we use GPU
        device = Device.get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            safe_to.to_device(sam, device)

        is_auto_mode = device_mode == "AUTO"

        sam_obj = SAMWrapper(sam, is_auto_mode=is_auto_mode, safe_to_gpu=safe_to)
        sam.sam_wrapper = sam_obj

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam,)


def make_sam_mask(
    sam,
    segs,
    image,
    detection_hint,
    dilation,
    threshold,
    bbox_expansion,
    mask_hint_threshold,
    mask_hint_use_negative,
):
    sam_obj = sam.sam_wrapper
    sam_obj.prepare_device()

    try:
        image = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []
        # seg_shape = segs[0]
        segs = segs[1]
        for i in range(len(segs)):
            bbox = segs[i].bbox
            center = mask_util.center_of_bbox(bbox)
            x1 = max(bbox[0] - bbox_expansion, 0)
            y1 = max(bbox[1] - bbox_expansion, 0)
            x2 = min(bbox[2] + bbox_expansion, image.shape[1])
            y2 = min(bbox[3] + bbox_expansion, image.shape[0])
            dilated_bbox = [x1, y1, x2, y2]
            points = []
            plabs = []
            points.append(center)
            plabs = [1]  # 1 = foreground point, 0 = background point
            detected_masks = sam_obj.predict(
                image, points, plabs, dilated_bbox, threshold
            )
            total_masks += detected_masks

        # merge every collected masks
        mask = mask_util.combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    if mask is not None:
        mask = mask.float()
        mask = mask_util.dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)

        mask = mask_util.make_3d_mask(mask)
        return mask
    else:
        return None


class SAMDetectorCombined:
    def doit(
        self,
        sam_model,
        segs,
        image,
        detection_hint,
        dilation,
        threshold,
        bbox_expansion,
        mask_hint_threshold,
        mask_hint_use_negative,
    ):
        sam = make_sam_mask(
            sam_model,
            segs,
            image,
            detection_hint,
            dilation,
            threshold,
            bbox_expansion,
            mask_hint_threshold,
            mask_hint_use_negative,
        )
        if sam is not None:
            return (sam,)
        else:
            return None
