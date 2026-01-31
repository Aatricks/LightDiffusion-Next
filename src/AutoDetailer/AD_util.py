from typing import List
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

orig_torch_load = torch.load
torch.load = orig_torch_load


def load_yolo(model_path: str) -> YOLO:
    """Load YOLO model from path."""
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        print("please download yolo model")


def inference_bbox(
    model: YOLO, image: Image.Image, confidence: float = 0.3, device: str = "cpu"
) -> List:
    """Perform YOLO inference and return [names, bboxes, segmasks, confidences]."""
    pred = model(image, conf=confidence, device=device)
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)[:, :, ::-1].copy()  # RGB to BGR
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        segms.append(cv2_mask.astype(bool))

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())
    return results


def create_segmasks(results: List) -> List:
    """Convert inference results to list of (bbox, segmask, confidence)."""
    return [(results[1][i], results[2][i].astype(np.float32), results[3][i]) 
            for i in range(len(results[2]))]


def dilate_masks(segmasks: List, dilation_factor: int, iter: int = 1) -> List:
    """Dilate segmentation masks by dilation_factor."""
    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def normalize_region(limit: int, startp: int, size: int) -> List:
    """Normalize region coords to fit within limit."""
    if startp < 0:
        return 0, min(limit, size)
    if startp + size > limit:
        return max(0, limit - size), limit
    return int(startp), int(min(limit, startp + size))


def make_crop_region(w: int, h: int, bbox: List, crop_factor: float) -> List:
    """Create expanded crop region from bbox."""
    x1, y1, x2, y2 = bbox
    bbox_w, bbox_h = x2 - x1, y2 - y1
    crop_w, crop_h = bbox_w * crop_factor, bbox_h * crop_factor
    kernel_x, kernel_y = x1 + bbox_w / 2, y1 + bbox_h / 2
    new_x1, new_x2 = normalize_region(w, int(kernel_x - crop_w / 2), crop_w)
    new_y1, new_y2 = normalize_region(h, int(kernel_y - crop_h / 2), crop_h)
    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """Crop 2D array [H,W]."""
    x1, y1, x2, y2 = crop_region
    return npimg[y1:y2, x1:x2]


def crop_ndarray4(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """Crop 4D array [B,H,W,C]."""
    x1, y1, x2, y2 = crop_region
    return npimg[:, y1:y2, x1:x2, :]


def crop_image(image: torch.Tensor, crop_region: List) -> torch.Tensor:
    """Crop tensor image."""
    if torch.is_tensor(image):
        if len(image.shape) == 4:
            return torch.from_numpy(crop_ndarray4(image.cpu().numpy(), crop_region))
        elif len(image.shape) == 3:
            cropped = crop_ndarray4(image.unsqueeze(0).cpu().numpy(), crop_region)
            return torch.from_numpy(cropped).squeeze(0)
        raise ValueError(f"Unsupported image tensor shape: {image.shape}")
    cropped = crop_ndarray4(image, crop_region)
    return torch.from_numpy(cropped) if isinstance(cropped, np.ndarray) else cropped


def segs_scale_match(segs: List[np.ndarray], target_shape: List) -> List:
    """Scale segmentation masks to target shape."""
    h, w = segs[0][0], segs[0][1]
    th, tw = target_shape[1], target_shape[2]
    if (h == th and w == tw) or h == 0 or w == 0:
        return segs
