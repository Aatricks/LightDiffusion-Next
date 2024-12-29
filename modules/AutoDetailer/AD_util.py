from typing import List
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

orig_torch_load = torch.load

# importing YOLO breaking original torch.load capabilities
torch.load = orig_torch_load


def load_yolo(model_path: str) -> YOLO:
    """#### Load YOLO model.

    #### Args:
        - `model_path` (str): The path to the YOLO model.

    #### Returns:
        - `YOLO`: The YOLO model initialized with the specified model path.
    """
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        print("please download yolo model")


def inference_bbox(
    model: YOLO,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
) -> List:
    """#### Perform inference on an image and return bounding boxes.

    #### Args:
        - `model` (YOLO): The YOLO model.
        - `image` (Image.Image): The image to perform inference on.
        - `confidence` (float): The confidence threshold for the bounding boxes.
        - `device` (str): The device to run the model on.

    #### Returns:
        - `List[List[str, List[int], np.ndarray, float]]`: The list of bounding boxes.
    """
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def create_segmasks(results: List) -> List:
    """#### Create segmentation masks from the results of the inference.

    #### Args:
        - `results` (List[List[str, List[int], np.ndarray, float]]): The results of the inference.

    #### Returns:
        - `List[List[int], np.ndarray, float]`: The list of segmentation masks.
    """
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def dilate_masks(segmasks: List, dilation_factor: int, iter: int = 1) -> List:
    """#### Dilate the segmentation masks.

    #### Args:
        - `segmasks` (List[List[int], np.ndarray, float]): The segmentation masks.
        - `dilation_factor` (int): The dilation factor.
        - `iter` (int): The number of iterations.

    #### Returns:
        - `List[List[int], np.ndarray, float]`: The dilated segmentation masks.
    """
    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def normalize_region(limit: int, startp: int, size: int) -> List:
    """#### Normalize the region.

    #### Args:
        - `limit` (int): The limit.
        - `startp` (int): The start point.
        - `size` (int): The size.

    #### Returns:
        - `List[int]`: The normalized start and end points.
    """
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp + size)

    return int(new_startp), int(new_endp)


def make_crop_region(w: int, h: int, bbox: List, crop_factor: float) -> List:
    """#### Make the crop region.

    #### Args:
        - `w` (int): The width.
        - `h` (int): The height.
        - `bbox` (List[int]): The bounding box.
        - `crop_factor` (float): The crop factor.

    #### Returns:
        - `List[x1: int, y1: int, x2: int, y2: int]`: The crop region.
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def crop_ndarray2(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """#### Crop the ndarray in 2 dimensions.

    #### Args:
        - `npimg` (np.ndarray): The ndarray to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `np.ndarray`: The cropped ndarray.
    """
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


def crop_ndarray4(npimg: np.ndarray, crop_region: List) -> np.ndarray:
    """#### Crop the ndarray in 4 dimensions.

    #### Args:
        - `npimg` (np.ndarray): The ndarray to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `np.ndarray`: The cropped ndarray.
    """
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


def crop_image(image: Image.Image, crop_region: List) -> Image.Image:
    """#### Crop the image.

    #### Args:
        - `image` (Image.Image): The image to crop.
        - `crop_region` (List[int]): The crop region.

    #### Returns:
        - `Image.Image`: The cropped image.
    """
    return crop_ndarray4(image, crop_region)


def segs_scale_match(segs: List[np.ndarray], target_shape: List) -> List:
    """#### Match the scale of the segmentation masks.

    #### Args:
        - `segs` (List[np.ndarray]): The segmentation masks.
        - `target_shape` (List[int]): The target shape.

    #### Returns:
        - `List[np.ndarray]`: The matched segmentation masks.
    """
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs
