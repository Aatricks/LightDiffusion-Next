import torch
from ultralytics import YOLO
from src.AutoDetailer import SEGS, AD_util, tensor_util
from typing import List, Tuple, Optional


class UltraBBoxDetector:
    """#### Class to detect bounding boxes using a YOLO model."""

    bbox_model: Optional[YOLO] = None

    def __init__(self, bbox_model: YOLO):
        """#### Initialize the UltraBBoxDetector with a YOLO model.

        #### Args:
            - `bbox_model` (YOLO): The YOLO model to use for detection.
        """
        self.bbox_model = bbox_model

    def detect(
        self,
        image: torch.Tensor,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int = 1,
        detailer_hook: Optional[callable] = None,
    ) -> Tuple[Tuple[int, int], List[SEGS.SEG]]:
        """#### Detect bounding boxes in an image.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `threshold` (float): The detection threshold.
            - `dilation` (int): The dilation factor for masks.
            - `crop_factor` (float): The crop factor for bounding boxes.
            - `drop_size` (int, optional): The minimum size of bounding boxes to keep. Defaults to 1.
            - `detailer_hook` (callable, optional): A hook function for additional processing. Defaults to None.

        #### Returns:
            - `Tuple[Tuple[int, int], List[SEGS.SEG]]`: The shape of the image and a list of detected segments.
        """
        drop_size = max(drop_size, 1)
        detected_results = AD_util.inference_bbox(
            self.bbox_model, tensor_util.tensor2pil(image), threshold
        )
        segmasks = AD_util.create_segmasks(detected_results)

        if dilation > 0:
            segmasks = AD_util.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if (
                x2 - x1 > drop_size and y2 - y1 > drop_size
            ):  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = AD_util.make_crop_region(w, h, item_bbox, crop_factor)

                cropped_image = AD_util.crop_image(image, crop_region)
                cropped_mask = AD_util.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]

                item = SEGS.SEG(
                    cropped_image,
                    cropped_mask,
                    confidence,
                    crop_region,
                    item_bbox,
                    label,
                    None,
                )

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        return segs


class UltraSegmDetector:
    """#### Class to detect segments using a YOLO model."""

    bbox_model: Optional[YOLO] = None

    def __init__(self, bbox_model: YOLO):
        """#### Initialize the UltraSegmDetector with a YOLO model.

        #### Args:
            - `bbox_model` (YOLO): The YOLO model to use for detection.
        """
        self.bbox_model = bbox_model


class NO_SEGM_DETECTOR:
    """#### Placeholder class for no segment detector."""

    pass


class UltralyticsDetectorProvider:
    """#### Class to provide YOLO models for detection."""

    def doit(self, model_name: str) -> Tuple[UltraBBoxDetector, UltraSegmDetector]:
        """#### Load a YOLO model and return detectors.

        #### Args:
            - `model_name` (str): The name of the YOLO model to load.

        #### Returns:
            - `Tuple[UltraBBoxDetector, UltraSegmDetector]`: The bounding box and segment detectors.
        """
        model = AD_util.load_yolo("./include/yolos/" + model_name)
        return UltraBBoxDetector(model), UltraSegmDetector(model)


class BboxDetectorForEach:
    """#### Class to detect bounding boxes for each segment."""

    def doit(
        self,
        bbox_detector: UltraBBoxDetector,
        image: torch.Tensor,
        threshold: float,
        dilation: int,
        crop_factor: float,
        drop_size: int,
        labels: Optional[str] = None,
        detailer_hook: Optional[callable] = None,
    ) -> Tuple[Tuple[int, int], List[SEGS.SEG]]:
        """#### Detect bounding boxes for each segment in an image.

        #### Args:
            - `bbox_detector` (UltraBBoxDetector): The bounding box detector.
            - `image` (torch.Tensor): The input image tensor.
            - `threshold` (float): The detection threshold.
            - `dilation` (int): The dilation factor for masks.
            - `crop_factor` (float): The crop factor for bounding boxes.
            - `drop_size` (int): The minimum size of bounding boxes to keep.
            - `labels` (str, optional): The labels to filter. Defaults to None.
            - `detailer_hook` (callable, optional): A hook function for additional processing. Defaults to None.

        #### Returns:
            - `Tuple[Tuple[int, int], List[SEGS.SEG]]`: The shape of the image and a list of detected segments.
        """
        segs = bbox_detector.detect(
            image, threshold, dilation, crop_factor, drop_size, detailer_hook
        )

        if labels is not None and labels != "":
            labels = labels.split(",")
            if len(labels) > 0:
                segs, _ = SEGS.SEGSLabelFilter.filter(segs, labels)

        return segs


class WildcardChooser:
    """#### Class to choose wildcards for segments."""

    def __init__(self, items: List[Tuple[None, str]], randomize_when_exhaust: bool):
        """#### Initialize the WildcardChooser.

        #### Args:
            - `items` (List[Tuple[None, str]]): The list of items to choose from.
            - `randomize_when_exhaust` (bool): Whether to randomize when the list is exhausted.
        """
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg: SEGS.SEG) -> Tuple[None, str]:
        """#### Get the next item from the list.

        #### Args:
            - `seg` (SEGS.SEG): The segment.

        #### Returns:
            - `Tuple[None, str]`: The next item from the list.
        """
        item = self.items[self.i]
        self.i += 1

        return item


def process_wildcard_for_segs(wildcard: str) -> Tuple[None, WildcardChooser]:
    """#### Process a wildcard for segments.

    #### Args:
        - `wildcard` (str): The wildcard.

    #### Returns:
        - `Tuple[None, WildcardChooser]`: The processed wildcard and a WildcardChooser.
    """
    return None, WildcardChooser([(None, wildcard)], False)
