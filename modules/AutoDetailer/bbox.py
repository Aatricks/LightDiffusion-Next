from modules.AutoDetailer import SEGS, AD_util


class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(
        self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None
    ):
        drop_size = max(drop_size, 1)
        detected_results = AD_util.inference_bbox(
            self.bbox_model, AD_util.tensor2pil(image), threshold
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
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

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
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model


class NO_SEGM_DETECTOR:
    pass


class UltralyticsDetectorProvider:
    def doit(self, model_name):
        model = AD_util.load_yolo("./_internal/yolos/" + model_name)
        return UltraBBoxDetector(model), UltraSegmDetector(model)


class BboxDetectorForEach:
    def doit(
        self,
        bbox_detector,
        image,
        threshold,
        dilation,
        crop_factor,
        drop_size,
        labels=None,
        detailer_hook=None,
    ):
        segs = bbox_detector.detect(
            image, threshold, dilation, crop_factor, drop_size, detailer_hook
        )

        if labels is not None and labels != "":
            labels = labels.split(",")
            if len(labels) > 0:
                segs, _ = SEGS.SEGSLabelFilter.filter(segs, labels)

        return (segs,)


class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        item = self.items[self.i]
        self.i += 1

        return item


def process_wildcard_for_segs(wildcard):
    return None, WildcardChooser([(None, wildcard)], False)
