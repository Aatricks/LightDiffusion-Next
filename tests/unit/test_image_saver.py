import logging
import torch

from src.FileManaging import ImageSaver


def test_save_images_guard(tmp_path, caplog):
    """save_images should abort and warn when asked to save too many images at once."""
    saver = ImageSaver.SaveImage()
    saver.output_dir = str(tmp_path)

    # Create more images than MAX_IMAGES_PER_SAVE but keep them small to avoid memory pressure
    images = [torch.rand(3, 32, 32) for _ in range(ImageSaver.MAX_IMAGES_PER_SAVE + 1)]

    caplog.set_level(logging.WARNING)
    res = saver.save_images(images)

    assert isinstance(res, dict)
    assert res["ui"]["images"] == []
    assert any("Attempting to save" in rec.getMessage() for rec in caplog.records)
