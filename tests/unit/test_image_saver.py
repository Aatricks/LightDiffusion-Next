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


def test_save_images_aborts_on_large_batched_tensor(caplog):
    """A single batched tensor with a very large batch dimension should be treated like many images and abort."""
    saver = ImageSaver.SaveImage()

    batch = 1024
    tensor = torch.zeros((batch, 3, 16, 16))

    caplog.set_level(logging.WARNING)
    res = saver.save_images([tensor])

    assert res == {"ui": {"images": []}}
    assert any("Attempting to save" in rec.getMessage() for rec in caplog.records)
    # Diagnostic details should include an idx=0 entry and the batch size (1024) in the message
    assert any("idx=0" in rec.getMessage() and "1024" in rec.getMessage() for rec in caplog.records)
    # Ensure the filename_prefix and store_bytes_prefix are included for tracing
    assert any("filename_prefix=LD" in rec.getMessage() for rec in caplog.records)


def test_save_images_saves_single_image(tmp_path):
    saver = ImageSaver.SaveImage()
    saver.output_dir = str(tmp_path)

    tensor = torch.rand((1, 3, 32, 32))
    res = saver.save_images([tensor], filename_prefix="LD", prompt="test")

    assert isinstance(res, dict)
    assert "ui" in res and "images" in res["ui"]
    assert len(res["ui"]["images"]) == 1
