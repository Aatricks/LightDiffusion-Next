import torch

from src.Processors.AutoHDRProcessor import AutoHDRProcessor


def test_autohdr_single_rgb_image_returns_batched_tensor():
    # Single RGB image shape: [H, W, C]
    img = torch.rand(64, 64, 3)

    out = AutoHDRProcessor.apply(img, ctx=None)
    assert isinstance(out, torch.Tensor)
    # AutoHDRProcessor.apply returns a batched tensor (B, H, W, C)
    assert out.ndim == 4
    assert out.shape[0] == 1
    assert out.shape[1] == 64 and out.shape[2] == 64 and out.shape[3] == 3


def test_autohdr_single_rgba_image_preserves_alpha_and_batches():
    # Single RGBA image shape: [H, W, 4]
    img = torch.rand(32, 48, 4)

    out = AutoHDRProcessor.apply(img, ctx=None)

    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4
    assert out.shape[0] == 1
    assert out.shape[1] == 32 and out.shape[2] == 48 and out.shape[3] == 4


def test_autohdr_and_saveimage_single_image_roundtrip(tmp_path):
    """Ensure AutoHDR output from a single image is accepted by SaveImage and saved as a single image."""
    img = torch.rand(64, 64, 3)
    out = AutoHDRProcessor.apply(img, ctx=None)

    from src.FileManaging.ImageSaver import SaveImage
    saver = SaveImage()
    saver.output_dir = str(tmp_path)

    res = saver.save_images([out], filename_prefix="LD-TEST", prompt="p", extra_pnginfo=None, store_bytes_prefix=None)
    ui = res.get("ui", {})
    images = ui.get("images", [])

    # Should save exactly one image
    assert len(images) == 1
    # Filename should be present
    assert isinstance(images[0].get("filename"), str)


def test_autohdr_large_single_image_does_not_produce_tiles(tmp_path):
    """Regression test for the tiled-slices issue reported in production.

    Previously a single large image (H, W, C) was mistakenly iterated
    over as a sequence of rows, producing ~H small slices (H x few px).
    This test ensures a large single image is processed as a single image
    and saved as one file.
    """
    large_img = torch.rand(1024, 1024, 3)
    out = AutoHDRProcessor.apply(large_img, ctx=None)

    from src.FileManaging.ImageSaver import SaveImage
    saver = SaveImage()
    saver.output_dir = str(tmp_path)

    res = saver.save_images([out], filename_prefix="LD-LARGE", prompt="p", extra_pnginfo=None, store_bytes_prefix=None)
    ui = res.get("ui", {})
    images = ui.get("images", [])

    assert len(images) == 1
    assert isinstance(images[0].get("filename"), str)