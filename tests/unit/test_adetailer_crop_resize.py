from src.AutoDetailer.ADetailer import _compute_detailer_resize


def test_compute_detailer_resize_rounds_to_16():
    # Non-trivial sizes that will require rescaling/rounding
    width, height = 123, 245
    guide_size = 512
    max_size = 512
    upscale, new_w, new_h, force_inpaint = _compute_detailer_resize(width, height, guide_size, max_size)
    assert new_w % 16 == 0
    assert new_h % 16 == 0
    assert not force_inpaint


def test_compute_detailer_resize_downscales_large_crops_without_force_inpaint():
    width, height = 1024, 1024
    guide_size = 512
    max_size = 768
    upscale, new_w, new_h, force_inpaint = _compute_detailer_resize(width, height, guide_size, max_size)
    assert not force_inpaint
    assert new_w == 512
    assert new_h == 512
    assert new_w % 16 == 0
    assert new_h % 16 == 0
