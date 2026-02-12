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


def test_compute_detailer_resize_force_inpaint_rounds():
    # If upscale <= 1, force_inpaint should be True and rounding should still produce multiples of 16
    width, height = 1024, 1024
    guide_size = 32
    max_size = 512
    upscale, new_w, new_h, force_inpaint = _compute_detailer_resize(width, height, guide_size, max_size)
    assert force_inpaint
    assert new_w % 16 == 0
    assert new_h % 16 == 0
