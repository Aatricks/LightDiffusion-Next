from src.Core.AbstractModel import ModelCapabilities

def test_validate_resolution_within_limits():
    caps = ModelCapabilities(min_resolution=256, max_resolution=2048, requires_resolution_multiple=64)
    # 1024x1024 should remain unchanged
    w, h = caps.validate_resolution(1024, 1024)
    assert w == 1024
    assert h == 1024

def test_validate_resolution_clamping_width():
    caps = ModelCapabilities(min_resolution=256, max_resolution=2048, requires_resolution_multiple=64)
    # 2688x1536 (Ratio 1.75)
    # Width exceeds 2048. Scale = 2048/2688 = 0.7619
    # New width = 2048
    # New height = 1536 * 0.7619 = 1170.28
    # Rounded to 64: (1170 // 64) * 64 = 18 * 64 = 1152
    w, h = caps.validate_resolution(2688, 1536)
    assert w == 2048
    assert h == 1152
    assert abs(w/h - 2688/1536) < 0.05

def test_validate_resolution_clamping_height():
    caps = ModelCapabilities(min_resolution=256, max_resolution=2048, requires_resolution_multiple=64)
    # 1536x2688 (Ratio 0.5714)
    # Height exceeds 2048. Scale = 2048/2688 = 0.7619
    # New height = 2048
    # New width = 1536 * 0.7619 = 1170.28
    # Rounded to 64: 1152
    w, h = caps.validate_resolution(1536, 2688)
    assert w == 1152
    assert h == 2048
    assert abs(w/h - 1536/2688) < 0.05

def test_validate_resolution_clamping_both():
    caps = ModelCapabilities(min_resolution=256, max_resolution=2048, requires_resolution_multiple=64)
    # 4096x3072 (Ratio 1.33)
    # Width exceeds 2048 more than height.
    # Scale = min(2048/4096, 2048/3072) = min(0.5, 0.66) = 0.5
    # New width = 2048
    # New height = 1536
    w, h = caps.validate_resolution(4096, 3072)
    assert w == 2048
    assert h == 1536

def test_flux_resolution_limits():
    # Flux has multiple 16
    caps = ModelCapabilities(min_resolution=256, max_resolution=4096, requires_resolution_multiple=16)
    # 1344x768 -> 2x Hires Fix -> 2688x1536
    # Should NOT be clamped if max_resolution is 4096
    w, h = caps.validate_resolution(2688, 1536)
    assert w == 2688
    assert h == 1536
