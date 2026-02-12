import torch
from src.NeuralNetwork.flux2.layers import apply_rope1


def make_identity_freqs(batch: int, seq: int, dim_half: int, device=None):
    # Build freqs_cis tensor where cos=1, sin=0 (identity rotation)
    device = device or torch.device('cpu')
    cos = torch.ones((batch, 1, seq, dim_half), dtype=torch.float32, device=device)
    msin = torch.zeros_like(cos)
    sin = torch.zeros_like(cos)
    freqs = torch.zeros((batch, 1, seq, dim_half, 2, 2), dtype=torch.float32, device=device)
    freqs[..., 0, 0] = cos
    freqs[..., 0, 1] = msin
    freqs[..., 1, 0] = sin
    freqs[..., 1, 1] = cos
    return freqs


def test_apply_rope1_upsamples_short_pe_ok():
    x = torch.randn(1, 1, 10, 4)
    freqs = make_identity_freqs(batch=1, seq=8, dim_half=2)
    out = apply_rope1(x, freqs)
    assert out.shape == x.shape
    # With identity freqs (cos=1, sin=0) the output should equal the input
    assert torch.allclose(out, x)


def test_apply_rope1_slices_long_pe_ok():
    x = torch.randn(1, 1, 10, 4)
    freqs = make_identity_freqs(batch=1, seq=12, dim_half=2)
    out = apply_rope1(x, freqs)
    assert out.shape == x.shape
    assert torch.allclose(out, x)
