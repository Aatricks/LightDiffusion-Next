import torch
from src.Utilities.Latent import Flux2 as Flux2LatentFormat


def _make_tiny_flux2(dtype=torch.float32):
    from src.NeuralNetwork.flux2.model import Flux2, Flux2Params

    params = Flux2Params(
        context_in_dim=16,
        vec_in_dim=16,
        hidden_size=64,
        num_heads=1,
        axes_dim=(16, 16, 16, 16),
        depth=1,
        depth_single_blocks=1,
    )
    return Flux2(params=params, dtype=dtype)


def test_patchify_from_vae_even_dims():
    fmt = Flux2LatentFormat()
    latent = torch.randn(1, 32, 64, 64)
    out = fmt.patchify_from_vae(latent)
    assert out.shape == (1, 128, 32, 32)


def test_patchify_from_vae_odd_dims_pads():
    fmt = Flux2LatentFormat()
    # Odd height/width should be padded internally so the operation succeeds
    latent = torch.randn(1, 32, 67, 65)
    out = fmt.patchify_from_vae(latent)
    # padded_h = 68 -> 68//2 = 34 ; padded_w = 66 -> 66//2 = 33
    assert out.shape == (1, 128, 34, 33)


def test_patchify_unpatchify_roundtrip_preserves_original_region():
    fmt = Flux2LatentFormat()
    latent = torch.randn(1, 32, 67, 65)
    patched = fmt.patchify_from_vae(latent)
    unpatched = fmt.unpatchify_for_vae(patched)
    # Flux2.forward crops back to original size; ensure round-trip preserves original region
    cropped = unpatched[:, :, :latent.shape[2], :latent.shape[3]]
    assert cropped.shape == latent.shape
    assert torch.allclose(cropped, latent)


def test_flux2_apply_model_accepts_vae_latent_with_odd_spatial_dims():
    """Simulate the exact call-site used by calc_cond_batch/_run_model_per_chunk
    to ensure Flux2.apply_model (and Latent.patchify_from_vae) accept odd H/W.
    """
    model = _make_tiny_flux2(dtype=torch.float32)
    # VAE-format latent with odd height (will require internal padding)
    vae_latent = torch.randn(1, 32, 67, 64)
    # timestep vector (single value is supported)
    t = torch.tensor([0.5])
    # dummy text conditioning matching model.params.context_in_dim
    txt = torch.zeros(1, 1, model.params.context_in_dim)

    out = model.apply_model(vae_latent, t, c_crossattn=txt)
    # output should match input spatial dims after model processing
    assert out.shape[0] == vae_latent.shape[0]
    assert out.shape[2] == vae_latent.shape[2]
    assert out.shape[3] == vae_latent.shape[3]


def test_flux2_apply_model_pads_or_crops_to_transformer_options():
    """When explicit transformer_options img_h/img_w are provided, the model
    must pad/crop the incoming latent so positional ids (RoPE) align.
    """
    model = _make_tiny_flux2(dtype=torch.float32)

    # Case A: VAE latent smaller than transformer_options -> should pad
    vae_latent_small = torch.randn(1, 32, 96, 38)
    t = torch.tensor([0.5])
    txt = torch.zeros(1, 1, model.params.context_in_dim)
    out_small = model.apply_model(vae_latent_small, t, c_crossattn=txt, transformer_options={"img_h": 512, "img_w": 512})
    # model should return same spatial shape as input VAE latent (crop back behavior)
    assert out_small.shape[2] == vae_latent_small.shape[2]
    assert out_small.shape[3] == vae_latent_small.shape[3]

    # Case B: VAE latent larger than transformer_options -> should crop safely
    vae_latent_large = torch.randn(1, 32, 160, 160)
    out_large = model.apply_model(vae_latent_large, t, c_crossattn=txt, transformer_options={"img_h": 256, "img_w": 256})
    assert out_large.shape[2] == vae_latent_large.shape[2]
    assert out_large.shape[3] == vae_latent_large.shape[3]
