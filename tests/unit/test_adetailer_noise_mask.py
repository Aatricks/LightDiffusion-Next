import torch

from src.AutoDetailer import tensor_util
import src.AutoDetailer.ADetailer as adetailer


def test_compute_detailer_resize_rounds_to_multiple_of_8():
    """Dimensions must be divisible by 8 to avoid VAE encode NaN."""
    # Face crop: 135x157 with guide_size=768, max_size=1024
    _, w, h, _ = adetailer._compute_detailer_resize(135, 157, 768, 1024)
    assert w % 8 == 0, f"width {w} not divisible by 8"
    assert h % 8 == 0, f"height {h} not divisible by 8"

    # Body crop: 612x1024 with guide_size=768, max_size=1024
    _, w, h, _ = adetailer._compute_detailer_resize(612, 1024, 768, 1024)
    assert w % 8 == 0, f"width {w} not divisible by 8"
    assert h % 8 == 0, f"height {h} not divisible by 8"

    # Edge case: very small crop
    _, w, h, _ = adetailer._compute_detailer_resize(7, 13, 768, 1024)
    assert w % 8 == 0, f"width {w} not divisible by 8"
    assert h % 8 == 0, f"height {h} not divisible by 8"
    assert w >= 8 and h >= 8

    # Exact multiples should stay the same
    _, w, h, _ = adetailer._compute_detailer_resize(768, 1024, 768, 1024)
    assert w % 8 == 0, f"width {w} not divisible by 8"
    assert h % 8 == 0, f"height {h} not divisible by 8"

    # Larger crops should downscale cleanly without forcing full-size inpaint
    _, w, h, fi = adetailer._compute_detailer_resize(800, 900, 768, 1024)
    assert w % 8 == 0, f"width {w} not divisible by 8"
    assert h % 8 == 0, f"height {h} not divisible by 8"
    assert fi is False


def test_enhance_detail_localizes_noise_to_mask(monkeypatch):
    image = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    mask = torch.zeros((8, 8), dtype=torch.float32)
    mask[2:6, 2:6] = 255.0
    captured = {}

    def fake_to_latent_image(pixels, vae):
        return {"samples": pixels.clone()}

    def fake_ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                              denoise, refiner_ratio=None, refiner_model=None, refiner_clip=None,
                              refiner_positive=None, refiner_negative=None, sigma_factor=1.0, noise=None,
                              callback=None, scheduler_func=None, pipeline=False):
        captured["mask"] = latent_image.get("noise_mask")
        samples = latent_image["samples"]
        noise_mask = captured["mask"]
        if noise_mask is None:
            return latent_image
        if noise_mask.dim() == 2:
            noise_mask = noise_mask.unsqueeze(0)
        if noise_mask.dim() == 3:
            noise_mask = noise_mask.unsqueeze(-1)
        # Resize mask to match samples if needed
        if noise_mask.shape[-2:] != samples.shape[1:3]:
            noise_mask = torch.nn.functional.interpolate(
                noise_mask.permute(0, 3, 1, 2),
                size=(samples.shape[1], samples.shape[2]),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        generator = torch.Generator(device=samples.device).manual_seed(seed)
        noise = torch.randn_like(samples, generator=generator)
        blended = samples * (1.0 - noise_mask) + noise * noise_mask
        return {**latent_image, "samples": blended}

    class DummyVAE:
        def decode(self, samples):
            return samples

        def decode_tiled(self, samples, tile_x=256, tile_y=256):
            return samples

    monkeypatch.setattr(adetailer, "to_latent_image", fake_to_latent_image)
    monkeypatch.setattr(adetailer, "ksampler_wrapper", fake_ksampler_wrapper)
    monkeypatch.setattr(tensor_util.Device, "get_torch_device", lambda: torch.device("cpu"))

    enhanced, _ = adetailer.enhance_detail(
        image,
        model=None,
        clip=None,
        vae=DummyVAE(),
        guide_size=16,
        guide_size_for_bbox=16,
        max_size=64,
        bbox=(0, 0, 8, 8),
        seed=123,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="ays",
        positive=None,
        negative=None,
        denoise=1.0,
        noise_mask=mask,
        force_inpaint=False,
        noise_mask_feather=0,
    )

    assert captured["mask"] is not None
    assert enhanced.shape == image.shape

    delta = (enhanced - image).abs().mean(dim=-1)[0]
    inside = delta[2:6, 2:6]
    outside = torch.cat([
        delta[:2, :].flatten(),
        delta[6:, :].flatten(),
        delta[2:6, :2].flatten(),
        delta[2:6, 6:].flatten(),
    ])

    assert inside.mean().item() > 0.05
    assert outside.mean().item() < inside.mean().item() * 0.4
