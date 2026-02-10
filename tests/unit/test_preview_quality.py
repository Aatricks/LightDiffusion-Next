import unittest
import torch
from PIL import Image

from src.Utilities import color
from src.user.app_instance import AppInstance
from src.AutoEncoders import taesd


class TestPreviewQuality(unittest.TestCase):
    def test_linear_to_srgb_values(self):
        # Known values across the transfer function
        vals = torch.tensor([0.0, 0.0031308, 0.04, 0.5, 1.0], dtype=torch.float32)
        out = color.linear_to_srgb(vals)
        # Expected values (approx)
        expected = torch.tensor([
            0.0,
            0.0031308 * 12.92,
            1.055 * (0.04 ** (1.0 / 2.4)) - 0.055,
            1.055 * (0.5 ** (1.0 / 2.4)) - 0.055,
            1.0,
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_decode_uses_lanczos_for_thumbnail(self):
        # Monkeypatch TAESD.decode to return a synthetic tensor large enough
        # to trigger downsampling, and monkeypatch Image.fromarray to capture
        # the resample argument passed to thumbnail.
        orig_decode = taesd.TAESD.decode
        orig_fromarray = Image.fromarray
        called = {}

        def fake_decode(self, x):
            # Return a [B, C, H, W] tensor in [-1, 1] so that after .add(1).mul(0.5)
            # values are in [0,1]. Use a large size to trigger thumbnail.
            return torch.full((1, 3, 1024, 1024), 0.0, dtype=x.dtype)

        class FakeImage:
            def __init__(self):
                self.width = 1024
                self.height = 1024

            def thumbnail(self, size, resample=None):
                called['resample'] = resample
                # Emulate behavior of PIL thumbnail
                self.width = min(self.width, size[0])
                self.height = min(self.height, size[1])

            def save(self, *args, **kwargs):
                return None

        def fake_fromarray(arr, mode=None):
            return FakeImage()

        try:
            taesd.TAESD.decode = fake_decode
            Image.fromarray = fake_fromarray

            latent = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
            imgs = taesd.decode_latents_to_images(latent)

            # Confirm our fake thumbnail captured the resample argument
            self.assertIn('resample', called)
            self.assertEqual(called['resample'], Image.Resampling.LANCZOS)
        finally:
            taesd.TAESD.decode = orig_decode
            Image.fromarray = orig_fromarray

    def test_app_preview_defaults(self):
        app = AppInstance()
        self.assertTrue(hasattr(app, 'preview_srgb'))
        self.assertTrue(app.preview_srgb)
        self.assertEqual(app.preview_format, 'WEBP')
        self.assertEqual(app.preview_quality, 90)

    def test_decode_applies_srgb_when_enabled(self):
        # Monkeypatch TAESD.decode to return constant zero (-> 0.5 after norm)
        orig_decode = taesd.TAESD.decode
        try:
            def fake_decode(self, x):
                return torch.zeros((1, 3, 4, 4), dtype=x.dtype, device=x.device)
            taesd.TAESD.decode = fake_decode

            # Ensure preview_srgb enabled
            from src.user.app_instance import app as global_app
            old_flag = global_app.preview_srgb
            global_app.preview_srgb = True

            latent = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
            imgs = taesd.decode_latents_to_images(latent)
            self.assertTrue(len(imgs) > 0)
            img = imgs[0]
            r, g, b = img.getpixel((0, 0))

            # Expected sRGB value for linear=0.5
            lin = 0.5
            srgb = 1.055 * (lin ** (1.0 / 2.4)) - 0.055
            # The implementation casts to uint8 (truncates), so expect floor behavior
            expect = int(srgb * 255.0)
            self.assertEqual(r, expect)
            self.assertEqual(g, expect)
            self.assertEqual(b, expect)

        finally:
            taesd.TAESD.decode = orig_decode
            global_app.preview_srgb = old_flag

    def test_server_callback_uses_preview_format(self):
        # Ensure server's preview callback attempts to use configured preview format
        import io
        import server as server_mod
        orig_save = Image.Image.save
        orig_decode = server_mod.decode_latents_to_images
        try:
            saved = []
            def fake_save(self, buffer, format=None, **kwargs):
                saved.append(format)
                # write some bytes so the buffer isn't empty
                try:
                    buffer.write(b"OK")
                except Exception:
                    pass
            Image.Image.save = fake_save

            def fake_decode(latents, flux=False):
                return [Image.new('RGB', (64, 64), color='red')]
            server_mod.decode_latents_to_images = fake_decode

            from src.user import app_instance as _app_instance
            old_fmt = _app_instance.app.preview_format
            _app_instance.app.preview_format = 'WEBP'

            cb = server_mod.make_server_callback(20)
            cb({'i': 0, 'total_steps': 20, 'denoised': torch.zeros((1, 4, 4, 4))})

            self.assertTrue(len(saved) > 0)
            self.assertEqual(saved[0].upper(), 'WEBP')
        finally:
            Image.Image.save = orig_save
            server_mod.decode_latents_to_images = orig_decode
            _app_instance.app.preview_format = old_fmt

if __name__ == "__main__":
    unittest.main()
