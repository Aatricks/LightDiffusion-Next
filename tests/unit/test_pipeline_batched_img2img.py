from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.Core.Context import Context
from src.Core.Pipeline import Pipeline


class DummyModel:
    def __init__(self):
        self.capabilities = SimpleNamespace(
            is_flux=False,
            is_flux2=False,
            uses_dual_clip=False,
            requires_size_conditioning=False,
        )
        self.model = SimpleNamespace(model_options={})

    def get_model_object(self, name):
        if name == "latent_format":
            return SimpleNamespace(latent_channels=4)
        return None

    def encode_prompt(self, prompts, negatives):
        positive = [[torch.randn(1, 77, 768), {}] for _ in prompts]
        negative = [[torch.randn(1, 77, 768), {}] for _ in negatives]
        return positive, negative

    def decode(self, latents):
        batch = latents.shape[0]
        return torch.zeros((batch, 64, 64, 3), dtype=torch.float32)

    def unload(self):
        return None


def test_run_batched_img2img_repeats_input_and_tags_both_conditionings(monkeypatch, tmp_path: Path):
    from src.Processors import Img2Img

    image_path = tmp_path / "img2img-input.png"
    Image.new("RGB", (64, 64), color=(64, 128, 192)).save(image_path)

    ctx = Context.from_kwargs(
        prompt=["prompt 1", "prompt 2"],
        negative_prompt=["neg 1", "neg 2"],
        w=64,
        h=64,
        number=2,
        batch=2,
        img2img=True,
        img2img_image=str(image_path),
        img2img_denoise=0.6,
        autohdr=False,
    )

    pipeline = Pipeline()
    dummy_model = DummyModel()

    monkeypatch.setattr(pipeline, "_load_model", lambda _ctx: dummy_model)
    monkeypatch.setattr(pipeline, "_apply_optimizations", lambda _ctx, _model: None)

    captured = {}

    def fake_simple_img2img(ctx, model, positive, negative, image_tensor, denoise=0.75, last_step=None, callback=None):
        captured["image_shape"] = tuple(image_tensor.shape)
        captured["positive_batch_index"] = [entry[1]["batch_index"] for entry in positive]
        captured["negative_batch_index"] = [entry[1]["batch_index"] for entry in negative]
        captured["denoise"] = denoise
        return ({"samples": torch.zeros((image_tensor.shape[0], 4, 8, 8), dtype=torch.float32)},)

    monkeypatch.setattr(Img2Img, "simple_img2img", staticmethod(fake_simple_img2img))

    saved_calls = []

    class DummySaveImage:
        def save_images(self, images, filename_prefix="LD", prompt=None, extra_pnginfo=None, store_bytes_prefix=None):
            saved_calls.append((filename_prefix, prompt, len(images)))
            return {"ui": {"images": [{"filename": f"{filename_prefix}.png", "subfolder": "Classic"}]}}

    monkeypatch.setattr("src.FileManaging.ImageSaver.SaveImage", DummySaveImage)

    result = pipeline.run_batched(
        ctx,
        per_sample_info=[
            {"request_id": "req-1", "filename_prefix": "LD-REQ-req-1"},
            {"request_id": "req-2", "filename_prefix": "LD-REQ-req-2"},
        ],
    )

    assert captured["image_shape"] == (2, 64, 64, 3)
    assert captured["positive_batch_index"] == [[0], [1]]
    assert captured["negative_batch_index"] == [[0], [1]]
    assert captured["denoise"] == 0.6
    assert saved_calls == [
        ("LD-REQ-req-1", "prompt 1", 1),
        ("LD-REQ-req-2", "prompt 2", 1),
    ]
    assert set(result["batched_results"]) == {"req-1", "req-2"}
